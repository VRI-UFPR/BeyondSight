'''
https://github.com/DLR-RM/stable-baselines3/blob/237223f834fe9b8143ea24235d087c4e32addd2f/stable_baselines3/common/policies.py#L318
'''

# from typing import Any, Dict, List, Optional, Tuple, Union
from typing import Any, Dict, List, Optional, Tuple, Type, Union
# from functools import partial

import gym
import torch
from gym import spaces
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal
import numpy as np


# A schedule takes the remaining progress as input
# and ouputs a scalar (e.g. learning rate, clip range, ...)
# Schedule = Callable[[float], float]

#beyond stuff
import mapper
import model

def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.
    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor

class CategoricalDistribution():
    """
    Categorical distribution for discrete actions.
    :param action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int):
        super(CategoricalDistribution, self).__init__()
        self.distribution = None
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.
        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(self, action_logits: torch.Tensor) -> "CategoricalDistribution":
        self.distribution = Categorical(logits=action_logits)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()

    def sample(self) -> torch.Tensor:
        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        return torch.argmax(self.distribution.probs, dim=1)

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """
        Return actions according to the probability distribution.
        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()

    def actions_from_params(self, action_logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob

class TanhBijector(object):
    """
    Bijective transformation of a probability distribution
    using a squashing function (tanh)
    TODO: use Pyro instead (https://pyro.ai/)
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, epsilon: float = 1e-6):
        super(TanhBijector, self).__init__()
        self.epsilon = epsilon

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    @staticmethod
    def atanh(x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of Tanh
        Taken from Pyro: https://github.com/pyro-ppl/pyro
        0.5 * torch.log((1 + x ) / (1 - x))
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y: torch.Tensor) -> torch.Tensor:
        """
        Inverse tanh.
        :param y:
        :return:
        """
        eps = torch.finfo(y.dtype).eps
        # Clip the action to avoid NaN
        return TanhBijector.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))

    def log_prob_correction(self, x: torch.Tensor) -> torch.Tensor:
        # Squash correction (from original SAC implementation)
        return torch.log(1.0 - torch.tanh(x) ** 2 + self.epsilon)

class StateDependentNoiseDistribution():
    """
    Distribution class for using generalized State Dependent Exploration (gSDE).
    Paper: https://arxiv.org/abs/2005.05719
    It is used to create the noise exploration matrix and
    compute the log probability of an action with that noise.
    :param action_dim: Dimension of the action space.
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,)
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this ensures bounds are satisfied.
    :param learn_features: Whether to learn features for gSDE or not.
        This will enable gradients to be backpropagated through the features
        ``latent_sde`` in the code.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(
        self,
        action_dim: int,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        learn_features: bool = False,
        epsilon: float = 1e-6,
    ):
        super(StateDependentNoiseDistribution, self).__init__()
        self.distribution = None
        self.action_dim = action_dim
        self.latent_sde_dim = None
        self.mean_actions = None
        self.log_std = None
        self.weights_dist = None
        self.exploration_mat = None
        self.exploration_matrices = None
        self._latent_sde = None
        self.use_expln = use_expln
        self.full_std = full_std
        self.epsilon = epsilon
        self.learn_features = learn_features
        if squash_output:
            self.bijector = TanhBijector(epsilon)
        else:
            self.bijector = None

    def get_std(self, log_std: torch.Tensor) -> torch.Tensor:
        """
        Get the standard deviation from the learned parameter
        (log of it by default). This ensures that the std is positive.
        :param log_std:
        :return:
        """
        if self.use_expln:
            # From gSDE paper, it allows to keep variance
            # above zero and prevent it from growing too fast
            below_threshold = torch.exp(log_std) * (log_std <= 0)
            # Avoid NaN: zeros values that are below zero
            safe_log_std = log_std * (log_std > 0) + self.epsilon
            above_threshold = (torch.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            # Use normal exponential
            std = torch.exp(log_std)

        if self.full_std:
            return std
        # Reduce the number of parameters:
        return torch.ones(self.latent_sde_dim, self.action_dim).to(log_std.device) * std

    def sample_weights(self, log_std: torch.Tensor, batch_size: int = 1) -> None:
        """
        Sample weights for the noise exploration matrix,
        using a centered Gaussian distribution.
        :param log_std:
        :param batch_size:
        """
        std = self.get_std(log_std)
        self.weights_dist = Normal(torch.zeros_like(std), std)
        # Reparametrization trick to pass gradients
        self.exploration_mat = self.weights_dist.rsample()
        # Pre-compute matrices in case of parallel exploration
        self.exploration_matrices = self.weights_dist.rsample((batch_size,))

    def proba_distribution_net(
        self, latent_dim: int, log_std_init: float = -2.0, latent_sde_dim: Optional[int] = None
    ) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the deterministic action, the other parameter will be the
        standard deviation of the distribution that control the weights of the noise matrix.
        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :param latent_sde_dim: Dimension of the last layer of the features extractor
            for gSDE. By default, it is shared with the policy network.
        :return:
        """
        # Network for the deterministic action, it represents the mean of the distribution
        mean_actions_net = nn.Linear(latent_dim, self.action_dim)
        # When we learn features for the noise, the feature dimension
        # can be different between the policy and the noise network
        self.latent_sde_dim = latent_dim if latent_sde_dim is None else latent_sde_dim
        # Reduce the number of parameters if needed
        log_std = torch.ones(self.latent_sde_dim, self.action_dim) if self.full_std else torch.ones(self.latent_sde_dim, 1)
        # Transform it to a parameter so it can be optimized
        log_std = nn.Parameter(log_std * log_std_init, requires_grad=True)
        # Sample an exploration matrix
        self.sample_weights(log_std)
        return mean_actions_net, log_std

    def proba_distribution(
        self, mean_actions: torch.Tensor, log_std: torch.Tensor, latent_sde: torch.Tensor
    ) -> "StateDependentNoiseDistribution":
        """
        Create the distribution given its parameters (mean, std)
        :param mean_actions:
        :param log_std:
        :param latent_sde:
        :return:
        """
        # Stop gradient if we don't want to influence the features
        self._latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        variance = torch.mm(self._latent_sde ** 2, self.get_std(log_std) ** 2)
        self.distribution = Normal(mean_actions, torch.sqrt(variance + self.epsilon))
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.bijector is not None:
            gaussian_actions = self.bijector.inverse(actions)
        else:
            gaussian_actions = actions
        # log likelihood for a gaussian
        log_prob = self.distribution.log_prob(gaussian_actions)
        # Sum along action dim
        log_prob = sum_independent_dims(log_prob)

        if self.bijector is not None:
            # Squash correction (from original SAC implementation)
            log_prob -= torch.sum(self.bijector.log_prob_correction(gaussian_actions), dim=1)
        return log_prob

    def entropy(self) -> Optional[torch.Tensor]:
        if self.bijector is not None:
            # No analytical form,
            # entropy needs to be estimated using -log_prob.mean()
            return None
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> torch.Tensor:
        noise = self.get_noise(self._latent_sde)
        actions = self.distribution.mean + noise
        if self.bijector is not None:
            return self.bijector.forward(actions)
        return actions

    def mode(self) -> torch.Tensor:
        actions = self.distribution.mean
        if self.bijector is not None:
            return self.bijector.forward(actions)
        return actions

    def get_noise(self, latent_sde: torch.Tensor) -> torch.Tensor:
        latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        # Default case: only one exploration matrix
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return torch.mm(latent_sde, self.exploration_mat)
        # Use batch matrix multiplication for efficient computation
        # (batch_size, n_features) -> (batch_size, 1, n_features)
        latent_sde = latent_sde.unsqueeze(1)
        # (batch_size, 1, n_actions)
        noise = torch.bmm(latent_sde, self.exploration_matrices)
        return noise.squeeze(1)

    def actions_from_params(
        self, mean_actions: torch.Tensor, log_std: torch.Tensor, latent_sde: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std, latent_sde)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self, mean_actions: torch.Tensor, log_std: torch.Tensor, latent_sde: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(mean_actions, log_std, latent_sde)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """
        Return actions according to the probability distribution.
        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()

class DiagGaussianDistribution():
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.
    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int):
        super(DiagGaussianDistribution, self).__init__()
        self.distribution = None
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)
        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        # TODO: allow action dependent std
        log_std = nn.Parameter(torch.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_actions, log_std

    def proba_distribution(self, mean_actions: torch.Tensor, log_std: torch.Tensor) -> "DiagGaussianDistribution":
        """
        Create the distribution given its parameters (mean, std)
        :param mean_actions:
        :param log_std:
        :return:
        """
        action_std = torch.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.
        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> torch.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> torch.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        return self.distribution.mean

    def actions_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.
        :param mean_actions:
        :param log_std:
        :return:
        """
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob


    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """
        Return actions according to the probability distribution.
        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()

# class MlpExtractor(nn.Module):
#     '''
#     Placeholder class which will be fused with my model later
#
#     ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``
#     '''
#     def __init__(
#         self,
#         feature_dim: int,
#         device: Union[torch.device, str] = "auto",
#     ):
#         '''
#         2048+8
#         '''
#
#         last_layer_dim_pi = feature_dim
#         last_layer_dim_vf = feature_dim
#
#         # Save dim, used to create the distributions
#         self.latent_dim_pi = last_layer_dim_pi
#         self.latent_dim_vf = last_layer_dim_vf

class ActorCriticPolicy(nn.Module):
    '''
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    '''
    def __init__(
        self,
        action_space: gym.spaces.Space,
        lr_schedule: float = 2.5e-4,
        activation_fn: Type[nn.Module] = nn.Sigmoid,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,

        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,

        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,

        device: Union[torch.device, str] = "cuda",
        config = None,
    ):
        super(ActorCriticPolicy, self).__init__()
        # print("lr=",lr_schedule,flush=True)
        self.device = device
        self.config = config

        ########################################################################
        self.feature_net = model.Goal_prediction_network(input_shape=config.BEYOND.GLOBAL_POLICY.INPUT_SHAPE, hidden_size=config.BEYOND.GLOBAL_POLICY.HIDDEN_SIZE, embedding_size=config.BEYOND.GLOBAL_POLICY.EMBEDDING_SIZE)

        self.critic = model.CriticHead(self.feature_net.feature_out_size)

        self.latent_dim_pi = self.feature_net.feature_out_size
        ########################################################################


        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = None  # type: Optional[torch.optim.Optimizer]

        self.activation_fn = activation_fn
        self.ortho_init = ortho_init
        self.log_std_init = log_std_init

        self.features_dim = self.feature_net.feature_out_size

        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": sde_net_arch is not None,
            }

        self.sde_features_extractor = None
        self.sde_net_arch = sde_net_arch
        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self._build(lr_schedule)
    ############################################################################
    def load_actor_critic_weights(self, pretrained_dict) -> None:
        if(pretrained_dict):
            model_dict = self.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            '''
            Test keys are different
            '''
            keys_to_delete = []
            for k, v in pretrained_dict.items():
                if( pretrained_dict[k].size() != model_dict[k].size()):
                    # print(k,"are different. loaded",pretrained_dict[k].size(), "model uses", model_dict[k].size())
                    keys_to_delete.append(k)
            for k in keys_to_delete:
                del pretrained_dict[k]


            # 1. add new keys
            # fused_dicts = dict(d1, **d2)
            # fused_dicts.update(d3)

            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict

            self.load_state_dict(model_dict)
            # print("load_actor_critic_weights finished")
        else:
            # print("using random weights for the actor_critic")
            pass
        ########################################################################

    ############################################################################
    def load_feature_net_weights(self, pretrained_dict) -> None:
        if(pretrained_dict):
            model_dict = self.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            '''
            Test keys are different
            '''
            keys_to_delete = []
            for k, v in pretrained_dict.items():
                if not( 'feature_net' in k):
                    keys_to_delete.append(k)
                    # print("removing",k)
                else:
                    if( pretrained_dict[k].size() != model_dict[k].size()):
                        # print(k,"are different. loaded",pretrained_dict[k].size(), "model uses", model_dict[k].size())
                        keys_to_delete.append(k)
            for k in keys_to_delete:
                del pretrained_dict[k]


            # 1. add new keys
            # fused_dicts = dict(d1, **d2)
            # fused_dicts.update(d3)

            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict

            self.load_state_dict(model_dict)
            # print("load_actor_critic_weights finished")
        else:
            # print("using random weights for the actor_critic")
            pass
        ########################################################################
    ############################################################################
    # def load_feature_net_weights(self, checkpoint_path) -> None:
    #     print("model is in",self.device)
    #     if(checkpoint_path):
    #         checkpoint_filepath = checkpoint_path
    #         print("loading feature_net model...",checkpoint_filepath)
    #
    #         model_dict = self.feature_net.state_dict()
    #         pretrained_dict = torch.load(checkpoint_filepath)
    #
    #
    #         # 1. filter out unnecessary keys
    #         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #
    #         '''
    #         Test keys are different
    #         '''
    #         keys_to_delete = []
    #         for k, v in pretrained_dict.items():
    #             if not( 'feature_net' in k):
    #                 keys_to_delete.append(k)
    #                 print("removing",k)
    #             if( pretrained_dict[k].size() != model_dict[k].size()):
    #                 print(k,"are different. loaded",pretrained_dict[k].size(), "model uses", model_dict[k].size())
    #                 keys_to_delete.append(k)
    #         for k in keys_to_delete:
    #             del pretrained_dict[k]
    #
    #
    #         # 1. add new keys
    #         # fused_dicts = dict(d1, **d2)
    #         # fused_dicts.update(d3)
    #
    #         # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #
    #         # 2. overwrite entries in the existing state dict
    #         model_dict.update(pretrained_dict)
    #         # 3. load the new state dict
    #         self.feature_net.load_state_dict(model_dict)
    #     else:
    #         print("using random values for the feature_net")
    #     ########################################################################
    ############################################################################
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor, latent_sde: Optional[torch.Tensor] = None):
            """
            Retrieve action distribution given the latent codes.
            :param latent_pi: Latent code for the actor
            :param latent_sde: Latent code for the gSDE exploration function
            :return: Action distribution
            """
            mean_actions = self.action_net(latent_pi)

            if isinstance(self.action_dist, DiagGaussianDistribution):
                return self.action_dist.proba_distribution(mean_actions, self.log_std)
            elif isinstance(self.action_dist, StateDependentNoiseDistribution):
                return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
            elif isinstance(self.action_dist, CategoricalDistribution):
                # Here mean_actions are the logits before the softmax
                return self.action_dist.proba_distribution(action_logits=mean_actions)
            else:
                raise ValueError("Invalid action distribution")


    def _build(self, lr_schedule) -> None:
        """
        Create the networks and the optimizer.
        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        # self._build_mlp_extractor()
        latent_dim_pi = self.latent_dim_pi

        # Separate features extractor for gSDE
        if self.sde_net_arch is not None:
            self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                self.features_dim, self.sde_net_arch, self.activation_fn
            )

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_sde_dim = latent_dim_pi if self.sde_net_arch is None else latent_sde_dim
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_sde_dim, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)

        # value_net is the same as the CriticHead self.critic
        # self.value_net = nn.Linear(self.latent_dim_vf, 1)

        # # Init weights: use orthogonal initialization
        # # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            # module_gains = {
            #     # self.features_extractor: np.sqrt(2),
            #     # self.mlp_extractor: np.sqrt(2),
            #     self.action_net: 0.01,
            #     # self.value_net: 1,
            #     self.critic: 1,
            # }

            self.init_weights(self.action_net, 0.01)
            #self.critic is already init with gain 1 by default
            # self.init_weights(self.critic, 1)
            # for module, gain in module_gains.items():
            #     module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule, **self.optimizer_kwargs)

        # return
    ############################################################################
    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
    ############################################################################
    def create_sde_features_extractor(
        features_dim: int, sde_net_arch: List[int], activation_fn: Type[nn.Module]
    ) -> Tuple[nn.Sequential, int]:
        """
        Create the neural network that will be used to extract features
        for the gSDE exploration function.
        :param features_dim:
        :param sde_net_arch:
        :param activation_fn:
        :return:
        """
        # Special case: when using states as features (i.e. sde_net_arch is an empty list)
        # don't use any activation function
        sde_activation = activation_fn if len(sde_net_arch) > 0 else None
        latent_sde_net = create_mlp(features_dim, -1, sde_net_arch, activation_fn=sde_activation, squash_output=False)
        latent_sde_dim = sde_net_arch[-1] if len(sde_net_arch) > 0 else features_dim
        sde_features_extractor = nn.Sequential(*latent_sde_net)
        return sde_features_extractor, latent_sde_dim
################################################################################

def create_mlp(
    input_dim: int, output_dim: int, net_arch: List[int], activation_fn: Type[nn.Module] = nn.ReLU, squash_output: bool = False
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.
    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules

def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.
    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")

def make_proba_distribution(
    action_space: gym.spaces.Space, use_sde: bool = False, dist_kwargs: Optional[Dict[str, Any]] = None
):
    """
    Return an instance of Distribution for the correct type of action space
    :param action_space: the input action space
    :param use_sde: Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: Keyword arguments to pass to the probability distribution
    :return: the appropriate Distribution object
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    if isinstance(action_space, spaces.Box):
        assert len(action_space.shape) == 1, "Error: the action space must be a vector"
        cls = StateDependentNoiseDistribution if use_sde else DiagGaussianDistribution
        return cls(get_action_dim(action_space), **dist_kwargs)
    elif isinstance(action_space, spaces.Discrete):
        return CategoricalDistribution(action_space.n, **dist_kwargs)
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for action space"
            f"of type {type(action_space)}."
            " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
        )
