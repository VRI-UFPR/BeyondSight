'''
This code is intended to be as simple as possible implement the PPO rl learning using
an actor critic model with BEyond
'''

'''
    load libraries
'''
##########################################################################

#load pytorch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import os
import numpy as np

#logging
# from torch.utils.tensorboard import SummaryWriter
'''
A Wrapper for tensorboard SummaryWriter. It creates a dummy writer
when log_dir is empty string or None. It also has functionality that
generates tb video directly from numpy images.
'''
import contextlib
# from habitat.utils import profiling_wrapper
# from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from tensorboard_utils import TensorboardWriter
from datetime import datetime
import time

import signal
import sys

#params
import argparse

#habitat stuff
import habitat
# from habitat import Config
# from habitat import logger
from logging_mine import logger

from config_default import get_config, get_task_config

# from habitat_baselines.config.default import get_config
# from habitat.config.default import get_config
# from habitat import get_config as get_task_config

# from habitat.config.default import get_config
# from habitat import get_config as get_task_config
# from habitat.core.simulator import AgentState
#
# from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal, StopAction
# from habitat.utils.test_utils import sample_non_stop_action
#
# from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
# from habitat.datasets.utils import get_action_shortest_path
# from habitat.core.simulator import ShortestPathPoint
#
from torch.optim.lr_scheduler import LambdaLR
from torch import Tensor
# from habitat_baselines.common.tensor_dict import TensorDict
from tensor_dict import TensorDict
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, deque
# import numbers
from gym.spaces.box import Box
import copy

#mine stuff
from beyond_agent import BeyondAgent
from beyond_agent import batch_obs
from actor_critic import ActorCriticPolicy

# from habitat.core.greedyfollowerenv import GreedyFollowerEnv

'''
has a habitat dependency because of the ShortestPathFollower
'''
from greedyfollowerenv import GreedyFollowerEnv

####
from numba import njit

# from habitat_baselines.common.tensor_dict import TensorDict

EPS_PPO = 1e-5
##########################################################################
# run = True
# def sigterm_handler(_signo, _stack_frame):
#     # Raises SystemExit(0):
#     # sys.exit(0)
#     global run
#     run = False
#     # logger.info("interrupt catched...")
#
# signal.signal(signal.SIGINT, sigterm_handler)
# signal.signal(signal.SIGTERM, sigterm_handler)

def make_follower_env(config, dataset, rank: int = 0, is_train=True, internal_idx=0):
    r"""Constructor for default habitat Env.
    :param config: configurations for environment
    :param dataset: dataset for environment
    :param rank: rank for setting seeds for environment
    :return: constructed habitat Env
    """
    env = GreedyFollowerEnv(config=config, dataset=dataset, is_train=is_train, internal_idx=internal_idx)
    env.seed(config.SEED + rank)
    return env

class RolloutStorage:
    r"""Class for storing rollout information for RL trainers."""

    '''
    A buffer for storing inputs and outputs
    '''
    def __init__(
        self,
        numsteps,
        num_envs,
        sensors,
        recurrent_hidden_state_size,
        num_recurrent_layers,
    ):

        self.buffers = TensorDict()
        self.buffers["observations"] = TensorDict()

        for sensor in sensors:
            # print(sensor)
            self.buffers["observations"][sensor['name']] = torch.zeros(
                (
                    numsteps + 1,
                    num_envs,
                    *sensor['shape'],
                ),
                dtype=sensor['dtype'],
            )

        self.buffers["rewards"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["value_preds"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["returns"] = torch.zeros(numsteps + 1, num_envs, 1)

        self.buffers["action_log_probs"] = torch.zeros(
            numsteps + 1, num_envs, 1
        )

        self.buffers["masks"] = torch.zeros(
            numsteps + 1, num_envs, 1, dtype=torch.bool
        )

        #when coordinates
        self.buffers["actions"] = torch.zeros(
            numsteps + 1, num_envs, 2
        )

        #when discrete actions
        # self.buffers["actions"] = torch.zeros(
        #     numsteps + 1, num_envs, 1
        # )

        # self.buffers["main_inputs"] = torch.zeros(
        #     numsteps + 1, num_envs, 26,256,256
        # )

        self.buffers["main_inputs"] = torch.zeros(
            numsteps + 1, num_envs, 25,256,256
        )

        self.buffers["local_planner_actions"] = torch.zeros(
            numsteps + 1, num_envs, 1
        )

        self.buffers["recurrent_hidden_states"] = torch.zeros(
            numsteps + 1,
            num_envs,
            num_recurrent_layers,
            recurrent_hidden_state_size,
        )

        self._num_envs = num_envs
        self.numsteps = numsteps
        self.current_rollout_step_idx = 0

    def to(self, device):
        self.buffers.map_in_place(lambda v: v.to(device))

    def insert(
        self,
        next_observations=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        local_planner_actions=None,
        main_inputs=None,
        next_masks=None,
        next_recurrent_hidden_states=None
    ):
        # print("next_observations",next_observations,flush=True)
        # print("actions",actions,flush=True)
        # print("action_log_probs",action_log_probs,flush=True)
        # print("value_preds",value_preds,flush=True)
        # print("rewards",rewards,flush=True)
        # print("local_planner_actions",local_planner_actions,flush=True)

        env_slice = slice(
            int(0),
            int(self._num_envs),
        )

        next_step = dict(
            observations=next_observations,
            recurrent_hidden_states=next_recurrent_hidden_states,
            masks=next_masks,
        )

        current_step = dict(
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            rewards=rewards,
            local_planner_actions=local_planner_actions,
            main_inputs=main_inputs,
        )

        next_step = {k: v for k, v in next_step.items() if v is not None}
        current_step = {k: v for k, v in current_step.items() if v is not None}

        if len(next_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idx + 1, env_slice),
                next_step,
                strict=False,
            )

        if len(current_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idx, env_slice),
                current_step,
                strict=False,
            )

    def advance_rollout(self):
        self.current_rollout_step_idx += 1

    def after_update(self):
        self.buffers[0] = self.buffers[self.current_rollout_step_idx]
        self.current_rollout_step_idx = 0

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.buffers["value_preds"][
                self.current_rollout_step_idx
            ] = next_value
            gae = 0
            for step in reversed(range(self.current_rollout_step_idx)):
                delta = (
                    self.buffers["rewards"][step]
                    + gamma
                    * self.buffers["value_preds"][step + 1]
                    * self.buffers["masks"][step + 1]
                    - self.buffers["value_preds"][step]
                )
                gae = (
                    delta + gamma * tau * gae * self.buffers["masks"][step + 1]
                )
                self.buffers["returns"][step] = (
                    gae + self.buffers["value_preds"][step]
                )
        else:
            self.buffers["returns"][self.current_rollout_step_idx] = next_value
            for step in reversed(range(self.current_rollout_step_idx)):
                self.buffers["returns"][step] = (
                    gamma
                    * self.buffers["returns"][step + 1]
                    * self.buffers["masks"][step + 1]
                    + self.buffers["rewards"][step]
                )

    # def cleanup(self):
    #     for i in range(0,self.current_rollout_step_idx):
    #         del self.buffers['observations']['rgb']
    #         del self.buffers['observations']['depth']
    #         del self.buffers['observations']['semantic']
    #         del self.buffers['observations']['compass']
    #         del self.buffers['observations']['gps']
    #         del self.buffers['observations']['oracle']

    def recurrent_generator(self, advantages, num_mini_batch):
        num_environments = advantages.size(1)
        assert num_environments >= num_mini_batch, (
            "Trainer requires the number of environments ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(
                num_environments, num_mini_batch
            )
        )
        if num_environments % num_mini_batch != 0:
            warnings.warn(
                "Number of environments ({}) is not a multiple of the"
                " number of mini batches ({}).  This results in mini batches"
                " of different sizes, which can harm training performance.".format(
                    num_environments, num_mini_batch
                )
            )
        batches = []
        for inds in torch.randperm(num_environments).chunk(num_mini_batch):
            batch = self.buffers[0 : self.current_rollout_step_idx, inds]
            # del batch['observations']['rgb']
            # del batch['observations']['depth']
            # del batch['observations']['semantic']
            # del batch['observations']['compass']
            del batch['observations']['gps']
            # del batch['observations']['oracle']

            batch["advantages"] = advantages[
                0 : self.current_rollout_step_idx, inds
            ]
            batch["recurrent_hidden_states"] = batch[
                "recurrent_hidden_states"
            ][0:1]

            # print("batch",batch)

            '''
            batch.map(lambda v: v.flatten(0, 1))
            creates a copy of the tensor
            doubling mememory usage which is not
            ideal
            since this data wont be reused after the update
            '''

            for key in batch:
                if key == 'observations':
                    for key2 in batch['observations']:
                        batch['observations'][key2] = batch['observations'][key2].flatten(0,1)
                else:
                    batch[key] = batch[key].flatten(0,1)

            batches.append( batch )
            # batches.append( batch.map(lambda v: v.view(0, 1)) )


            # yield batch.map(lambda v: v.flatten(0, 1))
        return batches

class PPO:
    '''
    Proximal Policy Optimization algorithm (PPO) (clip version)
    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)
    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html
    '''
    def __init__(self, config):
        self.config = config
        ##################################################################

        self.device = config.BEYOND_TRAINER.DEVICE
        print("DEVICE=",self.device)
        self.batch_size = config.BEYOND_TRAINER.BATCH_SIZE if config.BEYOND_TRAINER.TRAIN_MODE else config.BEYOND_TRAINER.EVAL_BATCH_SIZE
        print("self.batch_size",self.batch_size)
        # self.batch_size = config.BEYOND_TRAINER.BATCH_SIZE

        self.train_mode = True if config.BEYOND_TRAINER.TRAIN_MODE else False
        self.rollout_size = config.BEYOND_TRAINER.ROLLOUT_SIZE
        self.rollouts = None

        self.RESUME_TRAIN = config.BEYOND_TRAINER.RESUME_TRAIN
        self.RESUME_TRAIN_CHEKPOINT = config.BEYOND_TRAINER.RESUME_TRAIN_CHEKPOINT

        self.configs = None
        self.datasets = None
        self.num_envs = None
        self.envs = None

        self.agent = BeyondAgent(
            device=self.device,
            config=self.config,
            batch_size=self.batch_size,
        )

        self.big_success = 0
        self.big_reward = 0

        self.flush_secs = 30

        self.LOG_INTERVAL = config.BEYOND_TRAINER.LOG_INTERVAL

        self.NUM_UPDATES=config.BEYOND_TRAINER.NUM_UPDATES
        self.ppo_epoch = self.config.BEYOND_TRAINER.PPO.ppo_epoch
        self.num_mini_batch = min(self.batch_size, self.config.BEYOND_TRAINER.PPO.num_mini_batch)
        # self.num_mini_batch = max(self.batch_size, self.config.BEYOND_TRAINER.PPO.num_mini_batch)
        self.num_steps_done=0
        self.num_updates_done=0
        self._last_checkpoint_percent=0

        self.use_linear_lr_decay = self.config.BEYOND_TRAINER.PPO.use_linear_lr_decay
        self.use_clipped_value_loss = True
        self.use_normalized_advantage = self.config.BEYOND_TRAINER.PPO.use_normalized_advantage

        self.clip_param = self.config.BEYOND_TRAINER.PPO.clip_param
        self.value_loss_coef = self.config.BEYOND_TRAINER.PPO.value_loss_coef
        self.entropy_coef = self.config.BEYOND_TRAINER.PPO.entropy_coef
        self.max_grad_norm = self.config.BEYOND_TRAINER.PPO.max_grad_norm
        self.reward_window_size = self.config.BEYOND_TRAINER.PPO.reward_window_size


        # print("self.config.NUM_CHECKPOINTS",self.config.NUM_CHECKPOINTS)
        # print("self.config.CHECKPOINT_INTERVAL",self.config.CHECKPOINT_INTERVAL)
        # print("self.config.CHECKPOINT_FOLDER",self.config.CHECKPOINT_FOLDER)

        self.NUM_CHECKPOINTS = self.config.BEYOND_TRAINER.NUM_CHECKPOINTS
        self.CHECKPOINT_INTERVAL = self.config.BEYOND_TRAINER.CHECKPOINT_INTERVAL
        # self.CHECKPOINT_FOLDER = self.config.BEYOND_TRAINER.CHECKPOINT_FOLDER


        ########################################################################
        '''
            Logging stuff
        '''
        now = datetime.now()

        if config.BEYOND_TRAINER.TRAIN_MODE:
            EXPERIMENT_NAME = config.BEYOND_TRAINER.EXPERIMENT_NAME
            if EXPERIMENT_NAME!="":
                self.LOG_DIR = config.BEYOND_TRAINER.LOG_DIR+EXPERIMENT_NAME+now.strftime("%d-%m-%Y_%H-%M-%S")+"/"
            else:
                self.LOG_DIR = config.BEYOND_TRAINER.LOG_DIR+now.strftime("%d-%m-%Y_%H-%M-%S")+"/"
            if not os.path.exists(self.LOG_DIR):
                os.makedirs(self.LOG_DIR)


            self.CHECKPOINT_FOLDER = config.BEYOND_TRAINER.CHECKPOINT_FOLDER
            if EXPERIMENT_NAME!="":
                self.CHECKPOINT_FOLDER = self.CHECKPOINT_FOLDER+EXPERIMENT_NAME+now.strftime("%d-%m-%Y_%H-%M-%S")+"/"
            else:
                self.CHECKPOINT_FOLDER = self.CHECKPOINT_FOLDER+now.strftime("%d-%m-%Y_%H-%M-%S")+"/"
            if not os.path.exists(self.CHECKPOINT_FOLDER):
                os.makedirs(self.CHECKPOINT_FOLDER)

        ########################################################################



        '''
        NEURAL-SLAM Hyperparameters
        ## Global Policy RL PPO Hyperparameters
        parser.add_argument('--global_lr', type=float, default=2.5e-5,
                            help='global learning rate (default: 2.5e-5)')
        parser.add_argument('--global_hidden_size', type=int, default=256,
                            help='local_hidden_size')
        parser.add_argument('--eps', type=float, default=1e-5,
                            help='RL Optimizer epsilon (default: 1e-5)')
        parser.add_argument('--alpha', type=float, default=0.99,
                            help='RL Optimizer alpha (default: 0.99)')
        parser.add_argument('--gamma', type=float, default=0.99,
                            help='discount factor for rewards (default: 0.99)')
        parser.add_argument('--use_gae', action='store_true', default=False,
                            help='use generalized advantage estimation')
        parser.add_argument('--tau', type=float, default=0.95,
                            help='gae parameter (default: 0.95)')
        parser.add_argument('--entropy_coef', type=float, default=0.001,
                            help='entropy term coefficient (default: 0.01)')
        parser.add_argument('--value_loss_coef', type=float, default=0.5,
                            help='value loss coefficient (default: 0.5)')
        parser.add_argument('--max_grad_norm', type=float, default=0.5,
                            help='max norm of gradients (default: 0.5)')
        parser.add_argument('--num_global_steps', type=int, default=40,
                            help='number of forward steps in A2C (default: 5)')
        parser.add_argument('--ppo_epoch', type=int, default=4,
                            help='number of ppo epochs (default: 4)')
        parser.add_argument('--num_mini_batch', type=str, default="auto",
                            help='number of batches for ppo (default: 32)')
        parser.add_argument('--clip_param', type=float, default=0.2,
                            help='ppo clip parameter (default: 0.2)')
        parser.add_argument('--use_recurrent_global', type=int, default=0,
                            help='use a recurrent global policy')
        '''

        # self._last_checkpoint_percent
        # self.num_steps_done

        # if self.config.NUM_CHECKPOINTS != -1:
        #     checkpoint_every = 1 / self.config.NUM_CHECKPOINTS
        #     if (
        #         self._last_checkpoint_percent + checkpoint_every
        #         < self.percent_done()
        #     ):
        #         needs_checkpoint = True
        #         self._last_checkpoint_percent = self.percent_done()
        # else:
        #     needs_checkpoint = (
        #         self.num_steps_done % self.config.CHECKPOINT_INTERVAL
        #     ) == 0

        # self.agent.to(self.device)

        # clip_param: 0.2
        # ppo_epoch: 4
        # num_mini_batch: 2
        # value_loss_coef: 0.5
        # entropy_coef: 0.01
        # lr: 2.5e-4
        # eps: 1e-5
        # max_grad_norm: 0.2
        # num_steps: 64
        # use_gae: True
        # gamma: 0.99
        # tau: 0.95
        # use_linear_clip_decay: False
        # use_linear_lr_decay: False
        # reward_window_size: 50
        # use_normalized_advantage: False

        #
        # self.clip_param = clip_param
        # self.ppo_epoch = ppo_epoch
        # self.num_mini_batch = num_mini_batch
        #
        # self.value_loss_coef = value_loss_coef
        # self.entropy_coef = entropy_coef
        #
        # self.max_grad_norm = max_grad_norm
        # self.use_clipped_value_loss = use_clipped_value_loss
        #
        # self.optimizer = optim.Adam(
        #     list(filter(lambda p: p.requires_grad, actor_critic.parameters())),
        #     lr=lr,
        #     eps=eps,
        # )
        # self.use_normalized_advantage = use_normalized_advantage


    ############################################################################
    def send_computed_reward(self, reward):
        # print("reward",reward)
        # function_args_list=[{"reward":i} for i in reward]
        results = self.envs.call(function_names=["send_computed_reward" for _ in range(self.num_envs)],function_args_list=[{"reward":i} for i in reward])
        return results
    ############################################################################
    def percent_done(self) -> float:
        if self.NUM_UPDATES != -1:
            return self.num_updates_done / self.NUM_UPDATES
        else:
            return self.num_steps_done / self.config.TOTAL_NUM_STEPS
    ############################################################################
    def is_done(self) -> bool:
        return self.percent_done() >= 1.0
    ############################################################################
    def should_checkpoint(self) -> bool:
        needs_checkpoint = False
        if self.NUM_CHECKPOINTS != -1:
            checkpoint_every = 1 / self.NUM_CHECKPOINTS
            if (
                self._last_checkpoint_percent + checkpoint_every
                < self.percent_done()
            ):
                needs_checkpoint = True
                self._last_checkpoint_percent = self.percent_done()
        else:
            needs_checkpoint = (
                self.num_steps_done % self.CHECKPOINT_INTERVAL
            ) == 0

        return needs_checkpoint
    ############################################################################
    # def save_checkpoint(
    #     self, file_name: str, extra_state: Optional[Dict] = None
    # ) -> None:
    #     r"""Save checkpoint with specified name.
    #     Args:
    #         file_name: file name for checkpoint
    #     Returns:
    #         None
    #     """
    #     checkpoint = {
    #         "state_dict": self.agent.actor_critic.state_dict(),
    #         "config": self.config,
    #     }
    #     if extra_state is not None:
    #         checkpoint["extra_state"] = extra_state
    #
    #     torch.save(
    #         checkpoint, os.path.join(self.CHECKPOINT_FOLDER, file_name)
    #     )
    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.
        Args:
            file_name: file name for checkpoint
        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.actor_critic.state_dict(),
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.CHECKPOINT_FOLDER, file_name)
        )
    ############################################################################
    def get_advantages(self, rollouts: RolloutStorage) -> Tensor:
        advantages = (
            rollouts.buffers["returns"][:-1]
            - rollouts.buffers["value_preds"][:-1]
        )
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)
    ############################################################################
    def _evaluate_actions(
        self, observations, main_inputs, action, recurrent_hidden_states, masks
    ):
        r"""Internal method that calls Policy.evaluate_actions.  This is used instead of calling
        that directly so that that call can be overrided with inheritence
        """
        return self.agent.evaluate_actions(
            observations, main_inputs, action, recurrent_hidden_states, masks
        )
    ############################################################################
    def before_backward(self, loss: Tensor) -> None:
        pass

    def after_backward(self, loss: Tensor) -> None:
        pass

    def before_step(self) -> None:
        nn.utils.clip_grad_norm_(
            self.agent.actor_critic.parameters(), self.max_grad_norm
        )

    def after_step(self) -> None:
        pass
    ############################################################################
    def update(self, rollouts: RolloutStorage) -> Tuple[float, float, float]:
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0

        # print("before loop ppo_epoch")
        # rollouts.cleanup()
        # print("after cleanup")
        for _e in range(self.ppo_epoch):
            # profiling_wrapper.range_push("PPO.update epoch")
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )
            # print("created data_generator iterator")
            # print("data_generator",data_generator)
            # print("EXITING")
            # exit()

            for batch in data_generator:
                # print("batch['observations']['objectgoal'].shape",batch["observations"]['objectgoal'].shape)
                # print("batch['main_inputs'].shape",batch["main_inputs"].shape)
                # print("batch['actions'].shape",batch["actions"].shape)
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                ) = self._evaluate_actions(
                    batch["observations"],
                    batch["main_inputs"],
                    batch["actions"],
                    batch["recurrent_hidden_states"],
                    batch["masks"],
                )
                # print("inside data_generator loop")

                ratio = torch.exp(action_log_probs - batch["action_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * batch["advantages"]
                )
                action_loss = -(torch.min(surr1, surr2).mean())

                if self.use_clipped_value_loss:
                    value_pred_clipped = batch["value_preds"] + (
                        values - batch["value_preds"]
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - batch["returns"]).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - batch["returns"]
                    ).pow(2)
                    value_loss = 0.5 * torch.max(
                        value_losses, value_losses_clipped
                    )
                else:
                    value_loss = 0.5 * (batch["returns"] - values).pow(2)

                value_loss = value_loss.mean()
                dist_entropy = dist_entropy.mean()

                self.agent.actor_critic.optimizer.zero_grad()
                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                )

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step()
                self.agent.actor_critic.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

            # profiling_wrapper.range_pop()  # PPO.update epoch

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
    ############################################################################
    def _compute_actions(self):
        '''
        Pull from the rollout, act, and save agent output to rollouts
        '''
        num_envs = self.num_envs
        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():
            step_batch = self.rollouts.buffers["observations"][self.rollouts.current_rollout_step_idx]

            # profiling_wrapper.range_push("compute actions")

            (
                values,
                actions,
                actions_log_probs,
                local_planner_actions,
                main_inputs,
                computed_reward,
                rnn_states
            ) = self.agent.act_with_batch(
                step_batch,
            )
            # print("values",values)
            # print("actions",actions)
            # print("actions_log_probs",actions_log_probs)
            # print("local_planner_actions",local_planner_actions)

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            local_planner_actions = local_planner_actions.to(device="cpu")
            self.pth_time += time.time() - t_sample_action

            # profiling_wrapper.range_pop()  # compute actions

            t_step_env = time.time()

            ####################################################################
            # print("computed_reward",computed_reward,flush=True)
            results = self.send_computed_reward(computed_reward)
            # print("results",results,flush=True)

            # print("EXITING",flush=True)
            # exit()
            for index_env, act in zip(
                range(0, num_envs), local_planner_actions.unbind(0)
            ):
                self.envs.async_step_at(index_env, act.item())
            ####################################################################

            self.env_time += time.time() - t_step_env

            ####################################################################
            self.rollouts.insert(
                actions=actions,
                action_log_probs=actions_log_probs,
                value_preds=values,
                local_planner_actions=local_planner_actions,
                main_inputs=main_inputs,
                next_recurrent_hidden_states=rnn_states,
            )
            ####################################################################
            return

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}
    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _do_step_and_collect(self):
        '''
        Wait the async step then collect new obs and save obs to rollouts
        '''
        num_envs = self.num_envs
        env_slice = slice(
            int(0),
            int(num_envs),
        )
        ########################################################################
        t_step_env = time.time()

        outputs = [
            self.envs.wait_step_at(index_env)
            for index_env in range(env_slice.start, env_slice.stop)
        ]

        observations, rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]

        self.env_time += time.time() - t_step_env
        ########################################################################
        t_update_stats = time.time()
        batch = batch_obs(
            observations, device=self.device
        )

        rewards = torch.tensor(
            rewards_l,
            dtype=torch.float,
            device=self.current_episode_reward.device,
        )
        rewards = rewards.unsqueeze(1)

        not_done_masks = torch.tensor(
            [[not done] for done in dones],
            dtype=torch.bool,
            device=self.current_episode_reward.device,
        )
        done_masks = torch.logical_not(not_done_masks)

        for i,done in enumerate(dones):
            if(done):
                self.agent.reset(i)

        ########################################################################
        self.current_episode_reward[env_slice] += rewards
        current_ep_reward = self.current_episode_reward[env_slice]
        self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # type: ignore
        self.running_episode_stats["count"][env_slice] += done_masks.float()  # type: ignore
        for k, v_k in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v_k,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            ).unsqueeze(1)
            if k not in self.running_episode_stats:
                self.running_episode_stats[k] = torch.zeros_like(
                    self.running_episode_stats["count"]
                )

            self.running_episode_stats[k][env_slice] += v.where(done_masks, v.new_zeros(()))  # type: ignore

        self.current_episode_reward[env_slice].masked_fill_(done_masks, 0.0)
        ########################################################################
        self.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            next_masks=not_done_masks,
        )

        self.rollouts.advance_rollout()
        #######################################################################
        self.pth_time += time.time() - t_update_stats

        return env_slice.stop - env_slice.start
    ############################################################################
    def _load_environment_data(self, gpu2gpu=True, train_set_idx=0, dataset_mode="TRAIN"):
        configs = []
        datasets = []

        config = self.config

        for i in range(0,self.batch_size):
            config.defrost()
            mode = getattr(config.BEYOND_TRAINER.ENVIRONMENT_LIST,dataset_mode)
            # config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True if dataset_mode == "TRAIN" else False
            config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True
            config.DATASET.TYPE = config.BEYOND_TRAINER.ENVIRONMENT_LIST.TRAIN.TYPE
            config.DATASET.SPLIT = mode.NAMES[train_set_idx]
            # config.DATASET.SPLIT = "ac26ZMwG7aT"
            config.DATASET.DATA_PATH = mode.PATH+"{split}/{split}.json.gz"
            # config.DATASET.DATA_PATH = mode.PATH+"lv0_train/content/{split}.json.gz"
            if "habitat-challenge-data" not in config.DATASET.SCENES_DIR:
                config.DATASET.SCENES_DIR = config.DATASET.DATA_PATH.split("/")[0]+"/"+config.DATASET.SCENES_DIR
            config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = gpu2gpu
            config.freeze()

            print(config.DATASET.SPLIT, config.DATASET.DATA_PATH, config.DATASET)

            datasets.append(
                habitat.make_dataset(
                    id_dataset=config.DATASET.TYPE, config=config.DATASET
                )
            )
            np.random.shuffle(datasets[i].episodes)
            configs.append(config)

        return configs, datasets
    ############################################################################
    def setup_envs(self):
        self.configs, self.datasets = self._load_environment_data()

        num_envs = len(self.configs)
        self.num_envs = num_envs

        env_fn_args = tuple(zip(self.configs, self.datasets, range(num_envs), [self.train_mode for tmp in range(num_envs)], [i for i in range(num_envs)] ))
        ########################################################################

        multiprocessing_start_method="forkserver"

        self.envs = habitat.VectorEnv(
            make_env_fn=make_follower_env,
            env_fn_args=env_fn_args,
            multiprocessing_start_method=multiprocessing_start_method,
        )
        ########################################################################
        ########################################################################

        observations = self.envs.reset()
        for i in range(num_envs):
            self.agent.reset(i)
        # print("observations=\n",observations)

        batch = batch_obs(observations, device=self.device)

        sensors = []
        for sensor_name in batch:
            sensor_dict = {'name':None,'shape':None,'dtype':None}
            sensor_dict['name'] = sensor_name
            sensor_dict['shape'] = batch[sensor_name][0].shape
            sensor_dict['dtype'] = batch[sensor_name][0].dtype
            sensors.append(sensor_dict)

        # print("sensors=\n",sensors)
        # print("batch=\n",batch)

        '''
        store inital obs in rollout
        '''
        # print("self.envs.action_spaces[0]",self.envs.action_spaces[0])

        self.rollouts = RolloutStorage(
                            numsteps=self.rollout_size,
                            num_envs=self.num_envs,
                            sensors=sensors,
                            recurrent_hidden_state_size=self.agent.actor_critic.feature_net.feature_out_size,
                            num_recurrent_layers=self.agent.actor_critic.feature_net.rnn_layers
                        )

        self.rollouts.to(self.device)

        # print("self.rollout=\n",self.rollout)

        ########################################################################
        #STEP X BATCH X FEATURES
        self.rollouts.buffers["observations"][0] = batch

        self.current_episode_reward = torch.zeros(self.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.num_envs, 1),
            reward=torch.zeros(self.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=self.reward_window_size)
        )



        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()
    ############################################################################
    def _update_agent(self):
        ppo_cfg = self.config.BEYOND_TRAINER.PPO
        t_update_model = time.time()
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idx
            ]

            next_value = self.agent.get_value(
                step_batch["observations"],
                step_batch["main_inputs"],
                step_batch["recurrent_hidden_states"],
                step_batch["masks"],
            )

        self.rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        self.agent.train()

        # print("before self.update")
        value_loss, action_loss, dist_entropy = self.update(
            self.rollouts
        )
        # print("after self.update")

        self.rollouts.after_update()
        self.pth_time += time.time() - t_update_model

        return (
            value_loss,
            action_loss,
            dist_entropy,
        )
    ############################################################################
    def _coalesce_post_step(
        self, losses: Dict[str, float], count_steps_delta: int
    ) -> Dict[str, float]:
        stats_ordering = sorted(self.running_episode_stats.keys())
        stats = torch.stack(
            [self.running_episode_stats[k] for k in stats_ordering], 0
        )

        # stats = self._all_reduce(stats)

        for i, k in enumerate(stats_ordering):
            self.window_episode_stats[k].append(stats[i])

        # if self._is_distributed:
        #     loss_name_ordering = sorted(losses.keys())
        #     stats = torch.tensor(
        #         [losses[k] for k in loss_name_ordering] + [count_steps_delta],
        #         device="cpu",
        #         dtype=torch.float32,
        #     )
        #     stats = self._all_reduce(stats)
        #     count_steps_delta = int(stats[-1].item())
        #     stats /= torch.distributed.get_world_size()
        #
        #     losses = {
        #         k: stats[i].item() for i, k in enumerate(loss_name_ordering)
        #     }

        loss_name_ordering = sorted(losses.keys())
        stats = torch.tensor(
            [losses[k] for k in loss_name_ordering] + [count_steps_delta],
            device="cpu",
            dtype=torch.float32,
        )
        # stats = self._all_reduce(stats)
        count_steps_delta = int(stats[-1].item())
        # stats /= torch.distributed.get_world_size()

        losses = {
            k: stats[i].item() for i, k in enumerate(loss_name_ordering)
        }

        # if self._is_distributed and rank0_only():
        #     self.num_rollouts_done_store.set("num_done", "0")

        self.num_steps_done += count_steps_delta

        return losses
    ############################################################################
    def _training_log(
        self, writer, losses: Dict[str, float], prev_time: int = 0
    ):
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()
                if len(v) > 1
                else v[0].sum().item()
            )
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        writer.add_scalar(
            "reward",
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }
        if len(metrics) > 0:
            writer.add_scalars("metrics", metrics, self.num_steps_done)

        writer.add_scalars(
            "losses",
            losses,
            self.num_steps_done,
        )

        # log stats
        if self.num_updates_done % self.LOG_INTERVAL == 0:
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    self.num_steps_done
                    / ((time.time() - self.t_start) + prev_time),
                )
            )

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {}".format(
                    self.num_updates_done,
                    self.env_time,
                    self.pth_time,
                    self.num_steps_done,
                )
            )

            logger.info(
                "Average window size: {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                )
            )
    ############################################################################
    # def _check_interrupt(self):
    #     global run
    #     if(not run):
    #         #handle interrupt
    #         logger.info("interrupted saving checkpoint...")
    #         self.save_checkpoint(
    #             f"ckpt.interrupted.pth",
    #             dict(
    #                 step=self.num_steps_done,
    #                 wall_time=(time.time() - self.t_start) + prev_time,
    #                 smallest_value_loss=self.smallest_value_loss,
    #                 smallest_action_loss=self.smallest_action_loss,
    #             ),
    #         )
    #         sys.exit(0)

    def train(self):
        # global run

        # lr_scheduler = LambdaLR(
        #     optimizer=self.agent.actor_critic.optimizer,
        #     lr_lambda=lambda x: 1 - self.percent_done(),
        # )

        ########################################################################
        '''
        load interrupted state
        '''
        if(self.RESUME_TRAIN):
            print("RESUMING TRAINING FROM", self.RESUME_TRAIN_CHEKPOINT)
            # Map location CPU is almost always better than mapping to a CUDA device.
            # Models will be loaded Separate later on
            ckpt_dict = self.load_checkpoint(self.RESUME_TRAIN_CHEKPOINT, map_location="cpu")
            # print("ckpt_dict",ckpt_dict,flush=True)

            # self.agent.actor_critic.load_state_dict(ckpt_dict["state_dict"])
            if(self.config.BEYOND_TRAINER.RESUME_ONLY_FEATURE_NET):
                print("load_feature_net_weights")
                self.agent.actor_critic.load_feature_net_weights(ckpt_dict["state_dict"])
            else:
                print("load_actor_critic_weights")
                self.agent.actor_critic.load_actor_critic_weights(ckpt_dict["state_dict"])


            del ckpt_dict
            print("",flush=True)
        ########################################################################

        '''
        Setup logging

        Load weights if resuming

        Perform train_step
        '''
        ########################################################################
        self.setup_envs()
        ########################################################################

        count_checkpoints = 0
        prev_time = 0



        # print("signals set",flush=True)
        ########################################################################
        with (
            TensorboardWriter(
                self.LOG_DIR, flush_secs=self.flush_secs
            )
        ) as writer:
            while (not self.is_done()):
                self.agent.eval()
                self.agent.planner.eval()
                count_steps_delta = 0

                '''
                Then start rollout buffer loop,
                '''
                for i in range(self.rollout_size):
                    self._compute_actions()
                    count_steps_delta+=self._do_step_and_collect()
                ################################################################

                ################################################################
                '''
                Save model with best success
                '''
                if (self.num_updates_done > 64):
                    deltas = {
                        k: (
                            (v[-1] - v[0]).sum().item()
                            if len(v) > 1
                            else v[0].sum().item()
                        )
                        for k, v in self.window_episode_stats.items()
                    }
                    deltas["count"] = max(deltas["count"], 1.0)
                    # Check to see if there are any metrics
                    # that haven't been logged yet
                    metrics = {
                        k: v / deltas["count"]
                        for k, v in deltas.items()
                        if k not in {"reward", "count"}
                    }

                    if((metrics['success']) > (self.big_success) ):
                        self.big_success = metrics['success']
                        logger.info(
                            "saving with success: {:.3f}\t saving with ep_reward: {:.3f}\t".format(
                                self.big_success,
                                metrics['ep_reward'],
                            )
                        )

                        self.save_checkpoint(
                            f"ckpt.best_success.pth",
                            dict(
                                step=self.num_steps_done,
                                wall_time=(time.time() - self.t_start) + prev_time,
                                success=self.big_success,
                                reward=metrics['ep_reward'],
                            ),
                        )
                    if((metrics['ep_reward']) > (self.big_reward) ):
                        self.big_reward = metrics['ep_reward']
                        logger.info(
                            "saving with success: {:.3f}\t saving with ep_reward: {:.3f}\t".format(
                                metrics['success'],
                                self.big_reward,
                            )
                        )

                        self.save_checkpoint(
                            f"ckpt.best_reward.pth",
                            dict(
                                step=self.num_steps_done,
                                wall_time=(time.time() - self.t_start) + prev_time,
                                success=metrics['success'],
                                reward=self.big_reward,
                            ),
                        )
                ################################################################

                (
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent()

                # if self.use_linear_lr_decay:
                #     lr_scheduler.step()  # type: ignore

                self.num_updates_done += 1

                losses = self._coalesce_post_step(
                    dict(value_loss=value_loss, action_loss=action_loss),
                    count_steps_delta,
                )


                self._training_log(writer, losses, prev_time)

                # checkpoint model
                if self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    count_checkpoints += 1
        ########################################################################
        self.save_checkpoint(
            f"ckpt.last.pth",
            dict(
                step=self.num_steps_done,
                wall_time=(time.time() - self.t_start) + prev_time,
            ),
        )
        print("---------------------------------------------------------------")
        print("End of train")
        print("---------------------------------------------------------------")
        self.envs.close()
        torch.cuda.empty_cache()
        return
    ############################################################################
    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.
        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args
        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)
    ############################################################################
    def eval_checkpoint(
        self,
        checkpoint_path: str,
        num_episodes = None,
        # writer: TensorboardWriter,
        # checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.
        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging
        Returns:
            None
        """

        # Map location CPU is almost always better than mapping to a CUDA device.
        # Models will be loaded Separate later on
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        # print("ckpt_dict",ckpt_dict)

        #use the config saved
        # self.config = ckpt_dict["config"].clone()

        self.config.defrost()
        self.config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = True
        self.config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False

        self.config.DATASET.SPLIT= self.config.BEYOND_TRAINER.ENVIRONMENT_LIST.EVAL.NAMES[0]
        self.config.DATASET.DATA_PATH= self.config.BEYOND_TRAINER.ENVIRONMENT_LIST.EVAL.PATH+"/{split}/{split}.json.gz"

        # self.config.TASK.MEASUREMENTS.append('TOP_DOWN_MAP')

        self.config.freeze()

        print(self.config.DATASET.SPLIT, self.config.DATASET.DATA_PATH, self.config.DATASET)

        if(self.config.BEYOND_TRAINER.RESUME_ONLY_FEATURE_NET):
            print("load_feature_net_weights")
            self.agent.actor_critic.load_feature_net_weights(ckpt_dict["state_dict"])
        else:
            print("load_actor_critic_weights")
            self.agent.actor_critic.load_actor_critic_weights(ckpt_dict["state_dict"])

        # self.agent.actor_critic.load_state_dict(ckpt_dict["state_dict"])
        self.agent.eval()
        self.agent.planner.eval()

        ########################################################################
        self._env = GreedyFollowerEnv(config=self.config)

        if num_episodes is None:
            num_episodes = len(self._env.episodes)
        else:
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(
                    num_episodes, len(self._env.episodes)
                )
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        count_episodes = 0
        infos = []
        steps = max(int(num_episodes/100),1)
        print("steps",steps)
        ########################################################################
        while count_episodes < num_episodes:
            self.agent.reset(0)
            observations = self._env.reset()

            while not self._env.episode_over:
                action = self.agent.act(observations)
                # observations = self._env.step_obs_only(action)
                observations, reward, done, info = self._env.step(action)

            # metrics = self._env.get_metrics_extended()
            metrics = info
            print("metrics=\n",metrics,"\n\n")

            if(not infos):
                infos = [metrics]
            else:
                infos.append(metrics)

            ####################################################################
            if(count_episodes%steps==0):
                print("count_episodes",count_episodes)
                ################################################################
                # batch = {'distance_to_goal':[],'success':[],'softspl':[],'spl':[]}
                batch = {'distance_to_goal':[],'success':[],'softspl':[],'spl':[], 'step':[], 'first_obs':[], 'ep_reward': []}
                for i in infos:
                    for key in batch:
                        if(key=='collisions'):
                            batch[key].append(i[key]['count'])
                        else:
                            batch[key].append(i[key])

                for key in batch:
                    batch[key]=np.mean(np.array(batch[key]))
                ################################################################
                means = batch
                print(means, flush=True)
            ####################################################################
            count_episodes += 1
        ########################################################################
        # batch = {'distance_to_goal':[],'success':[],'softspl':[],'spl':[]}
        batch = {'distance_to_goal':[],'success':[],'softspl':[],'spl':[], 'step':[], 'first_obs':[], 'ep_reward': []}

        for i in infos:
            for key in batch:
                if(key=='collisions'):
                    batch[key].append(i[key]['count'])
                else:
                    batch[key].append(i[key])

        for key in batch:
            batch[key]=np.mean(np.array(batch[key]))

        means = batch
        print("FINAL=",means, flush=True)
        ########################################################################
        self._env.close()
        print("EXIT SUCCESS")



################################################################################

# from habitat.sims.habitat_simulator.actions import (
#     HabitatSimActions,
#     HabitatSimV1ActionSpaceConfiguration,
# )
# from habitat.tasks.nav.nav import SimulatorTaskAction
#
# HabitatSimActions.MOVE_FORWARD: habitat_sim.ActionSpec(
#     "move_forward",
#     habitat_sim.ActuationSpec(
#         amount=self.config.FORWARD_STEP_SIZE
#     ),
# ),
#
# @habitat.registry.register_task_action
# class StrafeRight(SimulatorTaskAction):
#     def _get_uuid(self, *args, **kwargs) -> str:
#         return "strafe_right"
#
#     def step(self, *args, **kwargs):
#         return self._sim.step(HabitatSimActions.STRAFE_RIGHT)
################################################################################
'''
    End of file and call for main function
'''
################################################################################
def main():
    '''
        Parser args and config files and call trainer!
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-type", default="blind", choices=["blind", "rgb", "depth", "rgbd"]
    )
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )

    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    parser.add_argument("--model-path", default="", type=str)
    args = parser.parse_args()

    # config = get_config(
    #     ["configs/beyond.yaml","configs/beyond_trainer.yaml","configs/ddppo_pointnav.yaml"], ["BASE_TASK_CONFIG_PATH", config_paths]
    # ).clone()

    config = get_config(
        ["configs/beyond.yaml","configs/beyond_trainer.yaml","configs/ddppo_pointnav.yaml", config_paths],
    ).clone()

    # config_ori = copy.deepcopy(config)
    # config = config.TASK_CONFIG

    # config.defrost()
    #
    # config.TORCH_GPU_ID = 0
    # config.INPUT_TYPE = args.input_type
    # config.MODEL_PATH = args.model_path
    # config.RANDOM_SEED = 7
    # config.TENSORBOARD_DIR = "habitat-challenge-data/log/"
    #
    # # config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)
    # config.freeze()

    config.defrost()
    config.TORCH_GPU_ID = 0
    config.INPUT_TYPE = args.input_type
    config.MODEL_PATH = args.model_path
    config.RANDOM_SEED = 7
    # config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP"]
    # config.TASK_CONFIG.SIMULATOR.TILT_ANGLE = 0.00001

    config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    config.freeze()

    print("-------------------------------------------------------------")
    # print(config)
    # print("-------------------------------------------------------------")

    trainer = PPO(config)

    if(config.BEYOND_TRAINER.TRAIN_MODE):
        print("TRAIN MODE")
        trainer.train()
    else:
        for weight in config.BEYOND_TRAINER.RESUME_EVAL_CHEKPOINT:
            print("EVAL MODE: checkpoint", weight)
            trainer.eval_checkpoint(weight)
    # print("EXITING DEBUG")
    # exit()

    #everything went fine so exit with 0
    return 0

if __name__ == "__main__":
    main()
##########################################################################
