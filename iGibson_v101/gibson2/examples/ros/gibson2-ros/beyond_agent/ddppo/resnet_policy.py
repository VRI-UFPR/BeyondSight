# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, Tuple

import numpy as np
import torch
from gym import spaces
from torch import nn as nn
from torch.nn import functional as F
from torch import Tensor

# from habitat.config import Config
from config_default import Config

################################################################################
#habitat sensors
# from habitat.tasks.nav.nav import (
#     EpisodicCompassSensor,
#     EpisodicGPSSensor,
#     HeadingSensor,
#     ImageGoalSensor,
#     IntegratedPointGoalGPSAndCompassSensor,
#     PointGoalSensor,
#     ProximitySensor,
# )
# from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
################################################################################

# from habitat_baselines.common.baseline_registry import baseline_registry
# from habitat_baselines.rl.ddppo.policy import resnet
import ddppo.resnet as resnet
# from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
#     RunningMeanAndVar,
# )
from ddppo.running_mean_and_var import RunningMeanAndVar

# from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from ddppo.rnn_encoder import RNNStateEncoder

# from habitat_baselines.rl.ppo import Net, Policy
from ddppo.policy import Net, Policy
# from habitat_baselines.utils.common import Flatten

from ddppo.common import ResizeCenterCropper

class Flatten(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.flatten(x, start_dim=1)


class PointNavResNetPolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size=512,
        num_recurrent_layers=2,
        rnn_type="LSTM",
        resnet_baseplanes=32,
        backbone="resnet50",
        normalize_visual_inputs=False,
        obs_transform=ResizeCenterCropper(size=(256, 256)),
    ):
        super().__init__(
            PointNavResNetNet(
                observation_space=observation_space,
                action_space=action_space,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                obs_transform=obs_transform,
            ),
            action_space.n,
        )


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        baseplanes=32,
        ngroups=32,
        spatial_size=128,
        make_backbone=None,
        normalize_visual_inputs=False,
        obs_transform=ResizeCenterCropper(size=(256, 256)),
    ):
        super().__init__()

        self.obs_transform = obs_transform
        if self.obs_transform is not None:
            observation_space = self.obs_transform.transform_observation_space(
                observation_space
            )

        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
            spatial_size = observation_space.spaces["rgb"].shape[0] // 2
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
            spatial_size = observation_space.spaces["depth"].shape[0] // 2
        else:
            self._n_input_depth = 0

        if normalize_visual_inputs:
            self.running_mean_and_var = RunningMeanAndVar(
                self._n_input_depth + self._n_input_rgb
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        if not self.is_blind:
            input_channels = self._n_input_depth + self._n_input_rgb
            self.backbone = make_backbone(input_channels, baseplanes, ngroups)

            final_spatial = int(
                spatial_size * self.backbone.final_spatial_compress
            )
            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(after_compression_flat_size / (final_spatial ** 2))
            )
            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )

            self.output_shape = (
                num_compression_channels,
                final_spatial,
                final_spatial,
            )

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations):
        if self.is_blind:
            return None

        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]

            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)

            cnn_input.append(depth_observations)

        if self.obs_transform:
            cnn_input = [self.obs_transform(inp) for inp in cnn_input]

        x = torch.cat(cnn_input, dim=1)
        x = F.avg_pool2d(x, 2)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


class PointNavResNetNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        num_recurrent_layers,
        rnn_type,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs,
        obs_transform=ResizeCenterCropper(size=(256, 256)),
    ):
        super().__init__()

        self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action

        if (
            "pointgoal_with_gps_compass"
            in observation_space.spaces
        ):
            n_input_goal = (
                observation_space.spaces[
                    "pointgoal_with_gps_compass"
                ].shape[0]
                + 1
            )
            self.tgt_embeding = nn.Linear(n_input_goal, 32)
            rnn_input_size += 32

        if "objectgoal" in observation_space.spaces:
            self._n_object_categories = (
                int(
                    observation_space.spaces["objectgoal"].high[0]
                )
                + 1
            )
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            rnn_input_size += 32

        if "gps" in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                "gps"
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32

        if "pointgoal" in observation_space.spaces:
            input_pointgoal_dim = observation_space.spaces[
                "pointgoal"
            ].shape[0]
            self.pointgoal_embedding = nn.Linear(input_pointgoal_dim, 32)
            rnn_input_size += 32

        if "heading" in observation_space.spaces:
            input_heading_dim = (
                observation_space.spaces["heading"].shape[0] + 1
            )
            assert input_heading_dim == 2, "Expected heading with 2D rotation."
            self.heading_embedding = nn.Linear(input_heading_dim, 32)
            rnn_input_size += 32

        if "proximity" in observation_space.spaces:
            input_proximity_dim = observation_space.spaces[
                "proximity"
            ].shape[0]
            self.proximity_embedding = nn.Linear(input_proximity_dim, 32)
            rnn_input_size += 32

        if "compass" in observation_space.spaces:
            assert (
                observation_space.spaces["compass"].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            rnn_input_size += 32

        self._hidden_size = hidden_size

        self.visual_encoder = ResNetEncoder(
            observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
            obs_transform=obs_transform,
        )

        if not self.visual_encoder.is_blind:
            self.visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(
                    np.prod(self.visual_encoder.output_shape), hidden_size
                ),
                nn.ReLU(True),
            )

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []
        if not self.is_blind:
            if "visual_features" in observations:
                visual_feats = observations["visual_features"]
            else:
                visual_feats = self.visual_encoder(observations)

            visual_feats = self.visual_fc(visual_feats)
            x.append(visual_feats)

        if "pointgoal_with_gps_compass" in observations:
            goal_observations = observations[
                "pointgoal_with_gps_compass"
            ]
            goal_observations = torch.stack(
                [
                    goal_observations[:, 0],
                    torch.cos(-goal_observations[:, 1]),
                    torch.sin(-goal_observations[:, 1]),
                ],
                -1,
            )

            x.append(self.tgt_embeding(goal_observations))

        if "pointgoal" in observations:
            goal_observations = observations["pointgoal"]
            x.append(self.pointgoal_embedding(goal_observations))

        if "proximity" in observations:
            sensor_observations = observations["proximity"]
            x.append(self.proximity_embedding(sensor_observations))

        if "heading" in observations:
            sensor_observations = observations["heading"]
            sensor_observations = torch.stack(
                [
                    torch.cos(sensor_observations[0]),
                    torch.sin(sensor_observations[0]),
                ],
                -1,
            )
            x.append(self.heading_embedding(sensor_observations))

        if "objectgoal" in observations:
            object_goal = observations["objectgoal"].long()
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if "compass" in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations["compass"]),
                    torch.sin(observations["compass"]),
                ],
                -1,
            )
            x.append(
                self.compass_embedding(compass_observations.squeeze(dim=1))
            )

        if "gps" in observations:
            x.append(
                self.gps_embedding(observations["gps"])
            )

        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().squeeze(dim=-1)
        )
        x.append(prev_actions)

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states


#
# # @baseline_registry.register_policy
# class PointNavResNetPolicy(Policy):
#     def __init__(
#         self,
#         observation_space: spaces.Dict,
#         action_space,
#         hidden_size: int = 512,
#         num_recurrent_layers: int = 2,
#         rnn_type: str = "LSTM",
#         resnet_baseplanes: int = 32,
#         backbone: str = "resnet50",
#         normalize_visual_inputs: bool = False,
#         force_blind_policy: bool = False,
#         **kwargs
#     ):
#         super().__init__(
#             PointNavResNetNet(
#                 observation_space=observation_space,
#                 action_space=action_space,
#                 hidden_size=hidden_size,
#                 num_recurrent_layers=num_recurrent_layers,
#                 rnn_type=rnn_type,
#                 backbone=backbone,
#                 resnet_baseplanes=resnet_baseplanes,
#                 normalize_visual_inputs=normalize_visual_inputs,
#                 force_blind_policy=force_blind_policy,
#             ),
#             action_space.n,
#         )
#
#     @classmethod
#     def from_config(
#         cls, config: Config, observation_space: spaces.Dict, action_space
#     ):
#         return cls(
#             observation_space=observation_space,
#             action_space=action_space,
#             hidden_size=config.RL.PPO.hidden_size,
#             rnn_type=config.RL.DDPPO.rnn_type,
#             num_recurrent_layers=config.RL.DDPPO.num_recurrent_layers,
#             backbone=config.RL.DDPPO.backbone,
#             normalize_visual_inputs="rgb" in observation_space.spaces,
#             force_blind_policy=config.FORCE_BLIND_POLICY,
#         )
#
#
# class ResNetEncoder(nn.Module):
#     def __init__(
#         self,
#         observation_space: spaces.Dict,
#         baseplanes: int = 32,
#         ngroups: int = 32,
#         spatial_size: int = 128,
#         make_backbone=None,
#         normalize_visual_inputs: bool = False,
#     ):
#         super().__init__()
#
#         if "rgb" in observation_space.spaces:
#             self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
#             spatial_size = observation_space.spaces["rgb"].shape[0] // 2
#         else:
#             self._n_input_rgb = 0
#
#         if "depth" in observation_space.spaces:
#             self._n_input_depth = observation_space.spaces["depth"].shape[2]
#             spatial_size = observation_space.spaces["depth"].shape[0] // 2
#         else:
#             self._n_input_depth = 0
#
#         if normalize_visual_inputs:
#             self.running_mean_and_var: nn.Module = RunningMeanAndVar(
#                 self._n_input_depth + self._n_input_rgb
#             )
#         else:
#             self.running_mean_and_var = nn.Sequential()
#
#         if not self.is_blind:
#             input_channels = self._n_input_depth + self._n_input_rgb
#             self.backbone = make_backbone(input_channels, baseplanes, ngroups)
#
#             final_spatial = int(
#                 spatial_size * self.backbone.final_spatial_compress
#             )
#             after_compression_flat_size = 2048
#             num_compression_channels = int(
#                 round(after_compression_flat_size / (final_spatial ** 2))
#             )
#             self.compression = nn.Sequential(
#                 nn.Conv2d(
#                     self.backbone.final_channels,
#                     num_compression_channels,
#                     kernel_size=3,
#                     padding=1,
#                     bias=False,
#                 ),
#                 nn.GroupNorm(1, num_compression_channels),
#                 nn.ReLU(True),
#             )
#
#             self.output_shape = (
#                 num_compression_channels,
#                 final_spatial,
#                 final_spatial,
#             )
#
#     @property
#     def is_blind(self):
#         return self._n_input_rgb + self._n_input_depth == 0
#
#     def layer_init(self):
#         for layer in self.modules():
#             if isinstance(layer, (nn.Conv2d, nn.Linear)):
#                 nn.init.kaiming_normal_(
#                     layer.weight, nn.init.calculate_gain("relu")
#                 )
#                 if layer.bias is not None:
#                     nn.init.constant_(layer.bias, val=0)
#
#     def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
#         if self.is_blind:
#             return None
#
#         cnn_input = []
#         if self._n_input_rgb > 0:
#             rgb_observations = observations["rgb"]
#             # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
#             rgb_observations = rgb_observations.permute(0, 3, 1, 2)
#             rgb_observations = rgb_observations / 255.0  # normalize RGB
#             cnn_input.append(rgb_observations)
#
#         if self._n_input_depth > 0:
#             depth_observations = observations["depth"]
#
#             # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
#             depth_observations = depth_observations.permute(0, 3, 1, 2)
#
#             cnn_input.append(depth_observations)
#
#         x = torch.cat(cnn_input, dim=1)
#         x = F.avg_pool2d(x, 2)
#
#         x = self.running_mean_and_var(x)
#         x = self.backbone(x)
#         x = self.compression(x)
#         return x
#
#
# class PointNavResNetNet(Net):
#     """Network which passes the input image through CNN and concatenates
#     goal vector with CNN's output and passes that through RNN.
#     """
#
#     def __init__(
#         self,
#         observation_space: spaces.Dict,
#         action_space,
#         hidden_size: int,
#         num_recurrent_layers: int,
#         rnn_type: str,
#         backbone,
#         resnet_baseplanes,
#         normalize_visual_inputs: bool,
#         force_blind_policy: bool = False,
#     ):
#         super().__init__()
#
#         self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
#         self._n_prev_action = 32
#         rnn_input_size = self._n_prev_action
#
#         if (
#             "pointgoal_with_gps_compass"
#             in observation_space.spaces
#         ):
#             n_input_goal = (
#                 observation_space.spaces[
#                     "pointgoal_with_gps_compass"
#                 ].shape[0]
#                 + 1
#             )
#             self.tgt_embeding = nn.Linear(n_input_goal, 32)
#             rnn_input_size += 32
#
#         if "objectgoal" in observation_space.spaces:
#             self._n_object_categories = (
#                 int(
#                     observation_space.spaces["objectgoal"].high[0]
#                 )
#                 + 1
#             )
#             self.obj_categories_embedding = nn.Embedding(
#                 self._n_object_categories, 32
#             )
#             rnn_input_size += 32
#
#         if "gps" in observation_space.spaces:
#             input_gps_dim = observation_space.spaces[
#                 "gps"
#             ].shape[0]
#             self.gps_embedding = nn.Linear(input_gps_dim, 32)
#             rnn_input_size += 32
#
#         if "pointgoal" in observation_space.spaces:
#             input_pointgoal_dim = observation_space.spaces[
#                 "pointgoal"
#             ].shape[0]
#             self.pointgoal_embedding = nn.Linear(input_pointgoal_dim, 32)
#             rnn_input_size += 32
#
#         if "heading" in observation_space.spaces:
#             input_heading_dim = (
#                 observation_space.spaces["heading"].shape[0] + 1
#             )
#             assert input_heading_dim == 2, "Expected heading with 2D rotation."
#             self.heading_embedding = nn.Linear(input_heading_dim, 32)
#             rnn_input_size += 32
#
#         if "proximity" in observation_space.spaces:
#             input_proximity_dim = observation_space.spaces[
#                 "proximity"
#             ].shape[0]
#             self.proximity_embedding = nn.Linear(input_proximity_dim, 32)
#             rnn_input_size += 32
#
#         if "compass" in observation_space.spaces:
#             assert (
#                 observation_space.spaces["compass"].shape[
#                     0
#                 ]
#                 == 1
#             ), "Expected compass with 2D rotation."
#             input_compass_dim = 2  # cos and sin of the angle
#             self.compass_embedding = nn.Linear(input_compass_dim, 32)
#             rnn_input_size += 32
#
#         if "imagegoal" in observation_space.spaces:
#             goal_observation_space = spaces.Dict(
#                 {"rgb": observation_space.spaces["imagegoal"]}
#             )
#             self.goal_visual_encoder = ResNetEncoder(
#                 goal_observation_space,
#                 baseplanes=resnet_baseplanes,
#                 ngroups=resnet_baseplanes // 2,
#                 make_backbone=getattr(resnet, backbone),
#                 normalize_visual_inputs=normalize_visual_inputs,
#             )
#
#             self.goal_visual_fc = nn.Sequential(
#                 Flatten(),
#                 nn.Linear(
#                     np.prod(self.goal_visual_encoder.output_shape), hidden_size
#                 ),
#                 nn.ReLU(True),
#             )
#
#             rnn_input_size += hidden_size
#
#         self._hidden_size = hidden_size
#
#         self.visual_encoder = ResNetEncoder(
#             observation_space if not force_blind_policy else spaces.Dict({}),
#             baseplanes=resnet_baseplanes,
#             ngroups=resnet_baseplanes // 2,
#             make_backbone=getattr(resnet, backbone),
#             normalize_visual_inputs=normalize_visual_inputs,
#         )
#
#         if not self.visual_encoder.is_blind:
#             self.visual_fc = nn.Sequential(
#                 Flatten(),
#                 nn.Linear(
#                     np.prod(self.visual_encoder.output_shape), hidden_size
#                 ),
#                 nn.ReLU(True),
#             )
#
#         self.state_encoder = RNNStateEncoder(
#             (0 if self.is_blind else self._hidden_size) + rnn_input_size,
#             self._hidden_size,
#             rnn_type=rnn_type,
#             num_layers=num_recurrent_layers,
#         )
#
#         self.train()
#
#     @property
#     def output_size(self):
#         return self._hidden_size
#
#     @property
#     def is_blind(self):
#         return self.visual_encoder.is_blind
#
#     @property
#     def num_recurrent_layers(self):
#         return self.state_encoder.num_recurrent_layers
#
#     def forward(
#         self,
#         observations: Dict[str, torch.Tensor],
#         rnn_hidden_states,
#         prev_actions,
#         masks,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         x = []
#         if not self.is_blind:
#             if "visual_features" in observations:
#                 visual_feats = observations["visual_features"]
#             else:
#                 visual_feats = self.visual_encoder(observations)
#
#             visual_feats = self.visual_fc(visual_feats)
#             x.append(visual_feats)
#
#         if "pointgoal_with_gps_compass" in observations:
#             goal_observations = observations[
#                 "pointgoal_with_gps_compass"
#             ]
#             if goal_observations.shape[1] == 2:
#                 # Polar Dimensionality 2
#                 # 2D polar transform
#                 goal_observations = torch.stack(
#                     [
#                         goal_observations[:, 0],
#                         torch.cos(-goal_observations[:, 1]),
#                         torch.sin(-goal_observations[:, 1]),
#                     ],
#                     -1,
#                 )
#             else:
#                 assert (
#                     goal_observations.shape[1] == 3
#                 ), "Unsupported dimensionality"
#                 vertical_angle_sin = torch.sin(goal_observations[:, 2])
#                 # Polar Dimensionality 3
#                 # 3D Polar transformation
#                 goal_observations = torch.stack(
#                     [
#                         goal_observations[:, 0],
#                         torch.cos(-goal_observations[:, 1])
#                         * vertical_angle_sin,
#                         torch.sin(-goal_observations[:, 1])
#                         * vertical_angle_sin,
#                         torch.cos(goal_observations[:, 2]),
#                     ],
#                     -1,
#                 )
#
#             x.append(self.tgt_embeding(goal_observations))
#
#         if "pointgoal" in observations:
#             goal_observations = observations["pointgoal"]
#             x.append(self.pointgoal_embedding(goal_observations))
#
#         if "proximity" in observations:
#             sensor_observations = observations["proximity"]
#             x.append(self.proximity_embedding(sensor_observations))
#
#         if "heading" in observations:
#             sensor_observations = observations["heading"]
#             sensor_observations = torch.stack(
#                 [
#                     torch.cos(sensor_observations[0]),
#                     torch.sin(sensor_observations[0]),
#                 ],
#                 -1,
#             )
#             x.append(self.heading_embedding(sensor_observations))
#
#         if "objectgoal" in observations:
#             object_goal = observations["objectgoal"].long()
#             x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))
#
#         if "compass" in observations:
#             compass_observations = torch.stack(
#                 [
#                     torch.cos(observations["compass"]),
#                     torch.sin(observations["compass"]),
#                 ],
#                 -1,
#             )
#             x.append(
#                 self.compass_embedding(compass_observations.squeeze(dim=1))
#             )
#
#         if "gps" in observations:
#             x.append(
#                 self.gps_embedding(observations["gps"])
#             )
#
#         if "imagegoal" in observations:
#             goal_image = observations["imagegoal"]
#             goal_output = self.goal_visual_encoder({"rgb": goal_image})
#             x.append(self.goal_visual_fc(goal_output))
#
#         prev_actions = self.prev_action_embedding(
#             ((prev_actions.float() + 1) * masks).long().squeeze(dim=-1)
#         )
#         x.append(prev_actions)
#
#         out = torch.cat(x, dim=1)
#         out, rnn_hidden_states = self.state_encoder(
#             out, rnn_hidden_states, masks
#         )
#
#         return out, rnn_hidden_states
