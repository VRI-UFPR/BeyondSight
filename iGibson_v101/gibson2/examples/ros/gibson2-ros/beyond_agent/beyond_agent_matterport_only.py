#!/usr/bin/env python3

import argparse
import os
import random
from collections import OrderedDict

import numba
import numpy as np
import PIL
import torch
from gym.spaces import Box, Dict, Discrete

import habitat
from habitat import Config
from habitat.core.agent import Agent
# from habitat_baselines.common.utils import batch_obs
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ppo import PointNavBaselinePolicy, Policy

from habitat_baselines.rl.ddppo.policy.resnet_policy import (  # isort:skip noqa
    PointNavResNetPolicy,
)

from typing import Any, Dict, List, Optional
from collections import defaultdict, deque

#beyond stuff
import mapper
import model
# import beyond_agent
from beyond_agent import BeyondAgent, DDPPOAgent_PointNav

from yolact.utils.functions import SavePath
from yolact.data import cfg, set_cfg
from yolact.yolact import Yolact
from yolact.utils.augmentations import BaseTransform, FastBaseTransform_rgb_2_rgb, Resize
from yolact.layers.output_utils import postprocess, undo_image_transformation
####

@numba.njit
def _seed_numba(seed: int):
    random.seed(seed)
    np.random.seed(seed)


class BeyondAgentMp3dOnly(BeyondAgent):
    def __init__(self, device, config, batch_size, is_batch_internal=False):
        # super().__init__(device, config, batch_size)
        self.device=device
        self.config=config
        self.batch_size=batch_size
        self.input_shape = config.BEYOND.GLOBAL_POLICY.INPUT_SHAPE

        self.orientation=torch.zeros(self.batch_size,1, device=self.device)
        self.array_of_preds=[[] for _ in range(self.batch_size)]
        self.ep_iteration=torch.zeros(self.batch_size)
        self.global_input = None
        self.is_batch_internal=is_batch_internal
        # self.ep_iteration=0

        # print(self.config)

        # if(self.config.BEYOND.GLOBAL_POLICY.USE_MATTERPORT_TO_GIBSON):
        #     self.m3pd_to_gibson = {"__background__": -1,
        #                "person": -1,
        #                "bicycle": -1,
        #                "car": -1,
        #                "motorcycle": -1,
        #                "airplane": -1,
        #                "bus": -1,
        #                "train": -1,
        #                "truck": -1,
        #                "boat": -1,
        #                "traffic light": -1,
        #                "fire hydrant": -1,
        #                "stop sign": -1,
        #                "parking meter": -1,
        #                "bench": -1,
        #                "bird": -1,
        #                "cat": -1,
        #                "dog": -1,
        #                "horse": -1,
        #                "sheep": -1,
        #                "cow": -1,
        #                "elephant": -1,
        #                "bear": -1,
        #                "zebra": -1,
        #                "giraffe": -1,
        #                "backpack": -1,
        #                "umbrella": -1,
        #                "handbag": -1,
        #                "tie": -1,
        #                "suitcase": -1,
        #                "frisbee": -1,
        #                "skis": -1,
        #                "snowboard": -1,
        #                "sports ball": -1,
        #                "kite": -1,
        #                "baseball bat": -1,
        #                "baseball glove": -1,
        #                "skateboard": -1,
        #                "surfboard": -1,
        #                "tennis racket": -1,
        #                "bottle": -1,
        #                "wine glass": -1,
        #                "cup": -1,
        #                "fork": -1,
        #                "knife": -1,
        #                "spoon": -1,
        #                "bowl": -1,
        #                "banana": -1,
        #                "apple": -1,
        #                "sandwich": -1,
        #                "orange": -1,
        #                "broccoli": -1,
        #                "carrot": -1,
        #                "hot dog": -1,
        #                "pizza": -1,
        #                "donut": -1,
        #                "cake": -1,
        #                "chair": 0,
        #                "couch": 5,
        #                "potted plant": 8,
        #                "bed": 6,
        #                "dining table": 1,
        #                "toilet": 10,
        #                "tv": 13,
        #                "laptop": -1,
        #                "mouse": -1,
        #                "remote": -1,
        #                "keyboard": -1,
        #                "cell phone": -1,
        #                "microwave": -1,
        #                "oven": -1,
        #                "toaster": -1,
        #                "sink": -1,
        #                "refrigerator": 3,
        #                "book": -1,
        #                "clock": -1,
        #                "vase": -1,
        #                "scissors": -1,
        #                "teddy bear": -1,
        #                "hair drier": -1,
        #                "toothbrush": -1
        #                }
        #     self.m3pd_to_gibson_inverted = {v: k for k, v in self.m3pd_to_gibson.items()}
        #
        # #internal dict for the COCO classes to map the 30 classes in the semantic map, similar classes in matterport3D map to the same number
        # self._internal_dict = {'__background__': 0, 'person': 0, 'bicycle': 26, 'car': 0, 'motorcycle': 0, 'airplane': 0, 'bus': 0, 'train': 0, 'truck': 0, 'boat': 0, 'traffic light': 0, 'fire hydrant': 0, 'stop sign': 0, 'parking meter': 0, 'bench': 17, 'bird': 0, 'cat': 0, 'dog': 0, 'horse': 0, 'sheep': 0, 'cow': 0, 'elephant': 0, 'bear': 0, 'zebra': 0, 'giraffe': 0, 'backpack': 0, 'umbrella': 21, 'handbag': 25, 'tie': 19, 'suitcase': 20, 'frisbee': 0, 'skis': 0, 'snowboard': 0, 'sports ball': 27, 'kite': 0, 'baseball bat': 0, 'baseball glove': 0, 'skateboard': 0, 'surfboard': 0, 'tennis racket': 0, 'bottle': 0, 'wine glass': 24, 'cup': 16, 'fork': 0, 'knife': 0, 'spoon': 0, 'bowl': 15, 'banana': 0, 'apple': 0, 'sandwich': 0, 'orange': 0, 'broccoli': 0, 'carrot': 0, 'hot dog': 0, 'pizza': 0, 'donut': 0, 'cake': 23, 'chair': 1, 'couch': 6, 'sofa': 6, 'potted plant': 2, 'plant': 2, 'bed': 7, 'dining table': 9, 'table': 9, 'toilet': 10, 'tv': 12, 'tv_monitor': 12, 'laptop': 29, 'mouse': 0, 'remote': 28, 'keyboard': 0, 'cell phone': 0, 'microwave': 18, 'oven': 14, 'toaster': 0, 'sink': 3, 'refrigerator': 11, 'cabinet': 11, 'book': 5, 'clock': 13, 'vase': 4, 'scissors': 0, 'teddy bear': 22, 'hair drier': 0, 'toothbrush': 0}
        # # self._internal_dict_inverted = {v: k for k, v in self._internal_dict.items()}
        #
        # self.YOLACT_COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        #                 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        #                 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        #                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        #                 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        #                 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        #                 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        #                 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        #                 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        #                 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        #                 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        #                 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        #                 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        #                 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


        '''
        load/create all necessary modules
        '''
        self.planner = DDPPOAgent_PointNav(config,self.batch_size)
        self.planner.reset_batch()
        # if(self.batch_size==1):

        # self.planner = None

        ########################################################################
        print("using gt sseg instead of YOLACT++")
        ########################################################################
        # model_path = SavePath.from_str(config.BEYOND.SSEG.CHECKPOINT)
        # # TODO: Bad practice? Probably want to do a name lookup instead.
        # args_config = model_path.model_name + '_config'
        # print('Config not specified. Parsed %s from the file name.\n' % args_config)
        # set_cfg(args_config)
        #
        # print('Loading model...', end='')
        # self.sseg = Yolact()
        # if(config.BEYOND.SSEG.LOAD_CHECKPOINT):
        #     print(config.BEYOND.SSEG.CHECKPOINT, end='')
        #     self.sseg.load_weights(config.BEYOND.SSEG.CHECKPOINT)
        # else:
        #     print("using random values for the SSEG")
        # #freeze weights
        # self.sseg.eval()
        # self.sseg.to(self.device)
        # print(' Done.')
        ########################################################################

        '''
        So far we only will use the Global_Policy
        '''

        self.g_policy = model.Global_Policy(input_shape=config.BEYOND.GLOBAL_POLICY.INPUT_SHAPE, hidden_size=config.BEYOND.GLOBAL_POLICY.HIDDEN_SIZE, is_mp3d=True)
        self.g_policy.to(self.device)

        print("loading model to device",self.device)
        if(config.BEYOND.GLOBAL_POLICY.LOAD_CHECKPOINT):
            checkpoint_filepath = config.BEYOND.GLOBAL_POLICY.CHECKPOINT
            print("loading model...",checkpoint_filepath)

            model_dict = self.g_policy.state_dict()
            pretrained_dict = torch.load(checkpoint_filepath)


            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            '''
            Test keys are different
            '''
            keys_to_delete = []
            for k, v in pretrained_dict.items():
                if( pretrained_dict[k].size() != model_dict[k].size()):
                    print(k,"are different. loaded",pretrained_dict[k].size(), "model uses", model_dict[k].size())
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
            self.g_policy.load_state_dict(model_dict)
        else:
            print("using random values for the GLOBAL_POLICY")

        self.reset_map(self.batch_size)

    def set_goal_train(self):
        self.g_policy.train()

    def set_goal_eval(self):
        self.g_policy.eval()

    def reset(self, env_idx=0):

        self.array_of_preds[env_idx]=[]
        self.ep_iteration[env_idx]=0
        self.mapper_wrapper.reset_map_batch_idx(env_idx)
        self.planner.reset_idx(env_idx)

    def evalimage(self,observations):
        '''
        pass observations into the net

        perform fake sseg
        '''
        ####################################################################
        dict_map = observations['sseg_dict']
        ####################################################################
        #remap obj_id to class_id
        y_pred= observations['semantic'].long()

        # for key in observations:
        #     print(key,observations[key].shape)

        for i in range(self.batch_size):
            y_pred[i] =  dict_map[i][observations['semantic'][i].long()].long()
        ####################################################################
        '''
            WARNING THIS IS A FAKE PREDICTION USE FOR TESTING ONLY
            THE REAL THING SHOULD USE YOLACT++ PREDICTIONS
        '''
        #batch x h x w
        y_pred_scores = (0. - 0.1) * torch.randn(y_pred.shape[0],y_pred.shape[1],y_pred.shape[2],self.config.BEYOND.GLOBAL_POLICY.N_CLASSES,device=self.device) + 0.1
        ####################################################################
        '''
        same but more efficient than
        def compute_fake_pred(y_pred, y_pred_scores):
            for b in range(y_pred.shape[0]):
                for i in range(y_pred.shape[1]):
                    for j in range(y_pred.shape[2]):
                        y_pred_scores[b][i][j][y_pred[b][i][j]] = 1.-y_pred_scores[b][i][j][y_pred[b][i][j]]
            return y_pred_scores
        '''
        xx,yy = torch.meshgrid([torch.arange(0,y_pred.shape[1]), torch.arange(0,y_pred.shape[2])])
        y_pred_scores[:,xx,yy,y_pred]=1.-y_pred_scores[:,xx,yy,y_pred]
        ####################################################################
        return y_pred_scores

    def act(self, observations, env_idx=0):
        with torch.no_grad():
            # print("observations['compass']",observations['compass'], "observations['gps']",observations['gps'], "observations['objectgoal']",observations['objectgoal'])
            if(self.is_batch_internal):
                # observations = batch_obs([observations], device=self.device)
                # print("observations",observations)
                if(self.batch_size==1):
                    observations = batch_obs([observations], device=self.device)
                else:
                    observations = batch_obs(observations, device=self.device)
            observations['pointgoal_with_gps_compass'] = torch.zeros(self.batch_size,2,device=self.device)
            ####################################################################
            # '''
            #     Set episodic orientation
            # '''
            for env_idx in range(self.batch_size):
                if(self.ep_iteration[env_idx]==0):
                    self.reset_orientation(observations)
            ####################################################################
            '''
            Deal with object goal
            '''
            # if(self.config.BEYOND.GLOBAL_POLICY.USE_MATTERPORT_TO_GIBSON):
            #     object_goal = torch.zeros(self.batch_size, device=self.device).long()
            #     for i in range(self.batch_size):
            #         obj_i = self.m3pd_to_gibson_inverted[ observations['objectgoal'][i][0].item() ]
            #         object_goal[i] = self._internal_dict[obj_i]
            #     object_goal = object_goal.unsqueeze(-1)
            object_goal = observations['objectgoal'].long().unsqueeze(-1)

            ####################################################################
            '''
                First call yolact and segment the rgb
            '''
            ####################################################################
            # y_pred_scores = self.evalimage(observations['rgb'])
            y_pred_scores = self.evalimage(observations)
            # print("y_pred_scores.shape",y_pred_scores.shape)

            '''
            y_pred_scores = BATCH x HEIGHT x WIDTH x CLASSES
            '''
            ####################################################################
            '''
                Resize depth from [0,1] to the actual distance for correct projection
            '''
            resized_depth = observations['depth']
            resized_depth = (resized_depth.squeeze(-1)*(self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH-self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH))+self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
            ####################################################################
            '''
                Then do the map projection
            '''
            ####################################################################
            pose_batch = torch.zeros(self.batch_size,4,4)
            phi_batch = torch.zeros(self.batch_size)

            for i in range(self.batch_size):
                pose,phi = mapper.pose_from_angle_and_position(observations['compass'][i].cpu(),observations['gps'][i].cpu())
                if(phi>2*np.pi):
                    phi=phi-(2*np.pi)

                phi_batch[i] = phi
                pose_batch[i] = pose
            ####################################################################
            self.mapper_wrapper.pose6D = pose_batch.float().to(self.device)
            ####################################################################
            '''
            mapper is leaking gpu memory
            '''
            global_input = self.mapper_wrapper.update_map(resized_depth, y_pred_scores, phi_batch)
            self.global_input = global_input
            # global_input = torch.zeros(1,512,512,32,device=self.device)
            orientation = self.orientation
            ###############################################
            '''
                Then feeed forward the goal prediction
            '''
            ####################################################################
            y_pred_perc = self.g_policy(global_input,orientation,object_goal)

            #transform [0,1] to actual map cell coordinates
            y_pred_map_coo = y_pred_perc*(mapper.get_map_size_in_cells(self.mapper_wrapper.mapper.map_size_meters, self.mapper_wrapper.mapper.map_cell_size))
            #transform map cell coordinates into meters
            y_pred_reprojected_goal = mapper.map_to_2dworld(y_pred_map_coo, self.mapper_wrapper.mapper.map_size_meters, self.mapper_wrapper.mapper.map_cell_size)
            #since the map follow a ZX notation, we have a y_pred_reprojected_goal in ZX in meters

            '''
            DDPPO was pretrained using polar coordinates [rho, phi]
            '''
            observations = self.compute_pointgoal_with_pred(observations, y_pred_reprojected_goal)

            '''
                Use the mean for more temporal consistency
            '''
            ######################################################
            for i in range(self.batch_size):
                self.array_of_preds[i].append(y_pred_reprojected_goal)
                if(self.ep_iteration[i] % 8 == 0):
                    _mean = torch.mean(torch.stack(self.array_of_preds[i]), dim=0)
                    # print("1","y_pred_reprojected_goal.shape",y_pred_reprojected_goal,"_mean",_mean)
                    observations = self.compute_pointgoal_with_pred(observations, _mean)
                    # self.update_followers_goal(i,_mean.tolist())

                # euclidean_distance = self.get_distance_to_goal(i)
                euclidean_distance = observations['pointgoal_with_gps_compass'][i][0]
                if(euclidean_distance<self.config.TASK_CONFIG.TASK.SUCCESS_DISTANCE):
                    _mean = torch.mean(torch.stack(self.array_of_preds[i]), dim=0)
                    # print("2","y_pred_reprojected_goal.shape",y_pred_reprojected_goal,"_mean",_mean)
                    observations = self.compute_pointgoal_with_pred(observations, _mean)
                    # self.update_followers_goal(i,_mean.tolist())
            ######################################################
            ###############################################

            ###############################################
            '''
                Then pick the goal and transform into a fake pointgoal sensor
                and feed it to the DDPPO pointNav agent
            '''
            ###############################################
            # self.ep_iteration[env_idx]+=1

            del observations['objectgoal']
            del observations['compass']
            del observations['gps']
            action = self.planner.act(observations)
            if(self.batch_size==1):
                action = action.item()
            # print("action",action)
            # print("EXITING before act")
            # exit()
        return action
    ###############################################

def _to_tensor(v) -> torch.Tensor:
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)

def batch_obs(
    observations: List[Dict], device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:

    r"""Transpose a batch of observation dicts to a dict of batched
    observations.
    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None
    Returns:
        transposed dict of lists of observations.
    """
    batch = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(_to_tensor(obs[sensor]))

    for sensor in batch:
        batch[sensor] = (
            torch.stack(batch[sensor], dim=0)
            .to(device=device)
            .to(dtype=torch.float)
        )

    return batch
