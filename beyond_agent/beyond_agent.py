#!/usr/bin/env python3

'''
Keep the implementation detached from a specific simulator
'''

import argparse
import os
import random
from collections import OrderedDict

import numba
import numpy as np
import PIL
import torch
from gym.spaces import Box, Discrete
from gym.spaces import Dict as gymDict

from typing import Any, Dict, List, Optional
from collections import defaultdict, deque

from config_default import Config
from ddppo.resnet_policy import PointNavResNetPolicy

#beyond stuff
import mapper
import model
from actor_critic import ActorCriticPolicy

import pyastar

#yolact stuff
from yolact.utils.functions import SavePath
from yolact.data import cfg, set_cfg
from yolact.yolact import Yolact
from yolact.utils.augmentations import BaseTransform, FastBaseTransform_rgb_2_rgb, Resize
from yolact.layers.output_utils import postprocess, undo_image_transformation
####

####FOR # DEBUG:
from habitat.core.utils import try_cv2_import
cv2 = try_cv2_import()
from PIL import Image

from habitat.utils.visualizations import maps
# from beyond_agent_without_internal_mapper import rotate_image
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST)
  return result
####

@numba.njit
def _seed_numba(seed: int):
    random.seed(seed)
    np.random.seed(seed)

# @numba.jit(nopython=True)
def draw_line_with_if(x0, y0, x1, y1, grid, test_value):
    """Implementation of Bresenham's line drawing algorithm
    See en.wikipedia.org/wiki/Bresenham's_line_algorithm
    """

    """Yield integer coordinates on the line from (x0, y0) to (x1, y1).
    Input coordinates should be integers.
    The result will contain both the start and the end point.
    """
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0

    for x in range(dx + 1):
        value = x0 + x*xx + y*yx, y0 + x*xy + y*yy
        if grid[value[0]][value[1]] == test_value:
            return value
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy

    return value

class BeyondAgent():
    def __init__(self, device, config, batch_size, is_batch_internal=False):
        ########################################################################
        self.device=device
        self.config=config
        self.batch_size=batch_size
        self.ep_iteration=torch.zeros(self.batch_size)

        self._updated_dict= {
            0:-1,
            "chair": 0,
            "table": 1,
            "picture": 2,
            "cabinet": 3,
            "cushion": 4,
            "sofa": 5,
            "bed": 6,
            "chest_of_drawers": 7,
            "plant": 8,
            "sink": 9,
            "toilet": 10,
            "stool": 11,
            "towel": 12,
            "tv_monitor": 13,
            "shower": 14,
            "bathtub": 15,
            "counter": 16,
            "fireplace": 17,
            "gym_equipment": 18,
            "seating": 19,
            "clothes": 20
        }
        self._updated_dict= {k: v+1 for k, v in self._updated_dict.items()}
        self._updated_dict_reverse = {v: k for k, v in self._updated_dict.items()} #this reverse key with value
        ########################################################################
        '''
        load/create all necessary modules
        '''
        ########################################################################
        '''
        Load DDPPO with handle local policy
        '''
        self.planner = DDPPOAgent_PointNav(config,self.batch_size)
        self.planner.reset_batch()
        ########################################################################

        ########################################################################
        # print("using gt sseg instead of YOLACT++")
        ########################################################################
        # model_path = SavePath.from_str(config.BEYOND.SSEG.CHECKPOINT)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        # args_config = model_path.model_name + '_config'
        args_config = "yolact_plus_resnet50_mp3d_2021_config"
        # print('Config not specified. Parsed %s from the file name.\n' % args_config)
        set_cfg(args_config)

        # print('Loading model...', end='')
        self.sseg = Yolact()
        if(config.BEYOND.SSEG.LOAD_CHECKPOINT):
            # print(config.BEYOND.SSEG.CHECKPOINT, end='')
            self.sseg.load_weights(config.BEYOND.SSEG.CHECKPOINT)
        # else:
        #     pass
        #     print("using random values for the SSEG")
        #freeze weights
        self.sseg.eval()
        self.sseg.to(self.device)
        # print(' Done.')
        ########################################################################

        ########################################################################
        '''
        Load Global_Policy
        '''
        self.action_space = Box( np.array([0.0,0.0],dtype=np.float32), np.array([1.0,1.0],dtype=np.float32), dtype=np.float32)

        # self.actor_critic = ActorCriticPolicy(
        #     action_space=self.action_space,
        #     lr_schedule=self.config.BEYOND_TRAINER.PPO.lr,
        #     use_sde=False,
        #     device=self.device,
        #     config=self.config,
        # )
        self.actor_critic = ActorCriticPolicy(
            action_space=self.action_space,
            use_sde=False,
            device=self.device,
            config=self.config,
        )
        self.actor_critic.to(self.device)

        # if(self.config.BEYOND.GLOBAL_POLICY.LOAD_CHECKPOINT):
        #     self.actor_critic.load_feature_net_weights(self.config.BEYOND.GLOBAL_POLICY.CHECKPOINT)

        # ########################################################################

        ##
        '''
        load policy
        '''
        if(self.config.BEYOND.GLOBAL_POLICY.LOAD_CHECKPOINT):
            ckpt_dict = torch.load(self.config.BEYOND.GLOBAL_POLICY.CHECKPOINT, map_location="cpu")
            self.actor_critic.load_actor_critic_weights(ckpt_dict["state_dict"])
            del ckpt_dict
    ########################################################################
        ##


        ########################################################################
        '''
        Print network
        '''
        # print("GLOBAL_POLICY's state_dict:")
        # print("-----------------------------------------")
        # for param_tensor in self.g_policy.state_dict():
        #     print(param_tensor, "\t", self.g_policy.state_dict()[param_tensor].size())
        # print("-----------------------------------------")
        ########################################################################
        # print("EXITING")
        # exit()
        ########################################################################

        ########################################################################
        '''
        Initialize internal map per env
        '''
        self.recreate_map_array(self.batch_size)
        self.not_done_masks = torch.zeros(
            self.batch_size, 1, device=self.device, dtype=torch.bool
        )
        self.test_recurrent_hidden_states = torch.zeros(
            self.batch_size,
            self.actor_critic.feature_net.rnn_layers,
            self.actor_critic.feature_net.feature_out_size,
            device=self.device,
        )

        # self.y_pred_map_coo_local = np.zeros((self.batch_size,2),dtype=np.int32)
        # self.y_pred_map_coo_long = np.zeros((self.batch_size,2),dtype=np.int32)
    ############################################################################
    def evaluate_actions(
        self, observations, main_inputs, action, rnn_states, masks
    ):
        # print("evaluate_actions observations.keys()",observations.keys())
        features, rnn_states = self.actor_critic.feature_net(
            main_inputs, observations['objectgoal'], mapper.compass_to_orientation(observations['compass']), rnn_states, masks
        )
        value = self.actor_critic.critic(features)
        distribution = self.actor_critic._get_action_dist_from_latent(features)
        # action = distribution.get_actions(deterministic=deterministic)

        action_log_probs = distribution.log_prob(action).unsqueeze(-1)
        distribution_entropy = distribution.entropy()

        return value, action_log_probs, distribution_entropy
    ############################################################################
    def get_value(self, observations, main_inputs, rnn_states, masks):
        # print("get_value observations.keys()",observations.keys())
        features, rnn_states = self.actor_critic.feature_net(
            main_inputs, observations['objectgoal'], mapper.compass_to_orientation(observations['compass']), rnn_states, masks
        )
        value = self.actor_critic.critic(features)
        return value
    ############################################################################
    def recreate_map_array(self, batch_size):
        self.batch_size = batch_size
        self.mapper_wrapper_array = [mapper.MapperWrapper(self.config, self.device) for i in range(batch_size)]
    ############################################################################
    def reset(self, env_idx=0):
        '''
        reset is per env
        '''
        # print("agent reset",env_idx)
        #by idx
        self.ep_iteration[env_idx]=0
        self.mapper_wrapper_array[env_idx].reset_map()
        self.planner.reset_idx(env_idx)
        self.not_done_masks[env_idx] = False
        # print("self.test_recurrent_hidden_states before",self.test_recurrent_hidden_states.shape,flush=True)
        self.test_recurrent_hidden_states[env_idx,:,:] = 0
        # print("self.test_recurrent_hidden_states after",self.test_recurrent_hidden_states.shape,flush=True)
        # self.y_pred_map_coo_local = np.zeros((self.batch_size,2),dtype=np.int32)
        # self.y_pred_map_coo_long = np.zeros((self.batch_size,2),dtype=np.int32)
    ############################################################################
    def visualize_semantic_map(self, main_input, y_pred_map_coo_local, target_class):
        '''
        DEBUG ENHANCED MAP
        '''
        print(self._updated_dict_reverse[target_class.cpu().item()])
        # tmp_img0 = np.array(self.colorized(torch.argmax((main_input[0,:22,:,:]).permute(1,2,0), dim=-1).cpu().numpy()) )
        # cv2.imshow("ARGMAX_map",tmp_img0)

        tmp_img = torch.zeros(256,256,3,device=self.device).long()
        #opencv bgr
        tmp_img[:,:,:] = torch.where(main_input[0,-3,:,:]>0,128,0).unsqueeze(-1)#gray occupied
        t_255 = torch.ones(256,256,device=self.device).long()*255
        tmp_img[:,:,0] = torch.where(main_input[0,-1,:,:]>0, t_255, tmp_img[:,:,0])#blue #current_location
        tmp_img[:,:,2] = torch.where(main_input[0,-2,:,:]>0, t_255, tmp_img[:,:,2])#red #past_locations

        # tmp_img[:,:,0:1] = torch.where(main_input[0,-2,:,:]>0,255,0).unsqueeze(-1)#blue #current_location
        # tmp_img[:,:,1:2] = torch.where(main_input[0,-1,:,:]>0,255,0).unsqueeze(-1)#green #pred
        # tmp_img[:,:,2:3] = torch.where(main_input[0,-3,:,:]>0,255,0).unsqueeze(-1)#red #past_locations


        tmp_img[y_pred_map_coo_local[0]][y_pred_map_coo_local[1]][0] = 0
        tmp_img[y_pred_map_coo_local[0]][y_pred_map_coo_local[1]][1] = 255
        tmp_img[y_pred_map_coo_local[0]][y_pred_map_coo_local[1]][2] = 0

        cv2.imshow("agent_traj_n_goal",tmp_img.byte().cpu().numpy())


        # tmp_img2 = torch.zeros(512,512,3,device=self.device).long()
        # #opencv bgr
        # tmp_img2[:,:,:] = torch.where(self.mapper_wrapper_array[0].map_aux[-4,:,:]>0,128,0).unsqueeze(-1)#red #past_locations
        #
        # tmp_img2[:,:,0:1] = torch.where(self.mapper_wrapper_array[0].map_aux[-2,:,:]>0,255,tmp_img2[:,:,0:1]).unsqueeze(-1)#blue #current_location
        # tmp_img2[:,:,1:2] = torch.where(self.mapper_wrapper_array[0].map_aux[-1,:,:]>0,255,tmp_img2[:,:,1:2]).unsqueeze(-1)#green #pred
        # tmp_img2[:,:,2:3] = torch.where(self.mapper_wrapper_array[0].map_aux[-3,:,:]>0,255,tmp_img2[:,:,2:3]).unsqueeze(-1)#red #past_locations
        #
        #
        # # tmp_img2[y_pred_map_coo_local[0]][y_pred_map_coo_local[1]][0] = 0
        # # tmp_img2[y_pred_map_coo_local[0]][y_pred_map_coo_local[1]][1] = 0
        # # tmp_img2[y_pred_map_coo_local[0]][y_pred_map_coo_local[1]][2] = 255
        #
        # cv2.imshow("agent_traj_n_goal_512",tmp_img2.byte().cpu().numpy())

        # tmp_img = torch.ones(512,512,3)*255
        # tmp_img2 = torch.ones(256,256,3)*255


        tmp_img2 = torch.zeros(256,256,3,device=self.device).long()
        # tmp_img2[:,:,1:2] = torch.where(main_input[0,target_class,:,:]>0,255,0).unsqueeze(-1)
        tmp_img2[:,:,0] = main_input[0,-1]*255#blue
        tmp_img2[:,:,1] = main_input[0,target_class]*255#green
        tmp_img2[:,:,2] = main_input[0,-2]*255#red


        # tmp_img2[:,:,0] = torch.where(main_input[0,-2,:,:]>0, t_255, tmp_img2[:,:,0])#blue #current_location
        # # tmp_img[:,:,2] = torch.where(main_input[0,-3,:,:]>0, t_255, tmp_img[:,:,2])#red #past_locations
        # tmp_img2[:,:,2] = torch.where(main_input[0,-1,:,:]>0, t_255, tmp_img2[:,:,2])#red #pred

        # tmp_img2[y_pred_map_coo_local[0]][y_pred_map_coo_local[1]][0] = 255
        # tmp_img2[y_pred_map_coo_local[0]][y_pred_map_coo_local[1]][1] = 0
        # tmp_img2[y_pred_map_coo_local[0]][y_pred_map_coo_local[1]][2] = 0

        cv2.imshow("target_class",tmp_img2.byte().cpu().numpy())
        # cv2.waitKey()

        # tmp_img3 = torch.ones(512,512,3)*255
        #
        # tmp_img[:,:,:1] = torch.where(main_input[0,0,:,:]>0,255,0).unsqueeze(-1)
        # # print("tmp_img.shape",tmp_img.shape)
        # cv2.imshow("bg_map",tmp_img.byte().cpu().numpy())
        #
        # print("tmp_img2.shape",tmp_img2.shape)
        #
        # tmp_img3[:,:,2:3] = torch.where(main_input[0,1,:,:]>0,255,0).unsqueeze(-1)
        # # print("tmp_img3.shape",tmp_img3.shape)
        # cv2.imshow("chair_map",tmp_img3.byte().cpu().numpy())
    ############################################################################
    def debug_join_sem(self, pred):
        pred = pred.permute(2,0,1)
        new = torch.zeros((480,640),device=self.device)

        for channel in range(pred.shape[0]):
            # mask = torch.logical_xor(mask,pred[channel])
            mask_inv = 1-pred[channel]
            new = (new * mask_inv) + (pred[channel]*channel)

        return new

    def visualize_observations(self, observations, y_pred_scores_yolact):
        arrow = cv2.imread("extras/arrow.png")
        arrow = rotate_image(arrow, np.rad2deg(-observations['compass'][0].item()) )
        cv2.imshow("arrow",arrow)

        # if(observations['top_down_map'][0]):
        #     # map = maps.colorize_draw_agent_and_fit_to_height(observations['top_down_map'], output_height=1024)
        #     map = maps.colorize_draw_agent_and_fit_to_height(observations['top_down_map'][0], output_height=256)
        #     cv2.imshow('top_down_map',map)
        #     # cv2.waitKey()

        perm = torch.LongTensor([2,1,0])
        rgb = observations['rgb'][0][:,:,perm].byte().cpu().numpy()


        # print("y_pred_scores_yolact.shape",y_pred_scores_yolact.shape)


        tmp_img = self.debug_join_sem(y_pred_scores_yolact[0])
        y_pred_scores_yolact_img = self.colorized( tmp_img.cpu().numpy() )

        # o_sem = torch.unique(observations['semantic'][0]).cpu().numpy()
        # print("o_sem",o_sem)
        # o_sem = [self._updated_dict_reverse[i] for i in o_sem]
        # print("o_sem",o_sem)
        # o_sem_pred = torch.unique(tmp_img).cpu().numpy()
        # print("o_sem_pred",o_sem_pred)
        # o_sem_pred = [self._updated_dict_reverse[i] for i in o_sem_pred]
        # print("o_sem_pred",o_sem_pred)

        # print("sem2", u_sem2 )
        # tmp = [self._updated_dict_reverse[i] for i in u_sem2]
        # print("sem2", tmp )
        # print()

        # sem2 = self.colorized(observations['semantic'][0].cpu().numpy())

        # img = np.hstack( (rgb,sem2) )
        # img = np.hstack( (rgb,sem2,y_pred_scores_yolact_img) )
        img = np.hstack( (rgb,y_pred_scores_yolact_img) )
        cv2.imshow('img',img)

        # print("compass",observations['compass'],"gps",observations['gps'])
        '''
        esc = 27
        Upkey : 2490368
        DownKey : 2621440
        LeftKey : 2424832
        RightKey: 2555904
        '''

        k = cv2.waitKey()
        if k==ord('w'):#  up
            action=1
        elif k==ord('a'):#  left
            action=2
        elif k==ord('d'):#  right
            action=3
        elif k==ord('s'):# esc
            action=0
        else:
            action=0

        return torch.from_numpy(np.array( [[action]] )).to(self.device)
    ############################################################################
    def colorized(self, img):
        d3_269_colors_rgb : np.ndarray = np.array([[0, 0, 0], [255, 255, 0], [28, 230, 255], [255, 52, 255], [255, 74, 70], [0, 137, 65], [0, 111, 166], [163, 0, 89], [255, 219, 229], [122, 73, 0], [0, 0, 166], [99, 255, 172], [183, 151, 98], [0, 77, 67], [143, 176, 255], [153, 125, 135], [90, 0, 7], [128, 150, 147], [254, 255, 230], [27, 68, 0], [79, 198, 1], [59, 93, 255], [74, 59, 83], [255, 47, 128], [97, 97, 90], [186, 9, 0], [107, 121, 0], [0, 194, 160], [255, 170, 146], [255, 144, 201], [185, 3, 170], [209, 97, 0], [221, 239, 255], [0, 0, 53], [123, 79, 75], [161, 194, 153], [48, 0, 24], [10, 166, 216], [1, 51, 73], [0, 132, 111], [55, 33, 1], [255, 181, 0], [194, 255, 237], [160, 121, 191], [204, 7, 68], [192, 185, 178], [194, 255, 153], [0, 30, 9], [0, 72, 156], [111, 0, 98], [12, 189, 102], [238, 195, 255], [69, 109, 117], [183, 123, 104], [122, 135, 161], [120, 141, 102], [136, 85, 120], [250, 208, 159], [255, 138, 154], [209, 87, 160], [190, 196, 89], [69, 102, 72], [0, 134, 237], [136, 111, 76], [52, 54, 45], [180, 168, 189], [0, 166, 170], [69, 44, 44], [99, 99, 117], [163, 200, 201], [255, 145, 63], [147, 138, 129], [87, 83, 41], [0, 254, 207], [176, 91, 111], [140, 208, 255], [59, 151, 0], [4, 247, 87], [200, 161, 161], [30, 110, 0], [121, 0, 215], [167, 117, 0], [99, 103, 169], [160, 88, 55], [107, 0, 44], [119, 38, 0], [215, 144, 255], [155, 151, 0], [84, 158, 121], [255, 246, 159], [32, 22, 37], [114, 65, 143], [188, 35, 255], [153, 173, 192], [58, 36, 101], [146, 35, 41], [91, 69, 52], [253, 232, 220], [64, 78, 85], [0, 137, 163], [203, 126, 152], [164, 232, 4], [50, 78, 114], [106, 58, 76], [131, 171, 88], [0, 28, 30], [209, 247, 206], [0, 75, 40], [200, 208, 246], [163, 164, 137], [128, 108, 102], [34, 40, 0], [191, 86, 80], [232, 48, 0], [102, 121, 109], [218, 0, 124], [255, 26, 89], [138, 219, 180], [30, 2, 0], [91, 78, 81], [200, 149, 197], [50, 0, 51], [255, 104, 50], [102, 225, 211], [207, 205, 172], [208, 172, 148], [126, 211, 121], [1, 44, 88], [122, 123, 255], [214, 142, 1], [53, 51, 57], [120, 175, 161], [254, 178, 198], [117, 121, 124], [131, 115, 147], [148, 58, 77], [181, 244, 255], [210, 220, 213], [149, 86, 189], [106, 113, 74], [0, 19, 37], [2, 82, 95], [10, 163, 247], [233, 129, 118], [219, 213, 221], [94, 188, 209], [61, 79, 68], [126, 100, 5], [2, 104, 78], [150, 43, 117], [141, 133, 70], [150, 149, 197], [231, 115, 206], [216, 106, 120], [62, 137, 190], [202, 131, 78], [81, 138, 135], [91, 17, 60], [85, 129, 59], [231, 4, 196], [0, 0, 95], [169, 115, 153], [75, 129, 96], [89, 115, 138], [255, 93, 167], [247, 201, 191], [100, 49, 39], [81, 58, 1], [107, 148, 170], [81, 160, 88], [164, 91, 2], [29, 23, 2], [226, 0, 39], [231, 171, 99], [76, 96, 1], [156, 105, 102], [100, 84, 123], [151, 151, 158], [0, 106, 102], [57, 20, 6], [244, 215, 73], [0, 69, 210], [0, 108, 49], [221, 182, 208], [124, 101, 113], [159, 178, 164], [0, 216, 145], [21, 160, 138], [188, 101, 233], [255, 255, 254], [198, 220, 153], [32, 59, 60], [103, 17, 144], [107, 58, 100], [245, 225, 255], [255, 160, 242], [204, 170, 53], [55, 69, 39], [139, 180, 0], [121, 120, 104], [198, 0, 90], [59, 0, 10], [200, 98, 64], [41, 96, 124], [64, 35, 52], [125, 90, 68], [204, 184, 124], [184, 129, 131], [170, 81, 153], [181, 214, 195], [163, 132, 105], [159, 148, 240], [167, 69, 113], [184, 148, 166], [113, 187, 140], [0, 180, 51], [120, 158, 201], [109, 128, 186], [149, 63, 0], [94, 255, 3], [228, 255, 252], [27, 225, 119], [188, 177, 229], [118, 145, 47], [0, 49, 9], [0, 96, 205], [210, 0, 150], [137, 85, 99], [41, 32, 29], [91, 50, 19], [167, 111, 66], [137, 65, 46], [26, 58, 42], [73, 75, 90], [168, 140, 133], [244, 171, 170], [163, 243, 171], [0, 198, 200], [234, 139, 102], [149, 138, 159], [189, 201, 210], [159, 160, 100], [190, 71, 0], [101, 129, 136], [131, 164, 133], [69, 60, 35], [71, 103, 93], [58, 63, 0], [6, 18, 3], [223, 251, 113], [134, 142, 126], [152, 208, 88], [108, 143, 125], [215, 191, 194], [60, 62, 110], [216, 61, 102], [47, 93, 155], [108, 94, 70], [210, 91, 136], [91, 101, 108], [0, 181, 127], [84, 92, 70], [134, 96, 151], [54, 93, 37], [37, 47, 153], [0, 204, 255], [103, 78, 96], [252, 0, 156], [146, 137, 107]],
        dtype=np.uint8,
        )

        semantic_img = Image.new("P", (img.shape[1], img.shape[0]))
        semantic_img.putpalette(d3_269_colors_rgb[:256].flatten())
        semantic_img.putdata((img.flatten()).astype(np.uint8))
        semantic_img = semantic_img.convert("RGB")

        semantic_img = np.array(semantic_img)
        perm = np.array([2,1,0])
        semantic_img = semantic_img[:,:,perm]#BGR

        return semantic_img
    ############################################################################
    def evalimage(self,frame,target):
        # print("frame.shape",frame.shape)
        #preprocess
        frame_transformed = FastBaseTransform_rgb_2_rgb()(frame.float())
        #feed forward
        preds = self.sseg(frame_transformed)
        #postprocess
        # y_pred_scores = torch.zeros(self.batch_size,frame.shape[1],frame.shape[2],self.config.BEYOND.GLOBAL_POLICY.N_CLASSES,device=self.device)
        y_pred_scores = torch.zeros(self.batch_size,self.config.BEYOND.GLOBAL_POLICY.N_CLASSES,frame.shape[1],frame.shape[2],device=self.device)
        seen_target = torch.zeros(self.batch_size)
        ########################################################################
        for i in range(self.batch_size):
            h, w, _ = frame[i].shape

            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            t = postprocess(preds, w, h, batch_idx=i, visualize_lincomb=self.config.BEYOND.SSEG.DISPLAY_LINCOMB,
                crop_masks=self.config.BEYOND.SSEG.CROP_MASK,
                score_threshold=self.config.BEYOND.SSEG.SCORE_THRESHOLD, binarize_masks=self.config.BEYOND.SSEG.BINARIZE_MASKS)
            # t = classes, scores, boxes, masks
            cfg.rescore_bbox = save

            classes, scores, boxes, masks = t
            masks_o = masks.clone()
            ########################################################################


            # ########################################################################
            # ########################################################################
            # # Quick and dirty lambda for selecting the color for a particular index
            # # Also keeps track of a per-gpu color cache for maximum speed
            # class_color = False
            # mask_alpha=0.45
            # num_dets_to_consider = classes.shape[0]
            # img_gpu = frame[0].clone()/255.0
            #
            # # print("img_gpu.shape",img_gpu.shape)
            # # undo_transform = False
            # def get_color(j, on_gpu=None):
            #     global color_cache
            #     color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
            #
            #     if on_gpu is not None and color_idx in color_cache[on_gpu]:
            #         return color_cache[on_gpu][color_idx]
            #     else:
            #         color = COLORS[color_idx]
            #         # if not undo_transform:
            #             # The image might come in as RGB or BRG, depending
            #             # color = (color[2], color[1], color[0])
            #         if on_gpu is not None:
            #             color = torch.Tensor(color).to(on_gpu).float() / 255.
            #             color_cache[on_gpu][color_idx] = color
            #         return color
            #
            # # First, draw the masks on the GPU where we can do it really fast
            # # Beware: very fast but possibly unintelligible mask-drawing code ahead
            # # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
            # # After this, mask is of size [num_dets, h, w, 1]
            # if num_dets_to_consider > 0:
            #     masks = masks[:num_dets_to_consider, :, :, None]
            #
            #     # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
            #     colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
            #     masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha
            #
            #     # This is 1 everywhere except for 1-mask_alpha where the mask is
            #     inv_alph_masks = masks * (-mask_alpha) + 1
            #
            #     # I did the math for this on pen and paper. This whole block should be equivalent to:
            #     #    for j in range(num_dets_to_consider):
            #     #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
            #     masks_color_summand = masks_color[0]
            #     if num_dets_to_consider > 1:
            #         inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            #         masks_color_cumul = masks_color[1:] * inv_alph_cumul
            #         masks_color_summand += masks_color_cumul.sum(dim=0)
            #
            #     img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
            #
            # img_numpy = (img_gpu * 255).byte().cpu().numpy()
            #
            # for j in reversed(range(num_dets_to_consider)):
            #     x1, y1, x2, y2 = boxes[j, :]
            #     color = get_color(j)
            #     score = scores[j]
            #
            #
            #     cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)
            #
            #
            #     _class = cfg.dataset.class_names[classes[j]]
            #     text_str = '%s: %.2f' % (_class, score)
            #
            #     font_face = cv2.FONT_HERSHEY_DUPLEX
            #     font_scale = 0.6
            #     font_thickness = 1
            #
            #     text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
            #
            #     text_pt = (x1, y1 - 3)
            #     text_color = [255, 255, 255]
            #
            #     cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
            #     cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
            #
            # perm = torch.LongTensor([2,1,0])
            # perm = np.array([2,1,0])
            # img_numpy = img_numpy[:,:,perm]
            # cv2.imwrite("/habitat-challenge-data/"+str(self.n_episode)+"_"+str(int(self.ep_iteration[0].numpy()))+"___sem.png",img_numpy)
            # ########################################################################
            # ########################################################################

            #masks is per instance. In this case we want per class
            #we want to merge masks of same class
            #classes follow coco default ordering

            #first group instances with same class
            unique_classes, inverse_indices, counts = torch.unique(classes,sorted=True,return_inverse=True,return_counts=True)


            if target[i].cpu() in unique_classes.cpu():
                seen_target[i] = 1.0

            # unique_classes x counts
            unique_classes_set = [[] for _ in range(unique_classes.shape[0])]
            for instance in range(masks_o.shape[0]):
                unique_classes_set[ inverse_indices[instance].item() ].append(masks_o[instance])


            #we want a single image per class
            for j in range(len(unique_classes_set)):
                unique_classes_set[j] = torch.stack( unique_classes_set[j] ).permute(1,2,0)
                # unique_classes_set[j], _ = torch.max( unique_classes_set[j], dim=-1 )
                tmp, _ = torch.max( unique_classes_set[j], dim=-1 )
                #then we need to order thoses images following the _internal_dict order
                # print("j",j,"unique_classes[j]",unique_classes[j],"self.YOLACT_COCO_CLASSES[unique_classes[j] ]",self.YOLACT_COCO_CLASSES[unique_classes[j] ],"self._internal_dict[ self.YOLACT_COCO_CLASSES[unique_classes[j] ] ]",self._internal_dict[ self.YOLACT_COCO_CLASSES[unique_classes[j] ] ])

                # y_pred_scores[i][ self._internal_dict[ self.YOLACT_COCO_CLASSES[unique_classes[j] ] ] ] = tmp

                # print("unique_classes",unique_classes)
                # print("inverse_indices",inverse_indices)
                # print("counts",counts)
                # print("unique_classes_set",unique_classes_set)
                # exit()

                #i is the env_idx, j is the idx in the detected classes
                # y_pred_scores[i][ unique_classes[j] ] = tmp

                #YOLACT CLASSES STARTS AFTER BG SO YOLACT CLASS 0 IS INTERNAL CLASS 1
                tmp = torch.where(tmp>self.config.BEYOND.SSEG.SCORE_THRESHOLD,1.0,0.0)
                y_pred_scores[i][ unique_classes[j]+1 ] = tmp



            # unique_classes_set = torch.stack(unique_classes_set)
            ########################################################################

        ########################################################################
        #BATCH x CLASS x H x W
        # print("y_pred_scores.shape",y_pred_scores.shape)

        y_pred_scores = y_pred_scores.permute(0,2,3,1)
        #BATCH x H x W x CLASS
        # print("y_pred_scores.shape",y_pred_scores.shape)
        # exit()
        del classes, scores, boxes, masks, t, preds, unique_classes, inverse_indices, counts

        return y_pred_scores, seen_target
    ############################################################################
    def split_into_channels(self, sseg):

        new = torch.zeros((self.config.BEYOND.GLOBAL_POLICY.N_CLASSES,self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT,self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH),device=self.device)
        u_sem = torch.unique(sseg)

        for channel in u_sem:
            new[channel] = torch.where(sseg == channel, 1.0, 0.0)

        new = new.permute(1,2,0)
        return new
    ############################################################################
    def compute_pointgoal_with_pred(self,observations, y_pred_reprojected_goal,env_idx):
        ###############################################
        '''
        Convert the point goal prediction to pointgoal_with_gps_compass

        Sensor that integrates PointGoals observations (which are used PointGoal Navigation) and GPS+Compass.
        For the agent in simulator the forward direction is along negative-z.
        In polar coordinate format the angle returned is azimuth to the goal.
        '''

        '''
        I just need to convert from cartesian_to_polar
        '''

        '''
        This is important gps is in episodic -zx notation

        pointgoal_with_gps_compass should be in world xyz, we will attempt to use episodic xyz
        '''
        #so -zx to xyz, y=0 since it is episodic,( -x, 0, -z as in pose? )
        # position = observations['gps'].cpu()
        # position = position[env_idx]+self.mapper_wrapper_array[env_idx].shift_origin_gps
        # agent_position = torch.zeros(3)
        # agent_position[0] = -position[1]
        # agent_position[2] = position[0]

        '''
        this block is to centered_cropped
        '''
        agent_position = torch.zeros(3)


        #prediction is in episodic ZX meters so do the same
        goal_position = torch.zeros(3)
        goal_position[0] = -y_pred_reprojected_goal[1]
        goal_position[2] = y_pred_reprojected_goal[0]

        # print("agent_position",agent_position)
        # print("goal_position",goal_position)
        '''
        Seems correct so far, I still unsure about the sign of the coordinates, needs further # DEBUG:
        '''

        '''
        Here this is important as well, compass represent the agent angle. The angle is 0 at state t=0.
        rotation_world_agent should be a quaternion representing the the true rotation betwen the agent
        and the world. Since we will attempt to use episodic coo instead of world coo.

        We will adapt this. We will convert the angle to a quaternion
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)
        '''

        q_r,q_i,q_j,q_k = mapper.angle_to_quaternion_coeff(observations['compass'][env_idx].cpu(), [0.,1.,0.])
        rotation_world_agent = np.quaternion(q_r,q_i,q_j,q_k)

        pointgoal_with_gps_compass = torch.from_numpy(mapper._compute_pointgoal(agent_position.detach().numpy(), rotation_world_agent, goal_position.detach().numpy()))
        #needs a better solution
        # pointgoal_with_gps_compass[1] -= np.pi
        # #ensure -pi,pi range
        # if pointgoal_with_gps_compass[1] < -np.pi:
        #     pointgoal_with_gps_compass[1] =  (pointgoal_with_gps_compass[1] + np.pi) + np.pi
        '''
        i need to analize if it needs to invert or not
        '''
        pointgoal_with_gps_compass[1] = pointgoal_with_gps_compass[1]*-1

        observations['pointgoal_with_gps_compass'][env_idx]=pointgoal_with_gps_compass
        # print("observations['pointgoal_with_gps_compass'][env_idx]",observations['pointgoal_with_gps_compass'][env_idx])

        return observations
    ############################################################################
    ############################################################################
    def eval(self):
        self.actor_critic.eval()

    def train(self):
        self.actor_critic.train()
    ############################################################################
    def act(self, observations):
        '''
        act should used for single env only, batch_size==1
        '''

        #test if the observations has a batch dim or not
        # print("observations",observations)
        shape = observations['objectgoal'].shape

        if(len(shape)>1):
            batch = observations
        else:
            batch = batch_obs([observations], device=self.device)

        value, action, action_log_probs, local_planner_actions, main_inputs, computed_reward, rnn_states = self.act_with_batch(batch)
        local_planner_actions = local_planner_actions.item()

        return local_planner_actions

    ############################################################################
    def get_free_cell_on_path(self, grid, src_pt, dst_pt):
        grid = np.where(grid > 0.0, 10000, 1.0)
        grid = grid.astype(np.float32)

        '''
        redo in the new version the map is scaled down 5 times
        including the coordinates then the final coordinates are scaled up 5 times
        '''
        h,w = grid.shape
        scale_o = 5
        scale = 1/scale_o

        # print("src_pt",src_pt)
        # print("dst_pt",dst_pt)
        src_pt=(np.ceil(np.array(src_pt)*scale)).astype(np.int32)
        dst_pt=(np.ceil(np.array(dst_pt)*scale)).astype(np.int32)
        # print("src_pt",src_pt)
        # print("dst_pt",dst_pt)

        # grid = cv2.resize(grid, (int(np.ceil(h*scale)), int(np.ceil(w*scale))), interpolation = cv2.INTER_NEAREST)
        grid = cv2.resize(grid, (int(np.ceil(h*scale)), int(np.ceil(w*scale))), interpolation = cv2.INTER_LINEAR)
        # grid = im.resize((h*scale, w*scale), resample=Image.NEAREST)

        path = pyastar.astar_path(grid, src_pt, dst_pt, allow_diagonal=True)

        if path is None:
            '''
            draw a line until the cell is different from 1
            '''
            # closest = draw_line_with_if(dst_pt[0],dst_pt[1],src_pt[0],src_pt[1],grid,1)
            # print(closest)

            # if(closest is None):
            #     print("closest IS NONE",flush=True)

            # path = pyastar.astar_path(grid, src_pt, closest, allow_diagonal=True)
            path = pyastar.astar_path(grid, src_pt, dst_pt, allow_diagonal=True)
            if path is None:
                # print("PATH IS NONE",flush=True)
                return None

        #steps are 0.25 m and cells are 0.05, path[0] is src, path[-1] is dst
        # target_idx = min(6,path.shape[0]-1)

        # target_idx = min(10,path.shape[0]-1)
        # target_idx = min(1,path.shape[0]-1)

        target_idx = min(2,path.shape[0]-1)

        # print("new",np.ceil(path[target_idx]*scale_o).astype(np.int32))
        return np.ceil(path[target_idx]*scale_o).astype(np.int32)
        # return path[target_idx]
    ############################################################################
    # def update_pred(self,map,pred_local,pred_coo):
    #     pred_local = pred_local.astype(np.int32)
    #     pred_coo = pred_coo.astype(np.int32)
    #     for env_idx in range(self.batch_size):
    #         if(self.ep_iteration[env_idx]>0):
    #             map[env_idx][-1].fill_(0)
    #             # print("pred_local",pred_local,"pred_coo",pred_coo,flush=True)
    #             map[env_idx, -1, pred_local[env_idx][0]-1:pred_local[env_idx][0]+2, pred_local[env_idx][1]-1:pred_local[env_idx][1]+2  ] = 0.5
    #
    #             map[env_idx, -1, pred_coo[env_idx][0]-1:pred_coo[env_idx][0]+2, pred_coo[env_idx][1]-1:pred_coo[env_idx][1]+2  ] = 1.0
    #     return map
    ############################################################################
    def act_with_batch(self, observations, deterministic=False):
        with torch.no_grad():
            '''
            22 classes + 4 aux (occupied, past_locations, current_location, previous_prediction)
            global map is 51.2 meters with 0.1 cell size
            local_map is cropped 256,256 around the agent
            '''
            # main_inputs = torch.zeros((self.batch_size,26,256,256),device=self.device)
            main_inputs = torch.zeros((self.batch_size,25,256,256),device=self.device)
            # phi = torch.zeros((self.batch_size,1),device=self.device).long()
            ####################################################################
            '''
                First call yolact and segment the rgb
            '''
            ####################################################################
            y_pred_scores,seen_target = self.evalimage(observations['rgb'], observations['objectgoal'])
            # y_pred_scores = torch.zeros((self.batch_size,480,640,22),device=self.device)
            # print("y_pred_scores",y_pred_scores.shape)

            explore_reward = np.zeros((self.batch_size))
            for env_idx in range(self.batch_size):
                '''
                    pre cnn
                '''
                ################################################################
                '''
                    First call yolact and segment the rgb
                '''
                ################################################################
                # print("observations['semantic']",observations['semantic'].shape)
                # print("observations['semantic'][env_idx]",observations['semantic'][env_idx].shape)

                # y_pred_scores[env_idx] = self.split_into_channels(observations['semantic'][env_idx])
                # print("y_pred_scores",y_pred_scores.shape)
                # exit()
                ################################################################
                '''
                    Resize depth from [0,1] to the actual distance for correct projection
                '''
                resized_depth = observations['depth'][env_idx]
                resized_depth = (resized_depth.squeeze(-1)*(self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH-self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH))+self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
                ################################################################
                '''
                    Then do the map projection
                '''
                ################################################################
                self.mapper_wrapper_array[env_idx].update_current_step(self.ep_iteration[env_idx])
                self.mapper_wrapper_array[env_idx].update_pose(observations['compass'][env_idx].cpu(),observations['gps'][env_idx].cpu())
                #sseg yolact
                main_inputs[env_idx] = self.mapper_wrapper_array[env_idx].update_map(resized_depth, y_pred_scores[env_idx])
                #sseg gt
                # main_inputs[env_idx] = self.mapper_wrapper_array[env_idx].update_map(resized_depth, y_pred_scores)

                # phi[env_idx] =  int(np.floor((self.mapper_wrapper_array[env_idx].orientation*360)/30))

                '''
                compute reward for exploring
                '''
                # explore_reward[env_idx] = self.mapper_wrapper_array[env_idx].compute_reward()
                ################################################################

            #end for
            # print("main_inputs",main_inputs.shape,main_inputs)
            # main_inputs = self.update_pred(main_inputs,self.y_pred_map_coo_local,self.y_pred_map_coo_long)

            features, self.test_recurrent_hidden_states = self.actor_critic.feature_net(
                main_inputs, observations['objectgoal'], mapper.compass_to_orientation(observations['compass']), self.test_recurrent_hidden_states, self.not_done_masks
            )

            #  Make masks not done till reset (end of episode) will be called
            # self.not_done_masks.fill_(True)
            self.not_done_masks[:,:]=1.0

            # print("features",features.shape,features)
            value = self.actor_critic.critic(features)
            # distribution = self.actor_critic.action_net(features)
            # distribution = self.actor_critic._get_action_dist_from_latent(features, latent_sde=latent_sde)
            distribution = self.actor_critic._get_action_dist_from_latent(features)
            action = distribution.get_actions(deterministic=deterministic)

            action_log_probs = distribution.log_prob(action).unsqueeze(-1)

            # return value, action, action_log_probs


            observations['pointgoal_with_gps_compass'] = torch.zeros(self.batch_size,2,device=self.device)
            # x = nn.Sigmoid()(self.out(x))
            #ensure that the prediction stays on the [0,1] range

            y_pred_perc = torch.sigmoid(action)


            y_pred_perc = y_pred_perc.cpu()
            # print("y_pred_perc",y_pred_perc)
            # computed_reward = np.zeros((self.batch_size))
            # computed_reward = seen_target.cpu().numpy()

            # computed_reward = {'seen_target':seen_target.cpu().numpy(), 'exp_reward':np.zeros((self.batch_size)) }

            computed_reward = []
            for env_idx in range(self.batch_size):
                # computed_reward.append( {'seen_target':(seen_target.cpu().numpy())[env_idx], 'exp_reward':explore_reward[env_idx] } )
                computed_reward.append( {'seen_target':(seen_target.cpu().numpy())[env_idx], 'exp_reward':0.0 } )

            # map_size=mapper.get_map_size_in_cells(12.8, 0.05) -1
            # y_pred_map_coo = y_pred_perc*map_size
            #
            # closest_target_idx = np.ones((self.batch_size))*np.inf
            # for env_idx in range(self.batch_size):
            #     # ################################################################
            #     # ####################################################################
            #     # targets = torch.nonzero( main_inputs[env_idx][observations['objectgoal'][env_idx]+1].squeeze(0) ).cpu().numpy()
            #     # #target shape is first dim is the point ,second dim is an array of coordinates, batch, coox, cooy
            #     # if(targets.shape[0]>0):
            #     #     # print("activate heuristic",flush=True)
            #     #     # print("main_inputs",main_inputs.shape)
            #     #     # print("main_inputs2",main_inputs[env_idx].shape)
            #     #     # print("main_inputs3",main_inputs[env_idx][observations['objectgoal'][env_idx]+1].shape)
            #     #     # print("targets",targets)
            #     #     # print("observations['objectgoal'][env_idx]+1",observations['objectgoal'][env_idx]+1)
            #     #     diff_in_cells = np.array([128,128]) - targets
            #     #     euclid_dist_in_cells = np.linalg.norm(diff_in_cells, ord=2, axis=-1)
            #     #     closest_target_idx[env_idx]=np.argmin(euclid_dist_in_cells)
            #     #     # print("euclid_dist_in_cells[closest_target_idx]",euclid_dist_in_cells[closest_target_idx])
            #     #
            #     #     # y_pred_map_coo_long = targets[closest_target_idx].astype(np.int32)
            #     #     # computed_reward[env_idx] = -1
            #     #
            #     #     # '''
            #     #     # computed_reward should be the rmse between
            #     #     # '''
            #     #
            #     #     # y_pred_map_coo_long_reward = torch.round(y_pred_map_coo[env_idx]).long().cpu()
            #     #     # if(y_pred_map_coo_long_reward[0] > map_size):
            #     #     #     y_pred_map_coo_long_reward[0]=map_size
            #     #     # if(y_pred_map_coo_long_reward[1] > map_size):
            #     #     #     y_pred_map_coo_long_reward[1]=map_size
            #     #
            #     #
            #     #     # print("y_pred_map_coo_long_reward",y_pred_map_coo_long_reward,"y_pred_map_coo_long", y_pred_map_coo_long)
            #     #     # rmse = RMSELoss(y_pred_map_coo_long_reward.float().numpy(), y_pred_map_coo_long.astype(np.float32))
            #     #     #
            #     #     # computed_reward[env_idx] = {'rmse':rmse,'y_pred_map_coo_long':y_pred_map_coo_long_reward}
            #     #
            #     #     # y_pred_map_coo_long = y_pred_map_coo_long_reward
            #     # ####################################################################
            #     # # else:
            #     #     # map_size=mapper.get_map_size_in_cells(self.mapper_wrapper_array[env_idx].map_size_meters/2, self.mapper_wrapper_array[env_idx].map_cell_size) -1
            #     #     # map_size=mapper.get_map_size_in_cells(25.6, 0.1) -1
            #
            #     y_pred_map_coo_long = torch.round(y_pred_map_coo[env_idx]).long().cpu().numpy()
            #     if(y_pred_map_coo_long[0] > map_size):
            #         y_pred_map_coo_long[0]=map_size
            #     if(y_pred_map_coo_long[1] > map_size):
            #         y_pred_map_coo_long[1]=map_size
            #
            #     # computed_reward[env_idx] = {'rmse':0.0,'y_pred_map_coo_long':y_pred_map_coo_long}
            #     '''
            #     update pred
            #     '''
            #     # pred_y_pred_map_coo_long = y_pred_map_coo_long.clone().detach()
            #     # computed_reward[env_idx] = self.mapper_wrapper_array[env_idx].update_prev_pred_in_map(pred_y_pred_map_coo_long, observations['objectgoal'][env_idx][0])
            #
            #     # y_pred_map_coo_long = torch.round(y_pred_map_coo[env_idx]).long().cpu().numpy()
            #     # if(y_pred_map_coo_long[0] > map_size):
            #     #     y_pred_map_coo_long[0]=map_size
            #     # if(y_pred_map_coo_long[1] > map_size):
            #     #     y_pred_map_coo_long[1]=map_size
            #     '''
            #     HERE WE WILL USE A* TO FIND THE CLOSEST FREE CELL WITHIN SENSORING
            #     RANGE THAT STEER TOWARDS THE PREDICTED OBJECTIVE,
            #
            #     HAVING ALWAYS A CLOSE POINT GOAL FOR THE DDPPO PLANNER ENSURES A
            #     BETTER CONVERGENCE
            #     '''
            #     ################################################################
            #     #22 is the map_aux[0] which is the occupied map
            #     src_pt_value = int(np.floor(main_inputs.shape[-1]/2))
            #     src_pt = (src_pt_value,src_pt_value)
            #     dst_pt = (y_pred_map_coo_long[0].item(),y_pred_map_coo_long[1].item())
            #     # if(self.mapper_wrapper_array[env_idx].target_astar is None):
            #     #     # print("dst_pt pred",dst_pt)
            #     # else:
            #     #     dst_pt = self.mapper_wrapper_array[env_idx].target_astar
            #     #     # print("dst_pt target_astar",dst_pt)
            #
            #     #main_inputs is centered cropped so the middle of the map is where the agent is
            #     grid = main_inputs[env_idx][22].clone().detach()
            #     # print("y_pred_map_coo_long",y_pred_map_coo_long)
            #     y_pred_map_coo_local = self.get_free_cell_on_path(grid=grid.cpu().numpy(), src_pt=src_pt, dst_pt=dst_pt)
            #     if(y_pred_map_coo_local is None):
            #         '''
            #         could not find a path
            #         '''
            #         y_pred_map_coo_local = y_pred_map_coo_long
            #     # self.y_pred_map_coo_local[env_idx] = y_pred_map_coo_local.astype(np.int32)
            #     # self.y_pred_map_coo_long[env_idx] = y_pred_map_coo_long.astype(np.int32)
            #
            #     # print("y_pred_map_coo_local",y_pred_map_coo_local)
            #     # y_pred_map_coo_local[0]=128
            #     # y_pred_map_coo_local[1]=128-10
            #
            #     y_pred_reprojected_goal = mapper.map_to_2dworld(y_pred_map_coo_local, self.mapper_wrapper_array[env_idx].map_size_meters/2, self.mapper_wrapper_array[env_idx].map_cell_size)
            #     # print("y_pred_reprojected_goal",y_pred_reprojected_goal)
            #
            #
            #     ################################################################
            #     # y_pred_reprojected_goal = mapper.map_to_2dworld(y_pred_map_coo, self.mapper_wrapper_array[env_idx].map_size_meters, self.mapper_wrapper_array[env_idx].map_cell_size)
            #     ################################################################
            #
            #     observations = self.compute_pointgoal_with_pred(observations, y_pred_reprojected_goal,env_idx)
            #     ################################################################

            # ####################################################################
            # '''
            # DEBUG
            # '''
            # print("observations['pointgoal_with_gps_compass']",observations['pointgoal_with_gps_compass'])
            # print("y_pred_perc",y_pred_perc)
            # print("y_pred_map_coo_long",y_pred_map_coo_long)
            # print("y_pred_map_coo_local",y_pred_map_coo_local)
            # self.visualize_semantic_map(main_inputs,y_pred_map_coo_local, (observations['objectgoal'][0])+1)
            # # local_planner_actions = self.visualize_observations(observations,y_pred_scores)
            # self.visualize_observations(observations, y_pred_scores)
            # # ####################################################################

            observations['pointgoal_with_gps_compass'] = y_pred_perc
            #from [0,1] to [-pi,pi]
            # observations['pointgoal_with_gps_compass'][:,1] = (observations['pointgoal_with_gps_compass'][:,1]- 0 ) * ((np.pi-(-np.pi))/(1-0)) + (-np.pi)
            observations['pointgoal_with_gps_compass'][:,1] = (observations['pointgoal_with_gps_compass'][:,1] * 2*np.pi) -np.pi
            observations['pointgoal_with_gps_compass'] = observations['pointgoal_with_gps_compass'].to(self.device)
            '''
            feed ddppo
            '''
            del observations['objectgoal']
            del observations['compass']
            del observations['gps']
            # del observations['semantic']
            # print("observations['pointgoal_with_gps_compass']",observations['pointgoal_with_gps_compass'])


            '''
            COMMENT FOR DEBUG ONLY
            '''
            local_planner_actions = self.planner.act(observations)
            # print("local_planner_actions",local_planner_actions,"\n\n")
            # local_planner_actions_debug = self.planner.act(observations)
            # print("local_planner_actions_debug",local_planner_actions_debug)


            # # ####################################################################
            # arrow = cv2.imread("extras/arrow.png")
            # arrow = rotate_image(arrow, np.rad2deg(-observations['pointgoal_with_gps_compass'][0][1].item()) )
            # cv2.imshow("arrow2",arrow)

            # print("local_planner_actions",local_planner_actions,"\n")
            for env_idx in range(self.batch_size):
                computed_reward[env_idx]['exp_reward'] = local_planner_actions[env_idx][0].item()
            # # # ####################################################################

            # print("local_planner_actions",local_planner_actions)

            # print(value, action, action_log_probs, local_planner_actions, main_inputs, computed_reward)
            # print("EXITING")
            # exit()

            # print("computed_reward_agent",computed_reward,flush=True)

            #all envs
            self.ep_iteration+=1


            return value, action, action_log_probs, local_planner_actions, main_inputs, computed_reward, self.test_recurrent_hidden_states,
    ############################################################################

###############################################

##########################################################################
class DDPPOAgent_PointNav():
    def __init__(self, config: Config, batch_size=1):
        # print("DDPPOAgent_PointNav")
        self.batch_size=batch_size
        # if "ObjectNav" in config.TASK_CONFIG.TASK.TYPE:
        #     OBJECT_CATEGORIES_NUM = 20
        #     # OBJECT_CATEGORIES_NUM = 80
        #     spaces = {
        #         "objectgoal": Box(
        #             low=0, high=OBJECT_CATEGORIES_NUM, shape=(1,), dtype=np.int64
        #         ),
        #         "compass": Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float),
        #         "gps": Box(
        #             low=np.finfo(np.float32).min,
        #             high=np.finfo(np.float32).max,
        #             shape=(2,),
        #             dtype=np.float32,
        #         ),
        #     }
        # else:
        #     spaces = {
        #         "pointgoal": Box(
        #             low=np.finfo(np.float32).min,
        #             high=np.finfo(np.float32).max,
        #             shape=(2,),
        #             dtype=np.float32,
        #         )
        #     }

        spaces = {
            "pointgoal_with_gps_compass": Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            )
        }

        if config.INPUT_TYPE in ["depth", "rgbd"]:
            spaces["depth"] = Box(
                low=0,
                high=1,
                shape=(
                    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT,
                    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH,
                    1,
                ),
                dtype=np.float32,
            )

        if config.INPUT_TYPE in ["rgb", "rgbd"]:
            spaces["rgb"] = Box(
                low=0,
                high=255,
                shape=(
                    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT,
                    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH,
                    3,
                ),
                dtype=np.uint8,
            )
        observation_spaces = gymDict(spaces)

        action_space = Discrete(len(config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS))

        # self.device = torch.device("cuda:{}".format(config.TORCH_GPU_ID))
        # self.device = config.BEYOND_TRAINER.DEVICE
        self.device = config.BEYOND.DEVICE
        self.hidden_size = config.RL.PPO.hidden_size

        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        _seed_numba(config.RANDOM_SEED)
        torch.random.manual_seed(config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        policy_arguments = OrderedDict(
            observation_space=observation_spaces,
            action_space=action_space,
            hidden_size=self.hidden_size,
            rnn_type=config.RL.DDPPO.rnn_type,
            num_recurrent_layers=config.RL.DDPPO.num_recurrent_layers,
            backbone=config.RL.DDPPO.backbone,
            normalize_visual_inputs="rgb"
            if config.INPUT_TYPE in ["rgb", "rgbd"]
            else False,
        )
        # if "ObjectNav" not in config.TASK_CONFIG.TASK.TYPE:
        #     policy_arguments[
        #         "goal_sensor_uuid"
        #     ] = config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID

        # policy_arguments[
        #     "goal_sensor_uuid"
        # ] = "pointgoal_with_gps_compass"

        self.actor_critic = PointNavResNetPolicy(**policy_arguments)
        self.actor_critic.to(self.device)

        if config.BEYOND.PLANNER.LOAD_CHECKPOINT:
            ckpt = torch.load(config.BEYOND.PLANNER.CHECKPOINT, map_location=self.device)
            # print(f"Checkpoint loaded: {config.MODEL_PATH}")
            #  Filter only actor_critic weights
            # print("ckpt['state_dict'].items()",ckpt["state_dict"].keys())

            self.actor_critic.load_state_dict(
                {
                    k.replace("actor_critic.", ""): v
                    for k, v in ckpt["state_dict"].items()
                    if "actor_critic" in k
                }
            )

        else:
            habitat.logger.error(
                "Model checkpoint wasn't loaded, evaluating " "a random model."
            )

        self.test_recurrent_hidden_states = None
        self.not_done_masks = None
        self.prev_actions = None


    def eval(self):
        self.actor_critic.eval()

    #code for version 2021
    # def reset(self) -> None:
    #     self.test_recurrent_hidden_states = torch.zeros(
    #         1,
    #         self.actor_critic.net.num_recurrent_layers,
    #         self.hidden_size,
    #         device=self.device,
    #     )
    #     self.not_done_masks = torch.zeros(1, 1, device=self.device, dtype=torch.bool)
    #     self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)

    #version 2020
    def reset(self):
        #num_recurrent_layers is 2

        self.test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            1,
            self.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(1, 1, device=self.device)
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)

    def reset_batch(self):
        self.test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.batch_size,
            self.hidden_size,
            device=self.device,
        )
        #seq,batch,feature
        self.not_done_masks = torch.zeros(self.batch_size, 1, device=self.device)
        self.prev_actions = torch.zeros(self.batch_size, 1, dtype=torch.long, device=self.device)

    def reset_idx(self, idx):
        tmp = self.test_recurrent_hidden_states.permute(1,0,2)
        tmp[:,idx,:]=0
        tmp = tmp.permute(1,0,2)
        self.test_recurrent_hidden_states= tmp
        self.not_done_masks[idx]=0
        self.prev_actions[idx]=0


    def act(self, observations):
        # batch = batch_obs([observations], device=self.device)
        batch = observations
        # for sensor in batch:
        #     #add seq dimension
        #     batch[sensor] = batch[sensor].unsqueeze(0)
        # print("internal planner batch",batch.shape,flush=True)
        # exit()

        with torch.no_grad():
            _, action, _, self.test_recurrent_hidden_states = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=False,
            )
            #  Make masks not done till reset (end of episode) will be called
            # self.not_done_masks.fill_(1.0)
            self.not_done_masks[:,:]=1.0
            self.prev_actions.copy_(action)

        return action
##########################################################################
##########################################################################


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

    # for obs in observations:
    #     for sensor in obs:
    #         batch[sensor].append(_to_tensor(obs[sensor]))

    for obs in observations:
        for sensor in obs:
            if(sensor == 'top_down_map'):
                batch[sensor].append((obs[sensor]))
            else:
                batch[sensor].append(_to_tensor(obs[sensor]))


    # for sensor in batch:
    #     batch[sensor] = (
    #         torch.stack(batch[sensor], dim=0)
    #         .to(device=device)
    #         .to(dtype=torch.float)
    #     )

    # for sensor in batch:
    #     batch[sensor] = (
    #         torch.stack(batch[sensor], dim=0)
    #         .to(device=device)
    #     )

    for sensor in batch:
        if(sensor != 'top_down_map'):
            batch[sensor] = (
                torch.stack(batch[sensor], dim=0)
                .to(device=device)
            )

    return batch
