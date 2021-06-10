import torch
import numpy as np
# from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast
import argparse
import os
import numba
import random
import contextlib

from model import ChannelPool
# from beyond_agent import BeyondAgent
from beyond_agent_without_internal_mapper import BeyondAgentWithoutInternalMapper
from fog_of_war import reveal_fog_of_war


from baselines_config_default import get_config

####################
#for debug
import cv2

import torchvision.transforms.functional as TF
from PIL import Image
# from torchvision.transforms.functional import InterpolationMode
# import torchvision.transforms.functional.InterpolationMode as InterpolationMode
####################

# // Normalizes any number to an arbitrary range
# // by assuming the range wraps around when going below min or above max
# double normalize( const double value, const double start, const double end )
# {
#   const double width       = end - start   ;   //
#   const double offsetValue = value - start ;   // value relative to 0
#
#   return ( offsetValue - ( floor( offsetValue / width ) * width ) ) + start ;
#   // + start to reset back to start of original range
# }

################################################################################
import urizen as uz
from urizen.core.entity_collection import C

def _vg_pillow_pixelated(M, scale=1, show=True, filepath=None):
    w, h = M.get_size()
    pixels = []
    im = Image.new('L', (w, h))
    for line in M.cells:
        for cell in line:
            if isinstance(cell, C.wall_dungeon_rough):
                pixels.append((1))
            if isinstance(cell, C.floor):
                pixels.append((0))
    im.putdata(pixels)
    map_small = np.array(im)
    im = im.resize((w*scale, h*scale), resample=Image.NEAREST)
    map = np.array(im)
    return map, map_small


def generate_rooms():
    M = uz.dungeon_bsp_tree(32, 32)
    return _vg_pillow_pixelated(M, scale=8)

def seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def generate_valid_map():
    map,map_small = generate_rooms()
    test= np.sum(map[128-1:128+2,128-1:128+2])
    while(test!=0):
        #regenerate
        map,map_small = generate_rooms()
        test= np.sum(map[128-1:128+2,128-1:128+2])
    return map,map_small

# map,map_small = generate_valid_map()
################################################################################

def translate_tensor_in2D(input, shift):
    '''
    The padding size by which to pad some dimensions of input are described starting from the last dimension and moving forward.

    (padding_left,padding_right) ;
    to pad the last 2 dimensions of the input tensor, then use (padding_left,padding_right),
    (padding_left,padding_right, padding_top,padding_bottom);
    '''

    #shift in form [-inf,+inf],[-inf,+inf] y,x
    shift_tuple = []

    if(shift[1]<0):
        #top
        tmp = (-1*shift[1],0)
        shift_tuple.append(tmp)
    elif(shift[1]>0):
        #bottom
        tmp = (0,shift[1])
        shift_tuple.append(tmp)
    else:
        tmp = (0,0)
        shift_tuple.append(tmp)

    if(shift[0]<0):
        #left
        tmp = (-1*shift[0],0)
        shift_tuple.append(tmp)
    elif(shift[0]>0):
        #right
        tmp = (0,shift[0])
        shift_tuple.append(tmp)
    else:
        tmp = (0,0)
        shift_tuple.append(tmp)

    shift_tuple = np.array(shift_tuple).flatten()
    if (np.sum(shift_tuple)==0):
        return input

    shift_tuple = tuple(shift_tuple)

    # print("shift_tuple",shift_tuple)
    # print("input.shape",input.shape)
    out = torch.nn.functional.pad(input, shift_tuple, "constant", 0)

    # print("out.shape",out.shape)
    # print(shift_tuple[3],out.shape[1],-shift_tuple[2], shift_tuple[1],out.shape[2],-shift_tuple[0])
    # print(shift_tuple[3],out.shape[1]-shift_tuple[2], shift_tuple[1],out.shape[2]-shift_tuple[0])
    out = out[:, shift_tuple[3]:out.shape[1]-shift_tuple[2], shift_tuple[1]:out.shape[2]-shift_tuple[0]  ]
    # print("out.shape",out.shape)

    # # np.pad(x,((0,0),(5,0)), mode='constant')[:, :-5]
    # # np.pad(x,((0,0),(0,5)), mode='constant')[:, 5:])

    # # shift_tuple = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
    # # out = F.pad(input, shift_tuple, "constant", 0)

    return out
################################################################################

def angle_between_vectors(vector_1,vector_2):
    #assume that the vector is p0 origin 0,0 and p1 represented by vector_n
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    #in radians
    return angle

def normalize(value, start, end):
    width = end - start
    offsetValue = value - start
    return ( offsetValue - ( np.floor( offsetValue / width ) * width ) ) + start

# def rotate_image(image, angle):
#   image_center = tuple(np.array(image.shape[1::-1]) / 2)
#   rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#   result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST)
#   return result

def rotate_image(image, angle):
    # result = TF.rotate(img=image, angle=angle, interpolation=InterpolationMode.NEAREST)
    result = TF.rotate(img=image, angle=angle, resample=Image.NEAREST)
    return result

class VectorEnv_without_habitat():
    def __init__(self, config, device="cuda", batch_size=1, n_ep=21):
        self.config = config
        self.device = device
        self.n_ep = n_ep
        self.envs = [SemanticEnv(device=self.device, n_ep=self.n_ep,seed=config.RANDOM_SEED+i+1,env_id=i) for i in range(batch_size)]
        # self.envs = [SemanticEnv(device=self.device, n_ep=self.n_ep,seed=config.RANDOM_SEED+i) for i in range(batch_size)]
        # self.envs = [SemanticEnv(device=self.device, n_ep=self.n_ep,seed=config.RANDOM_SEED) for i in range(batch_size)]
        self.outputs = [None for _ in range(batch_size)]

        #since it is a single process only once is necessary
        self.envs[0].seed(config.RANDOM_SEED)
        for env in self.envs:
            env.init_eps()

    def reset(self):
        observations = []
        for env in self.envs:
            observations.append(env.reset())
        return observations

    def async_step_at(self, env_idx, action):
        self.outputs[env_idx] = self.envs[env_idx].step(action)

    def wait_step_at(self, env_idx):
        #tuple observations, reward, done, info
        tmp = self.outputs[env_idx]
        if(tmp[-2]):#is_done
            self.envs[env_idx].reset()
        return tmp

    # def send_computed_reward(self, reward):
    #     for i in range(len(self.envs)):
    #         self.envs[i].y_pred_coo = reward[i]

    # def send_computed_reward(self, reward):
    #     for i in range(len(self.envs)):
    #         self.envs[i].computed_reward = reward[i]

    def send_computed_reward(self, reward):
        for i in range(len(self.envs)):
            self.envs[i].computed_reward = reward[i]['rmse']
            self.envs[i].y_pred_coo = reward[i]['y_pred_map_coo_long']

    def close(self):
        pass

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

class SemanticEnv():
    def __init__(self, device="cuda", ep_list=None, n_ep=21, cycle_enabled=True, classes=21, max_steps=15, seed=42, env_id=0):
        self.env_id = env_id
        self._seed = seed
        self.device = device
        self.n_ep = n_ep
        self.current_ep_id = -1
        self.n_classes = classes
        self.cycle_enabled = cycle_enabled
        self.episode_over =  False
        self.iteration = 0
        self.agent_orientation = 0
        self.collide_count = 0
        self.max_steps = max_steps
        self.metrics = {"distance_to_goal":0,"spl":0,"success":0,"softspl":0,"collisions":0,"steps":0}
        # self.max_possible_error = 181 #rounded hipotenuse of 128,128
        self.max_possible_error = 362 #rounded hipotenuse of 256,256
        # self.success_distance = 0.15#because of diagonal hip
        # self.success_distance = 0.2#because of diagonal hip + margin due to corners
        self.success_distance = 0.25#because of diagonal hip + margin due to corners

        self.max_collisions = 10
        self.computed_reward = 0
        '''
        # area of a trapezium isosceles of 11px to 66px
        due is because of hfov of 79 res of 640x480 w,h near clip of 0.6 and max_depth of 5meters
        for cell res of 0.1 is 1330px
        so for cell res of 0.05 is the double
        '''
        self.max_possible_explored_per_step = 2660

        # self.cell_resolution = 0.1#meters
        # self.cell_resolution_inverse = 10

        self.cell_resolution = 0.05#meters
        self.cell_resolution_inverse = 20

        self.ep_list = None

        self.pool = ChannelPool(1)
        self.map = None
        self.agent_internal_map = None
        self._previous_measure = 0
        self.epoch = 0
        # self.epoch_limit_to_repeat = 4
        # self.epoch_limit_to_repeat = 512
        self.epoch_limit_to_repeat = 256
        # self.epoch_limit_to_repeat = 128
        # self.epoch_limit_to_repeat = 64
        # self.epoch_limit_to_repeat = 64
        self.y_pred_coo = np.zeros((2),dtype=np.int32)
        self.y_pred_coo_previous = np.zeros((2),dtype=np.int32)
        self.angle_between_predictions = None
        self._previous_measure_collisions = 0
        self.agent_turn_angle = 30

    def send_computed_reward(self, reward):
        self.computed_reward = reward['rmse']
        self.y_pred_coo = reward['y_pred_map_coo_long']

    def init_eps(self):
        print("init_eps",self.env_id,flush=True)
        size=self.n_ep
        # with temp_seed(self._seed):
        #     signs = np.random.choice(np.array([-1,1]),size=size)
        #     angle = (np.random.random_sample((size,))*np.pi) * signs
        angle_between = np.deg2rad(360/self.n_classes)
        # angle = np.arange(-10,11)*angle_between
        angle = np.arange(0,21)*angle_between
        angle = normalize(angle, -np.pi, np.pi)

        # print("angle",angle)

        if self.ep_list is None:
            # self.ep_list = np.random.randint(low=1,high=22,size=n_ep)
            # self.ep_list = np.arange(start=0,stop=21)

            self.ep_list=[]
            for i in range(size):
                # print(angle[i])
                self.ep_list.append({'objectgoal':i % self.n_classes,'agent_starting_orientation':angle[i]})

            # rng = np.random.default_rng()
            # numbers = rng.choice(np.arange(start=1,stop=22), size=n_ep, replace=False)
        # print("self.ep_list",self.ep_list)
        self.create_explorable_map()

    @staticmethod
    @numba.njit
    def _seed_numba(seed: int):
        random.seed(seed)
        np.random.seed(seed)

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self._seed_numba(seed)
        torch.random.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def init_metric_spl(self):
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = self.metrics['distance_to_goal']
        self._previous_position = self.agent_current_pos

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric_spl(self):
        ep_success = self.metrics['success']
        current_position = self.agent_current_pos
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )*self.cell_resolution
        self._previous_position = current_position

        self.metrics['spl'] = ep_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )

    def update_metric_softspl(self):
        current_position = self.agent_current_pos
        distance_to_target = self.metrics['distance_to_goal']

        ep_soft_success = max(
            0, (1 - distance_to_target / self._start_end_episode_distance)
        )

        # self._agent_episode_distance += self._euclidean_distance(
        #     current_position, self._previous_position
        # )

        # self._previous_position = current_position

        self.metrics['softspl'] = ep_soft_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )

    def update_metric_distance_pred_to_goal(self):

        if self.iteration > 1:
            #origin is the agent so translate to it
            current_vector = self.y_pred_coo-128
            previous_vector = self.y_pred_coo_previous-128

            self.angle_between_predictions = angle_between_vectors(current_vector,previous_vector)
            self.y_pred_coo_previous = self.y_pred_coo
        else:
            self.y_pred_coo_previous = self.y_pred_coo

        # diff_in_cells = self.y_pred_coo-self.ep_list[self.current_ep_id]['target_pos'].cpu().numpy()
        # euclid_dist_in_cells = np.linalg.norm(diff_in_cells, ord=2, axis=-1)
        # self.metrics['distance_pred_to_goal']=euclid_dist_in_cells
        # self.metrics['distance_pred_to_goal']=np.min(self.metrics['distance_pred_to_goal'])

    def update_metric_distance_to_goal(self):
        diff_in_cells = self.agent_current_pos-self.ep_list[self.current_ep_id]['target_pos'].cpu().numpy()
        # print("diff_in_cells",diff_in_cells)
        euclid_dist_in_cells = np.linalg.norm(diff_in_cells, ord=2, axis=-1)
        # print("euclid_dist_in_cells",euclid_dist_in_cells)
        # cell_res = 0.1#meters
        self.metrics['distance_to_goal']=euclid_dist_in_cells*self.cell_resolution
        self.metrics['distance_to_goal']=np.min(self.metrics['distance_to_goal'])

    # def update_metric_success(self):
    #     if self.metrics['distance_to_goal'] <= self.success_distance:
    #         self.metrics['success']=1
    #         self.episode_over = True
    #     self.metrics['steps']=self.iteration

    def update_metric_success(self):
        #force early stop to avoid wrong # OPTIMIZE:
        if self.current_explored_target_diff > 0:
            self.metrics['success']=1
            self.episode_over = True
        self.metrics['steps']=self.iteration

    def step(self, action):
        r"""Perform an action in the environment and return observations.
        :param action: action (belonging to :ref:`action_space`) to be
            performed inside the environment. Action is a name or index of
            allowed task's action and action arguments (belonging to action's
            :ref:`action_space`) to support parametrized and continuous
            actions.
        :return: observations after taking action in environment.
        """

        # if (self.iteration == 500):
        # if ((self.iteration == self.max_steps)or(self.collide_count == self.max_collisions)or(self.computed_reward == -1) ):
        if ((self.iteration == self.max_steps)or(self.collide_count == self.max_collisions) ):
            self.episode_over = True
        else:
            self.iteration += 1

        # POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
        observations = None
        angle = np.deg2rad(0)
        agent_current_pos_gps = np.array([0.0, 0.0],dtype=np.float32)
        if(action==0):
            self.episode_over = True
        elif(action==1):
            y = np.sin(self.agent_orientation - (np.pi/2)) * -1
            x = np.cos(self.agent_orientation - (np.pi/2))

            # cell_res = 10#0.1meters
            agent_current_pos_gps = np.rint(np.array([y, x],dtype=np.float32) * 0.25 * self.cell_resolution_inverse)
            # print("agent_current_pos_gps",agent_current_pos_gps)
            # exit()

            tmp = np.rint(self.agent_current_pos + agent_current_pos_gps).astype(np.int32)

            # print("self.agent_internal_map", self.agent_internal_map.shape )
            # print("tmp",tmp)
            # print("self.agent_internal_map", self.agent_internal_map[-4][tmp[0]][tmp[1]] )

            if( self.agent_internal_map[-4][tmp[0]][tmp[1]] == 0 ):#move only if it is free
                self.agent_current_pos = np.rint(self.agent_current_pos + agent_current_pos_gps).astype(np.int32)
            else:
                self._previous_measure_collisions = self.collide_count
                self.collide_count += 1
                # print("self.collide_count",self.collide_count)

        elif(action==2):
            #update compass
            angle = np.deg2rad(self.agent_turn_angle)
        elif(action==3):
            #update compass
            angle = np.deg2rad(-self.agent_turn_angle)

        self.agent_orientation += angle
        #ensure -pi,pi range
        self.agent_orientation = normalize(self.agent_orientation, -np.pi, np.pi)
        # print("self.agent_orientation",self.agent_orientation)

        self.get_step_map()

        #gps is in meters
        # cell_size = 0.1#0.1meters
        gps = (self.agent_current_pos-np.array([128,128]))
        # print("gps",gps)
        gps = gps *self.cell_resolution
        gps[1] = -1*gps[1]

        compass = np.array([self.agent_orientation],dtype=np.float32)

        # print("gps step",gps)
        # print("compass step",compass)

        self.fog_of_war_mask = reveal_fog_of_war(
            top_down_map=self.agent_internal_map[-4].detach().clone().float().cpu().numpy(),
            current_fog_of_war_mask=self.fog_of_war_mask,
            current_point=np.array([self.agent_current_pos[0], self.agent_current_pos[1]],dtype=np.int32),
            current_angle=compass.item(),
            fov=79,
            max_line_len=100,
            )

        map = self.agent_internal_map.detach().clone()

        # print(torch.nonzero(map[1]))
        # cv2.imshow("map", ((map[-4]+map[-3])*255).byte().cpu().numpy() )
        # cv2.waitKey()

        #inplace operation
        # map[-1].fill_(0)
        # map[-1, pred_coo[0]-1:pred_coo[0]+2 , pred_coo[1]-1:pred_coo[1]+2  ] = 1.0

        # map[:23] = self.agent_internal_map[:23] * self.fog_of_war_mask.astype(np.float32)
        map[:23] = self.agent_internal_map[:23] * torch.from_numpy(self.fog_of_war_mask.astype(np.float32)).to(self.device)

        '''
        to reward exploration
        '''
        tmp = torch.sum(map[-4])
        self.current_explored_diff = tmp - self.previous_explored
        self.previous_explored = tmp

        tmp = torch.sum(map[ self.ep_list[self.current_ep_id]['objectgoal']+1 ]  )
        self.current_explored_target_diff = tmp - self.previous_explored_target

        # #force early stop to avoid wrong # OPTIMIZE:
        # if self.current_explored_target_diff > 0:
        #     self.episode_over = True

        self.previous_explored_target = tmp
        ########################################################################

        # # ########################################################################
        # # '''
        # # FOR # DEBUG:
        # # '''
        # # cv2.imshow("map-3", ((map[-4]+map[-3]+map[-1])*255).byte().cpu().numpy() )
        # trash = ((map[-4]+map[-3])*255).byte().cpu().numpy()
        # trash = trash+(map[-1].byte().cpu().numpy()*128)
        #
        # cv2.imshow("map-3", trash )
        # cv2.imshow("map-2", ((map[-4]+map[-2])*255).byte().cpu().numpy() )
        #
        # arrow = cv2.imread("extras/arrow.png")
        # arrow = rotate_image(torch.from_numpy(arrow), np.rad2deg(compass.item()) )
        # cv2.imshow("arrow",arrow.numpy())
        #
        # cv2.waitKey()
        # # ########################################################################

        # ########################################################################
        # #force the agent to be at the center always
        # print(self.agent_current_pos)
        # shift = (self.agent_current_pos-np.array([128,128]))*-1
        shift = (self.agent_current_pos-np.array([128,128]))
        map = translate_tensor_in2D(map, shift)
        # print("map[-2][128][128]",map[-2][128][128])
        ########################################################################

        objectgoal = np.array([ self.ep_list[self.current_ep_id]['objectgoal'] ] )#net expect starting from 0

        observations = {"map":map,"gps":gps,"compass":compass,"objectgoal":objectgoal}


        info = self.get_info(observations)
        done = self.get_done(observations)
        reward = self.get_reward(observations)

        return observations, reward, done, info

    @property
    def current_target(self):
        return self.ep_list[self.current_ep_id]['objectgoal']


    # def get_reward(self,observations):
    #     reward = self.max_possible_error - self.metrics['distance_pred_to_goal']
    #     # reward = (self._previous_measure - self.metrics['distance_pred_to_goal'])
    #     # self._previous_measure = self.metrics['distance_pred_to_goal']
    #     return reward

    # ############################################################################
    # def get_reward(self,observations):
    #     #slack punish
    #     reward = -1e-4
    #
    #     '''
    #     collision -1
    #     large angle 0.1*np.pi ~= 0.314
    #
    #     max_possible_explored_per_step_std 0.1
    #     max_possible_explored_per_step_target ~0.33
    #
    #     distance_pred_to_goal [0,0.2]
    #
    #     wrong step -step_size*2 == -0.25*2
    #     correct step step_size*2 == 0.25*2
    #     success 5.0
    #     '''
    #
    #     standard_exploration_scaling_factor = 1.0
    #     #target represent a tiny fraction of the pixels on the map so the scaling should be higher than std
    #     target_exploration_scaling_factor = 100.0
    #     distance_pred_to_goal_scaling_factor = 0.2
    #     success_scaling_factor = 5.0
    #     step_scaling_factor = 2.0
    #     angle_diff_scaling_factor = 0.1
    #
    #     explored = self.current_explored_diff *(1/self.max_possible_explored_per_step) * standard_exploration_scaling_factor
    #     explored_target = self.current_explored_target_diff *(1/self.max_possible_explored_per_step) * target_exploration_scaling_factor
    #
    #     reward += explored
    #     reward += explored_target
    #
    #     if not(self.angle_between_predictions is None):
    #         ang = np.deg2rad(self.agent_turn_angle)
    #         if( (self.angle_between_predictions > ang)or(self.angle_between_predictions < -ang) ):
    #             # punish_large_angle
    #             # reward -= 1.0
    #             reward -= np.abs(self.angle_between_predictions-ang)*angle_diff_scaling_factor
    #
    #     #small distance big reward
    #     #encourage pred to be closer to the objective instead on the edges of the prediction ceiling
    #     reward += (1 - (self.metrics['distance_pred_to_goal']*(1/self.max_possible_error)) )*distance_pred_to_goal_scaling_factor
    #
    #     reward += (self._previous_measure - self.metrics['distance_to_goal'])*step_scaling_factor
    #     reward += self.metrics['success']*success_scaling_factor
    #     reward -= self.metrics['collisions'] - self._previous_measure_collisions
    #
    #     self._previous_measure = self.metrics['distance_to_goal']
    #     return reward
    # ############################################################################

    ############################################################################
    def get_reward(self,observations):
        #slack punish
        reward = -1e-4

        '''
        collision -1
        large angle 0.1*np.pi ~= 0.314

        max_possible_explored_per_step_std 0.1
        max_possible_explored_per_step_target ~0.33

        distance_pred_to_goal [0,0.2]

        wrong step -step_size*2 == -0.25*2
        correct step step_size*2 == 0.25*2
        success 5.0
        '''

        standard_exploration_scaling_factor = 10.0
        # #target represent a tiny fraction of the pixels on the map so the scaling should be higher than std
        target_exploration_scaling_factor = 10.0
        # distance_pred_to_goal_scaling_factor = 0.15
        # success_scaling_factor = 10.0
        success_scaling_factor = 2.5
        step_scaling_factor = 1.0
        # step_scaling_factor = 20.0*(1/256)
        # step_scaling_factor = 4.0
        # angle_diff_scaling_factor = 0.05
        angle_diff_scaling_factor = 1.0

        # explored = self.current_explored_diff *(1/self.max_possible_explored_per_step)*standard_exploration_scaling_factor
        # explored_target = self.current_explored_target_diff*target_exploration_scaling_factor
        #
        # reward += explored
        # reward += explored_target

        # if not(self.angle_between_predictions is None):
        #     ang = np.deg2rad(self.agent_turn_angle)
        #     if( (self.angle_between_predictions > ang)or(self.angle_between_predictions < -ang) ):
        #         # punish_large_angle
        #         # reward -= 1.0
        #         reward -= np.abs(self.angle_between_predictions-ang)*angle_diff_scaling_factor
        #
        # #small distance big reward
        # #encourage pred to be closer to the objective instead on the edges of the prediction ceiling
        # # reward += (1 - (self.metrics['distance_pred_to_goal']*(1/self.max_possible_error)) )*distance_pred_to_goal_scaling_factor
        # reward -= (self.metrics['distance_pred_to_goal']*(1/self.max_possible_error)) *distance_pred_to_goal_scaling_factor

        reward += (self._previous_measure - self.metrics['distance_to_goal'])*step_scaling_factor
        reward += self.metrics['success']*success_scaling_factor
        # # reward -= self.metrics['collisions'] - self._previous_measure_collisions
        #
        self._previous_measure = self.metrics['distance_to_goal']

        # if (self.computed_reward == -1):
        #     #disable reward when using heuristic
        #     reward = 0
        # print("self.computed_reward",self.computed_reward,self.computed_reward*(1/256))
        # reward -= self.computed_reward*(1/256)

        '''
        ideally reward should be within -1,1 range

        1+9+0.25+10
        '''
        # reward = reward*(1/20.25)

        return reward
    ############################################################################


    def get_done(self,observations):
        return self.episode_over
    def get_info(self,observations):
        # self.update_metric_distance_pred_to_goal()
        self.update_metric_distance_to_goal()
        self.update_metric_success()
        self.update_metric_spl()
        self.update_metric_softspl()

        self.metrics['collisions']=self.collide_count
        return self.metrics

    def get_step_map(self):
        data_idxs = self.agent_current_pos

        #0=-4,1=-3,2=-2,3=-1
        #occupied
        # map[-4] = self.pool(map[:-4].unsqueeze(0)).squeeze(0)
        #past_locations
        # map[-3]
        # #current_location
        # map[-2]
        #depend on prev_pred
        # map[-1]

        #clear past pred
        self.agent_internal_map[-2].fill_(0)
        self.agent_internal_map[-3:-1, data_idxs[0]-1:data_idxs[0]+2 , data_idxs[1]-1:data_idxs[1]+2  ] = 1.0

    def reset(self):
        r"""Resets the environments and returns the initial observations.
        :return: initial observations from the environment.
        """
        self.metrics = {"distance_to_goal":0,"spl":0,"success":0,"softspl":0,"collisions":0,"steps":0}
        self.iteration = 0
        self.collide_count = 0
        self._previous_measure_collisions = 0
        self.episode_over = False

        self.angle_between_predictions = None
        self.y_pred_coo = np.zeros((2),dtype=np.int32)
        self.y_pred_coo_previous = np.zeros((2),dtype=np.int32)

        if(self.cycle_enabled):
            if( self.current_ep_id+1) == len(self.ep_list ):
                # print("end of epoch")
                self.epoch+=1
                # if (self.epoch % self.epoch_limit_to_repeat == 0):
                #     self.init_eps()
                #     self.epoch=0

            self.current_ep_id = (self.current_ep_id+1)%len(self.ep_list)
        else:
            if( self.current_ep_id+1) == len(self.ep_list ):
                print("end of epoch")
                return None
            else:
                self.current_ep_id += 1

        self.agent_orientation = self.ep_list[self.current_ep_id]['agent_starting_orientation']

        # self.create_explorable_map()
        self.reset_explorable_map()

        '''
        Rotate to improve diversity
        '''
        # if (self.iteration % == 0):
        # sign = -1**(self.current_ep_id % 2)
        # sign = -1 if (self.env_id % 2 == 0) else 1
        # sign = np.random.choice(np.array([-1,1]),size=1).item()
        # # print("sign",sign)
        # map_tmp = rotate_image(self.map,  np.rad2deg(sign*self.agent_orientation))
        # map_tmp = self.map

        #internal compass start with 0 always
        self.agent_orientation = 0


        # self.map[-4] = self.pool( self.map[:-4].unsqueeze(0) ).squeeze(0)

        # data_idxs = self.agent_current_pos


        # observations = None
        # map = torch.zeros(26,256,256,device=self.device)

        self.get_step_map()

        gps = np.array([0.0, 0.0],dtype=np.float32)
        # compass = np.array([0.0],dtype=np.float32)
        compass = np.array([self.agent_orientation],dtype=np.float32)

        # print("gps reset",gps)
        # print("compass reset",compass)

        self.fog_of_war_mask = reveal_fog_of_war(
            top_down_map=self.agent_internal_map[-4].detach().clone().float().cpu().numpy(),
            current_fog_of_war_mask=self.fog_of_war_mask,
            current_point=np.array([128, 128],dtype=np.int32),
            current_angle=compass.item(),
            fov=79,
            max_line_len=100,
            )

        if not ('target_pos' in self.ep_list[self.current_ep_id]):
            self.ep_list[self.current_ep_id]['target_pos']=torch.nonzero( self.agent_internal_map[self.ep_list[self.current_ep_id]['objectgoal']+1],as_tuple=False )
        # print("self.ep_list[self.current_ep_id]['target_pos']",self.ep_list[self.current_ep_id]['target_pos'])

        map = self.agent_internal_map.detach().clone()
        # cv2.imshow("map", ((map[-4]+map[-3])*255).byte().cpu().numpy() )
        # cv2.waitKey()
        # map[:23] = self.agent_internal_map[:23] * self.fog_of_war_mask.astype(np.float32)
        map[:23] = self.agent_internal_map[:23] * torch.from_numpy(self.fog_of_war_mask.astype(np.float32)).to(self.device)

        ########################################################################
        '''
        to reward exploration
        '''
        self.current_explored_diff = 0
        self.previous_explored = torch.sum(map[-4])

        self.current_explored_target_diff = 0
        self.previous_explored_target = torch.sum(map[ self.ep_list[self.current_ep_id]['objectgoal']+1 ]  )
        ########################################################################


        # cv2.imshow("mapr", ((map[-4]+map[-3])*255).byte().cpu().numpy() )


        objectgoal = np.array([ self.ep_list[self.current_ep_id]['objectgoal'] ])#net expect starting from 0
        # print("objectgoal",objectgoal,"self.ep_list[self.current_ep_id]['objectgoal']",self.ep_list[self.current_ep_id]['objectgoal'])

        # depth = torch.ones(480,640,device=self.device)*0.79#max
        # depth = torch.ones(480,640,device=self.device)*0.03#min

        observations = {"map":map,"gps":gps,"compass":compass,"objectgoal":objectgoal}
        # observations = {"map":map,"gps":gps,"compass":compass,"objectgoal":objectgoal,"depth":depth}
        # print("env observations",observations)

        # cv2.imshow("map",(self.agent_internal_map[-4]*255).cpu().numpy())
        # cv2.waitKey()

        self.update_metric_distance_to_goal()
        self.init_metric_spl()
        self.get_info(observations)
        # self._previous_measure = self.metrics['distance_pred_to_goal']
        self._previous_measure = self.metrics['distance_to_goal']

        return observations

    def plot_map(self, channel):
        cv2.imshow("map", (self.map[channel]*255).byte().cpu().numpy() )
        cv2.waitKey()

    # def populate_ep_map(self, target_idx):
    #
    #     # pos = np.random.randint(low=self.agent_current_pos-9,high=self.agent_current_pos+10,size=2)
    #     with temp_seed(self._seed):
    #         pos = np.random.randint(low=self.agent_current_pos-18,high=self.agent_current_pos+19,size=2)
    #     # pos = np.array([250,250],dtype=np.int32)
    #
    #     while( not( (pos[0] < self.agent_current_pos[0]-3)or(pos[0] > self.agent_current_pos[0]+3)or(pos[1] < self.agent_current_pos[1]-3)or(pos[1] > self.agent_current_pos[1]+3) ) ):
    #         #invalid
    #         # pos = np.random.randint(low=self.agent_current_pos-9,high=self.agent_current_pos+10,size=2)
    #         with temp_seed(self._seed):
    #             pos = np.random.randint(low=self.agent_current_pos-18,high=self.agent_current_pos+19,size=2)
    #         # pos = np.array([250,250],dtype=np.int32)
    #
    #     # print("pos",pos,"target_idx",target_idx)
    #     # self.map[target_idx][pos[0]][pos[1]] = 1.0
    #     self.map[target_idx, pos[0]-1:pos[0]+2, pos[1]-1:pos[1]+2 ] = 1.0
    #     #to ensure exclusive per class
    #     self.map[0, pos[0]-1:pos[0]+2, pos[1]-1:pos[1]+2 ] = 0.0
    # ############################################################################
    # def populate_ep_map(self, map_small):
    #     valid_coo = torch.nonzero(torch.from_numpy(map_small))
    #     with temp_seed(self._seed):
    #         idxs = np.random.choice(np.arange(valid_coo.shape[0]),size=self.n_classes)
    #
    #     target_idx = 0
    #     for idx in idxs:
    #         target_idx+=1
    #         pos = valid_coo[idx]
    #         pos = (pos*8)+128
    #
    #         self.map[target_idx, pos[0]-1:pos[0]+2, pos[1]-1:pos[1]+2 ] = 1.0
    #         self.map[0, pos[0]-1:pos[0]+2, pos[1]-1:pos[1]+2 ] = 0.0
    ############################################################################
    # ############################################################################
    def populate_ep_map(self, map_small):
        angle_between = 360/self.n_classes
        # radius = 25#px
        # radius = 50#px
        # radius = 110#px
        radius = 105#px

        # initial_angle = np.deg2rad(270)
        # initial_angle = np.deg2rad(120)
        initial_angle = np.deg2rad(150)
        with temp_seed(self._seed):
            # radius += np.random.randint(low=0, high=20, size=1).item()
            # initial_angle += np.deg2rad(np.random.randint(low=0, high=90, size=1).item())
            wiggle = np.random.randint(low=-2, high=2, size=self.n_classes*2)
            wiggle = wiggle.reshape(-1,2)
            wiggle_radius = np.random.randint(low=0, high=10, size=self.n_classes)

            # initial_angle = np.random.randint(low=0, high=self.n_classes, size=1).item()
            # initial_angle = np.random.choice(np.arange(self.n_classes),size=1)
            # initial_angle = np.deg2rad(angle_between*initial_angle)

        for i in range(self.n_classes):
            # print(i)
            target_idx = i+1
            # angle = np.deg2rad(angle_between*(i+initial_angle) )
            angle = -np.deg2rad(angle_between*(i) ) + initial_angle

            y = np.sin(angle)*(radius+wiggle_radius[i])
            x = np.cos(angle)*(radius+wiggle_radius[i])
            pos = np.array([y,x],dtype=np.int32)+256+wiggle[i]
            # print("pos",pos)

            self.map[0:22, pos[0]-1:pos[0]+2, pos[1]-1:pos[1]+2 ] = 0.0
            self.map[target_idx, pos[0]-1:pos[0]+2, pos[1]-1:pos[1]+2 ] = 1.0
    ############################################################################


    # def create_room_layout(self):
    #    '''
    #        dungeon like rooms
    #    '''
    #     map,map_small = generate_valid_map()
    #     #invert
    #     inv_map_small = (map_small*-1)+1
    #     #tweek center
    #     center = int(inv_map_small.shape[0]/2)
    #     inv_map_small[ center-3:center+3,center-3:center+3 ] = 0
    #     self.map_small_layout = inv_map_small
    #     self.map_layout = torch.from_numpy(map).to(self.device)

    def create_room_layout(self):
        '''
        simple square
        '''
        # map = np.ones((256,256))
        # map[4*8:(32-4)*8, 4*8:(32-4)*8] = 0.0

        map_small = np.ones((32,32))
        # map_small[4:32-4,4:32-4] = 0.0
        map_small[1:32-1,1:32-1] = 0.0

        #include white noise on channel 0 then draw zero circle over it
        map = np.ones((256,256))
        # map[4*8:(32-4)*8, 4*8:(32-4)*8] = 0.0
        map[1*8:(32-1)*8, 1*8:(32-1)*8] = 0.0
        with temp_seed(self._seed):
            # map = np.random.randint(low=0,high=2,size=(256,256))
            points_x = np.random.choice(np.arange(32,256-32), size=32)
            points_y = np.random.choice(np.arange(32,256-32), size=32)

        points = np.stack([points_x,points_y],axis=-1)
        # print(points)
        map[ points[:,:1],points[:,1:] ] = 1
        map[128-50:128+50,128-50:128+50] = 0.0
        # cv2.imshow("tmp",map*255)
        # cv2.waitKey()
        # map = cv2.circle(map.astype(np.int8), (128,128), 100, 0, -1)
        # cv2.imshow("tmp",map*255)
        # cv2.waitKey()
        map = map.astype(np.float32)

        # map[4*8:(32-4)*8, 4*8:(32-4)*8] = 0.0

        #invert
        inv_map_small = (map_small*-1)+1
        #tweek center
        center = int(inv_map_small.shape[0]/2)
        inv_map_small[ center-3:center+3,center-3:center+3 ] = 0
        self.map_small_layout = inv_map_small
        self.map_layout = torch.from_numpy(map).to(self.device)

    def reset_explorable_map(self):
        self.agent_internal_map[-3:].fill_(0)
        self.fog_of_war_mask = np.zeros( (256,256) )
        self.agent_current_pos = np.array([128,128])

    def create_explorable_map(self):
        '''
        Map will follow the beyond definitions
        each cell represent 0.1 meters
        FORWARD_STEP_SIZE: 0.25 meters
        TURN_ANGLE: 30 degrees
        SUCCESS_DISTANCE: 0.1
        '''

        #init map
        #0 to 21 is the semantic classes, 22=aux[0] is all occupied
        self.map = torch.zeros( 26,512,512 ,device=self.device)
        self.map[0].fill_(1)
        # self.map = torch.ones( 26,512,512 ,device=self.device)

        self.agent_current_pos = np.array([256,256])

        self.create_room_layout()

        #create room as bg 2m x 2m room
        #draw filled rectancle
        # self.map[0, self.agent_current_pos[0]-10:self.agent_current_pos[0]+10, self.agent_current_pos[1]-10:self.agent_current_pos[1]+10] = 1.0
        #make it hollow with 1px thick walls
        # self.map[0, self.agent_current_pos[0]-9:self.agent_current_pos[0]+9, self.agent_current_pos[1]-9:self.agent_current_pos[1]+9] = 0.0
        # self.map[0, self.agent_current_pos[0]-18:self.agent_current_pos[0]+18, self.agent_current_pos[1]-18:self.agent_current_pos[1]+18] = 0.0


        self.map[0,128:128+256,128:128+256] = self.map_layout

        self.fog_of_war_mask = np.zeros( (256,256) )

        # self.populate_ep_map(self.ep_list[self.current_ep_id]['objectgoal']+1, self.map_small_layout)
        self.populate_ep_map(self.map_small_layout)

        self.map[-4] = self.pool( self.map[:-4].unsqueeze(0) ).squeeze(0)

        # vflip
        # if(self.env_id % 2 == 0):
        #     self.map = torch.flip(self.map,dims=[-2])

        min_x=self.agent_current_pos[0]-128
        max_x=self.agent_current_pos[0]+128
        min_y=self.agent_current_pos[1]-128
        max_y=self.agent_current_pos[1]+128
        self.agent_internal_map = self.map[:,min_x:max_x,min_y:max_y].detach().clone()
        self.agent_current_pos = np.array([128,128])

        # cv2.imshow("tmp",(self.map[-4]*255).byte().cpu().numpy())
        # cv2.waitKey()
        # # ########################################################################
        # # '''
        # # FOR # DEBUG:
        # # '''
        # # cv2.imshow("map-3", ((map[-4]+map[-3]+map[-1])*255).byte().cpu().numpy() )
        # trash = ((self.map[0]*255)+(self.map[self.ep_list[self.current_ep_id]['objectgoal']+1]*128)).byte().cpu().numpy()
        # # trash = trash+(map[-1].byte().cpu().numpy()*128)
        #
        # cv2.imshow("map", trash )
        # # cv2.imshow("map-2", ((map[-4]+map[-2])*255).byte().cpu().numpy() )
        #
        # # arrow = cv2.imread("extras/arrow.png")
        # # arrow = rotate_image(torch.from_numpy(arrow), np.rad2deg(compass.item()) )
        # # cv2.imshow("arrow",arrow.numpy())
        #
        # cv2.waitKey()
        # exit()
        # # ########################################################################

        # '''
        # aux 0 , is the occupied cells for the a* planner
        # '''
        # self.map_aux[0] = self.pool(self.long_term_map.unsqueeze(0)).squeeze(0)
        #
        # '''
        # aux 1 , is past locations
        # aux 2 , is the current location
        # '''
        # self.map_aux[2].fill_(0)
        # self.map_aux[1:3, data_idxs[0]-1:data_idxs[0]+2 , data_idxs[1]-1:data_idxs[1]+2  ] = 1.0
        #
        # '''
        # aux 3, is the new prediction
        # '''
################################################################################

'''
Main to run evaluation
'''
def main():
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

    config = get_config(
        ["configs/beyond.yaml","configs/ddppo_pointnav.yaml"], ["BASE_TASK_CONFIG_PATH", config_paths]
    ).clone()

    config.defrost()
    config.TORCH_GPU_ID = 0
    config.INPUT_TYPE = args.input_type
    config.MODEL_PATH = args.model_path
    config.RANDOM_SEED = 7
    config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    config.freeze()

    device = config.BEYOND.DEVICE
    # device = "cpu"

    # agent = BeyondAgent(device=config.BEYOND.DEVICE, config=config, batch_size=1, is_batch_internal=True)

    # agent = BeyondAgentWithoutInternalMapper(device=config.BEYOND.DEVICE, config=config, batch_size=1, is_batch_internal=True)
    agent = BeyondAgentWithoutInternalMapper(device=device, config=config, batch_size=1, is_batch_internal=True)

    # env = SemanticEnv(device=device, n_ep=256)
    env = SemanticEnv(device=device, n_ep=21)
    env.seed(config.RANDOM_SEED)
    env.init_eps()

    infos = []
    reward_mean_eps = []
    for ep in range(len(env.ep_list)):
        observations = env.reset()
        agent.reset()
        reward_mean = []
        while not env.episode_over:
            # action = agent.act(observation)
            action,computed_reward = agent.act(observations)
            env.send_computed_reward(computed_reward[0])
            # print("action",action,"\n")
            # observation = env.step(action,)
            observations, reward, done, info = env.step(action)
            # reward_mean.append(reward.cpu().numpy())
            reward_mean.append(reward)
        # print("done ep",ep)
        reward_mean = np.mean(np.array(reward_mean))
        reward_mean_eps.append(reward_mean)
        print("reward_mean",reward_mean)
        print(info)

        if(not infos):
            infos = [info]
        else:
            infos.append(info)
        # exit()

    # batch = {'distance_to_goal':[],'success':[],'softspl':[],'spl':[]}
    batch = {'distance_to_goal':[],'success':[],'softspl':[],'spl':[], "collisions":[], "steps":[]}
    for i in infos:
        for key in batch:
            batch[key].append(i[key])
            # if(key=='collisions'):
            #     batch[key].append(i[key]['count'])
            # else:
            #     batch[key].append(i[key])

    for key in batch:
        batch[key]=np.mean(np.array(batch[key]))
    ###########################################################
    # means = _convert_list_to_means(infos)
    means = batch
    print("\n\n",means)
    print("reward_mean_eps",np.mean(np.array(reward_mean_eps)))

if __name__ == "__main__":
    os.environ["CHALLENGE_CONFIG_FILE"] = "configs/challenge_objectnav2021.local.rgbd.yaml"
    main()
