'''
    load libraries
'''
##########################################################################

#load pytorch stuff
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torch.autograd import Variable

# import os
import numpy as np

#logging
# from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
# from datetime import datetime
# import time

#params
# import argparse

#habitat stuff
# import habitat
# from habitat.config.default import get_config
# from habitat import get_config as get_task_config
# from habitat.core.simulator import AgentState
#
# from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal, StopAction
# from habitat.utils.test_utils import sample_non_stop_action

from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
# from habitat.datasets.utils import get_action_shortest_path
# from habitat.core.simulator import ShortestPathPoint

from habitat.core.env import RLEnv

# from typing import Any, Dict, List, Optional
# from collections import defaultdict, deque
# import numbers
# from gym.spaces.box import Box
# import copy

#mine stuff
# import mapper
# import model
# from beyond_agent_matterport_only import BeyondAgentMp3dOnly
####
# from numba import njit

# class GreedyFollowerEnv(habitat.RLEnv):
class GreedyFollowerEnv(RLEnv):
    def __init__(self, config, dataset=None, env_ind=0, is_train=True):
        super(GreedyFollowerEnv, self).__init__(config, dataset)

        '''
            Wrapper logic
        '''
        # self._follower_goal_radius = self._env._config.TASK.SUCCESS.SUCCESS_DISTANCE*0.25

        # self.is_train = False
        self.is_train = is_train

        #best result for eval so far
        self._follower_goal_radius = self._env._config.TASK.SUCCESS.SUCCESS_DISTANCE*0.5


        # self._follower_goal_radius = self._env._config.TASK.SUCCESS.SUCCESS_DISTANCE
        self._follower_target_viewpoint_pos = None
        self._follower_final_pos = None

        self._follower = ShortestPathFollower(self._env._sim, goal_radius=self._follower_goal_radius, return_one_hot=False)

        self._updated_dict= {
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

        # self.mapper_wrapper = mapper.MapperWrapper(map_size_meters=self._env._config.TRAINER.GLOBAL_POLICY.MAP_SIZE_METERS, map_cell_size=self._env._config.TRAINER.GLOBAL_POLICY.MAP_CELL_SIZE,config=self._env._config,n_classes=self._env._config.TRAINER.GLOBAL_POLICY.N_CLASSES)

        # dataset_json = process_file(self._env._config.DATASET.DATA_PATH.replace('{split}',self._env._config.DATASET.SPLIT))

        '''
        WARNING UNCOMMENT IT WHEN USING matterport3D
        '''
        # self.dict_map = self.create_mapping_dict()

        self.object_goal = None
        self.agent_starting_state = None
        self.shortest_path = None
        self._follower_idx = 0
        self.smallest_dist = -1

        self._follower_step_size = self._env._config.SIMULATOR.FORWARD_STEP_SIZE
        self._follower_max_visible_dis = self._env._config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH

    @property
    def episode_over(self):
        return self._env.episode_over

    def create_mapping_dict(self):
        # dict_map = np.zeros(1024)
        # dict_map = np.zeros(16384)
        # dict_map = torch.zeros(16384,device=self._env._config.BEYOND_TRAINER.DEVICE)
        dict_map = torch.zeros(16384,device="cuda")

        for cat in self._env._dataset.goals_by_category:
            for obj in self._env._dataset.goals_by_category[cat]:
                dict_map[obj.object_id]=self._updated_dict[obj.object_category]

        return dict_map

    def get_sseg_dict(self):
        return self.dict_map

    # def get_object_goal(self):
    #     return self.object_goal

    def get_distance_to_goal(self):
        source_position = self._env._sim.get_agent_state().position
        target_position = self._follower_target_viewpoint_pos
        # dist = self._env._sim.geodesic_distance(source_position,target_position)

        tmp_diff = target_position - source_position
        dist = np.linalg.norm(tmp_diff, ord=2, axis=-1)

        return dist

    # def update_map(self,depth,sseg,pose):
    #     self.mapper_wrapper.update_map(depth,sseg,pose)

    def randomize_env(self):
        self._env._episode_iterator._shuffle_iterator()

    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        # self._follower_idx+=1
        return 0

    def get_done(self, observations):
        return self._env.episode_over

    def get_metrics(self):
        return self._env.get_metrics()

    def get_metrics_extended(self):
        infos= self._env.get_metrics()

        infos['step']= self._follower_idx
        infos['object_goal']= self._env._current_episode.object_category
        infos['episode_id']= self._env._current_episode.episode_id
        infos['agent_starting_state']= {'position': self.agent_starting_state.position, 'rotation': self.agent_starting_state.rotation}
        infos['geodesic_distance']= self.smallest_dist

        return infos

    def get_info(self, observations):
        # return self._env.get_metrics()
        infos= self._env.get_metrics()

        # infos['step']= self._follower_idx
        # infos['object_goal']= self._env._current_episode.object_category
        # infos['episode_id']= self._env._current_episode.episode_id
        # infos['agent_starting_state']= {'position': self.agent_starting_state.position, 'rotation': self.agent_starting_state.rotation}
        # infos['geodesic_distance']= self.smallest_dist
        #
        # # infos['object_goal']=self.object_goal
        # # infos['object_goal']=self.object_goal
        #
        # # if(self._env.episode_over):
        # #     print("infos",infos)
        # #     # print("#END-EP####################################")
        #
        # # infos['episode_id']=self._env._current_episode.episode_id
        # # infos['mark']=self._env._current_episode.info['mark']


        return infos

    def get_goal_gt(self):
        current_state = self._env._sim.get_agent_state()
        points = []

        for i in range(len(self._env._current_episode.goals)):
            for j in range(len(self._env._current_episode.goals[i].view_points)):
                points.append(self._env._current_episode.goals[i].view_points[j].agent_state.position)

        points = np.array(points)
        tmp_diff = points - current_state.position
        euclid_vec = np.linalg.norm(tmp_diff, ord=2, axis=-1)

        target_position = points[np.argsort(euclid_vec)]

        smallest_dist = self._env._sim.geodesic_distance(current_state.position,target_position)

        '''
        start_position = self._env._current_episode._shortest_path_cache.points[0]
        final_position = self._env._current_episode._shortest_path_cache.points[-1]
        '''

        shortest_path = np.copy(self._env._current_episode._shortest_path_cache.points)
        shortest_path = np.array(shortest_path )
        shortest_path = shortest_path[1:]

        #small steps
        # gt_pos = shortest_path[0]
        #only final point
        gt_pos = shortest_path[-1]

        return gt_pos

    def get_goal(self):
        return self._follower_target_viewpoint_pos

    def get_starting_state(self):
        return self.agent_starting_state

    def get_next_action_eval(self):
        return self._follower.get_next_action(self._follower_target_viewpoint_pos)

    def get_next_action(self):
        # return self._follower.get_next_action(self._follower_target_viewpoint_pos)
        action = self._follower.get_next_action(self._follower_target_viewpoint_pos)

        if(self.is_train):
            if action == 0:
                # final_goal_reached
                if (self._follower_idx == len(self.shortest_path)-1):
                    return action
                else:
                    self._follower_idx+=1
                    self._follower_target_viewpoint_pos = self.shortest_path[self._follower_idx]
                    return self.get_next_action()

        return action

    def reset(self):
        # print("#NEW-EP#######################################")
        tmp = self._env.reset()
        self._follower_idx = 0

        # self.object_goal = self._env._current_episode.info['closest_goal_object_id']
        # print("EXITING")
        # exit()

        # self.object_goal = self._updated_dict[self._env._current_episode.object_category]
        # print("self._env._current_episode.object_category",self._env._current_episode.object_category,"self.object_goal",self.object_goal)
        # print("EXITING")
        # exit()
        self.agent_starting_state = self._env._sim.get_agent_state()

        # if(self.is_train):
        self.update_goal()


        # print("self.agent_starting_state.rotation",self.agent_starting_state.rotation)
        # self.agent_starting_state.rotation = self._env._current_episode.start_rotation
        #
        # print("self.agent_starting_state.rotation episode start",mapper.quaternion_from_coeff(self._env._current_episode.start_rotation))


        return tmp

    def step_obs_only(self, *args, **kwargs):
        r"""Perform an action in the environment.
        :return: :py:`(observations)`
        """
        self._follower_idx+=1
        observations = self._env.step(*args, **kwargs)
        return observations

    ######################################################################
    def is_navigable(self, point):
        value = self._env._sim._sim.pathfinder.is_navigable(point)

        if(value):
            return value
        else:
            closest_pt = self._env._sim._sim.pathfinder.snap_point(point)
            value = self._env._sim._sim.pathfinder.is_navigable(closest_pt)
            return value


    def update_goal_param(self,goal):
        if(self._env._sim._sim.pathfinder.is_navigable(goal)):
            self._follower_target_viewpoint_pos = goal
        else:
            closest_pt = self._env._sim._sim.pathfinder.snap_point(goal)

            if(self._env._sim._sim.pathfinder.is_navigable(closest_pt)):
                self._follower_target_viewpoint_pos = closest_pt


    def update_goal(self):
        # self._follower_target_viewpoint_pos = self._env._current_episode.info["target_viewpoint_pos"]

        points = []

        for i in range(len(self._env._current_episode.goals)):
            for j in range(len(self._env._current_episode.goals[i].view_points)):
                points.append(self._env._current_episode.goals[i].view_points[j].agent_state.position)

        points = np.array(points)
        tmp_diff = points - self.agent_starting_state.position
        euclid_vec = np.linalg.norm(tmp_diff, ord=2, axis=-1)

        target_position = points[np.argsort(euclid_vec)]

        smallest_dist = self._env._sim.geodesic_distance(self.agent_starting_state.position,target_position)
        self.smallest_dist = smallest_dist
