'''
    load libraries
'''
##########################################################################

#load pytorch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import os
import numpy as np

#logging
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time

#params
import argparse

#habitat stuff
import habitat
from habitat.config.default import get_config
from habitat import get_config as get_task_config
from habitat.core.simulator import AgentState

from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal, StopAction
from habitat.utils.test_utils import sample_non_stop_action

from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.datasets.utils import get_action_shortest_path
from habitat.core.simulator import ShortestPathPoint

from typing import Any, Dict, List, Optional
from collections import defaultdict, deque
import numbers
from gym.spaces.box import Box
import copy

#mine stuff
import mapper
import model
from beyond_agent_matterport_only import BeyondAgentMp3dOnly
####
from numba import njit

import gzip
import json
def process_file(jsonfilename):
    with gzip.GzipFile(jsonfilename, 'r') as fin:    # 4. gzip
        json_bytes = fin.read()                      # 3. bytes (i.e. UTF-8)

    json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
    data = json.loads(json_str)                      # 1. data
    return data

##########################################################################

##########################################################################
class MyTrainer():
    def __init__(self, config):

        self.config = config
        ##################################################################

        self.device = config.BEYOND_TRAINER.DEVICE
        print("DEVICE=",self.device)

        self.SAVE_EVERY = config.BEYOND_TRAINER.SAVE_EVERY

        self.LOSS = config.BEYOND_TRAINER.LOSS
        self.OPTIMIZER = config.BEYOND_TRAINER.OPTIMIZER
        print("OPTIMIZER",self.OPTIMIZER)

        self.LR = config.BEYOND_TRAINER.LR
        print("INITIAL LR",self.LR)

        self.N_ITERATIONS = config.BEYOND_TRAINER.N_ITERATIONS

        self.BATCH_SIZE = config.BEYOND_TRAINER.BATCH_SIZE
        self.EVAL_BATCH_SIZE = config.BEYOND_TRAINER.EVAL_BATCH_SIZE

        self.PRINT_EVERY = int(np.ceil(config.BEYOND_TRAINER.PRINT_EVERY))

        '''
            Logging stuff
        '''
        now = datetime.now()


        EXPERIMENT_NAME = config.BEYOND_TRAINER.EXPERIMENT_NAME
        if EXPERIMENT_NAME!="":
            LOG_DIR = config.BEYOND_TRAINER.LOG_DIR+EXPERIMENT_NAME+now.strftime("%d-%m-%Y_%H-%M-%S")+"/"
        else:
            LOG_DIR = config.BEYOND_TRAINER.LOG_DIR+now.strftime("%d-%m-%Y_%H-%M-%S")+"/"
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)


        self.CHECKPOINT = config.BEYOND_TRAINER.CHECKPOINT_DIR
        if EXPERIMENT_NAME!="":
            self.CHECKPOINT = self.CHECKPOINT+EXPERIMENT_NAME+now.strftime("%d-%m-%Y_%H-%M-%S")+"/"
        else:
            self.CHECKPOINT = self.CHECKPOINT+now.strftime("%d-%m-%Y_%H-%M-%S")+"/"
        if not os.path.exists(self.CHECKPOINT):
            os.makedirs(self.CHECKPOINT)


        self.writer = SummaryWriter(log_dir=LOG_DIR)
        '''
            MAKE EVERYTHING EASER FOR YOURSELF AND
            PARAMETRIZE LOGGING!
        '''
        hparams_dict = {'loss': self.LOSS, 'optimizer': self.OPTIMIZER, 'batch_size': self.BATCH_SIZE}
        hparams_metrics = {}

        self.writer.add_hparams(hparams_dict, hparams_metrics)

        if(self.LOSS == "MaskedLogCoshLoss"):
            self.loss_function = MaskedLogCoshLoss()
        elif(self.LOSS == "MaskedMSELoss"):
            self.loss_function = MaskedMSELoss()
        elif(self.LOSS == "LogCoshLoss"):
            self.loss_function = LogCoshLoss()
        elif(self.LOSS == "CrossEntropyLoss"):
            weight=torch.tensor([0.84317787, 0.01911674, 0.06394188, 0.07376351],device=self.device)
            self.loss_function = nn.CrossEntropyLoss(weight=weight)
        elif(self.LOSS == "MSELoss"):
            self.loss_function = nn.MSELoss()

        # elif(self.LOSS == "CustomMSELossPlusCrossEntropyLoss"):
        #     self.loss_function = CustomMSELossPlusCrossEntropyLoss()

        else:
            print("Loss Function not set ABORT!")
            exit(1)

        '''
            OPTIMIZER depends on the model so init them later!
        '''

        valid_loss_min = np.Inf

        self.envs = None
        self.configs = None
        self.datasets = None

        self.optimizer = None
        self.scheduler = None
        self.scheduler2 = None

        self.agent = None
        self.num_envs = None

        self.iteration_idx = 0
        self.gt_goal_in_map_coo_perc = None

        self.train_mode = config.BEYOND_TRAINER.TRAIN_MODE
    ############################################################################
    def _load_enviroment_data(self, gpu2gpu, config_main, batch_size, scene_idx, shuffle=True, dataset_mode="TRAIN"):
        configs = []
        datasets = []
        total_episodes = 0
        episode_per_env = 0
        dataset_idx = 0


        config = config_main
        # print("scene",config.DATASET.SPLIT)

        dataset_idx = 0
        for j in range(0,batch_size):

            config.defrost()

            mode = getattr(config.BEYOND_TRAINER.ENVIRONMENT_LIST,dataset_mode)

            config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True if dataset_mode == "TRAIN" else False


            config.DATASET.TYPE = config.BEYOND_TRAINER.ENVIRONMENT_LIST.TRAIN.TYPE
            config.DATASET.SPLIT = mode.NAMES[scene_idx+j]
            config.DATASET.DATA_PATH = mode.PATH+"{split}.json.gz"
            if "habitat-challenge-data" not in config.DATASET.SCENES_DIR:
                config.DATASET.SCENES_DIR = config.DATASET.DATA_PATH.split("/")[0]+"/"+config.DATASET.SCENES_DIR
            # config.TASK.MEASUREMENTS.append("COLLISIONS")
            # config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = gpu2gpu
            # config.DATASET.TYPE = "ObjectNav-v1"
            print("config.DATASET.TYPE",config.DATASET.TYPE )
            config.freeze()

            print(config.DATASET.SPLIT)
            # print("config",config)

            datasets.append(
                habitat.make_dataset(
                    id_dataset=config.DATASET.TYPE, config=config.DATASET
                )
            )

            '''
                shuffle the episodes
            '''
            if(shuffle):
                datasets[dataset_idx].episodes = np.random.permutation(datasets[dataset_idx].episodes)

            print("dataset_idx",dataset_idx)
            print("after len(datasets[dataset_idx].episodes)",len(datasets[dataset_idx].episodes))




            configs.append(config)
            dataset_idx+=1

        return configs, datasets
    ############################################################################
    def init_optimizer(self, model):
        '''
            Put this after loading model to GPU
        '''

        if(self.OPTIMIZER == "AdamW"):
            self.optimizer = optim.AdamW(model.parameters(), lr=self.LR, weight_decay=self.DECAY_RATE)
        elif(self.OPTIMIZER == "Adam"):
            # self.optimizer = optim.Adam(model.parameters(), lr=self.LR, weight_decay=self.DECAY_RATE)
            self.optimizer = optim.Adam(model.parameters(), lr=self.LR)
        elif(self.OPTIMIZER == "SGD"):
            print("USING SGD")
            self.momentum = float(self.config.BEYOND_TRAINER.MOMENTUM)
            self.optimizer = optim.SGD(model.parameters(), lr=self.LR, momentum=self.momentum)
        elif(self.OPTIMIZER == "SGDOL"):
            from sgdol import SGDOL

            smoothness= 10.0
            alpha= 10.0
            sum_inner_prods = alpha
            sum_grad_normsq = alpha
            lr = sum_inner_prods / (smoothness * sum_grad_normsq)
            lr = max(min(lr, 2.0/smoothness), 0.0)

            # smoothness= 10.0
            # alpha= 10.0
            # sum_inner_prods= 118.4196164513043
            # sum_grad_normsq= 127.23471481889624
            # lr= 0.09307178203673486

            self.optimizer = SGDOL(model.parameters(), smoothness=smoothness, alpha=smoothness, sum_inner_prods=sum_inner_prods,sum_grad_normsq=sum_grad_normsq,lr=lr)
        else:
            print("OPTIMIZER not set ABORT!")
            exit(1)

        if(self.config.BEYOND_TRAINER.SCHEDULER == "CosineAnnealing"):
            print("USING CosineAnnealing")
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.N_ITERATIONS)

        if(self.config.BEYOND_TRAINER.SCHEDULER2 == "ReduceLROnPlateau"):
            print("USING ReduceLROnPlateau")
            self.scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', factor=0.1, patience=1, cooldown=0, min_lr=self.LR*(10**-3))
    ############################################################################
    def get_sseg_dict(self):
        dict_map = self.envs.call(function_names=["get_sseg_dict" for _ in range(self.num_envs)])
        return dict_map
    ############################################################################
    def get_goals_gt(self):
        goals = self.envs.call(function_names=["get_goal_gt" for _ in range(self.num_envs)])
        return goals
    ############################################################################
    def get_starting_state(self):
        states = self.envs.call(function_names=["get_starting_state" for _ in range(self.num_envs)])
        return states
    ############################################################################
    def train_step(self, global_input, orientation, object_goal):

        self.agent.g_policy.train()
        losses=[]
        y = self.gt_goal_in_map_coo_perc

        '''
            this can be called directly on the model no need to
            call act wrapper since there is no interaction with
            the simulation in this step
        '''

        '''
            ask the entire rollout as a single batch x sequence x features
        '''
        self.agent.g_policy.zero_grad()
        y_pred = self.agent.g_policy(global_input, orientation, object_goal)

        '''
        loss should be RMSELoss for the y_pred + BinaryCrossEntropy for nav_pred
        '''

        # loss = self.loss_function(y_pred, y, nav_pred, nav_gt)
        loss = self.loss_function(y_pred, y)

        loss.backward()
        # clip_gradient(self.agent.g_policy,self.CLIP_VALUE)
        self.optimizer.step()

        losses.append(loss.item())
        # print("losses",losses)

        return np.mean(np.array(losses))
    ############################################################################
    def train(self):
        '''
            Instanciate agent and load models to gpu
        '''
        self.agent=BeyondAgentMp3dOnly(device=self.device, config=self.config, batch_size=self.BATCH_SIZE)
        self.agent.set_goal_eval()
        # self.agent.set_goal_train()

        self.init_optimizer(self.agent.g_policy)
        '''
            Loop for scenes, BEGIN
        '''
        n_scenes = len(self.config.BEYOND_TRAINER.ENVIRONMENT_LIST.TRAIN.NAMES)
        self.n_scenes = n_scenes
        losses = []
        ####################################################################

        '''
            initial environment configurations
        '''
        ####################################################################
        '''
            Load first scene and divide its episodes in batches
        '''
        # scene_idx = 0
        self.iteration_idx=0
        for scene_idx in range(0,n_scenes,self.BATCH_SIZE):
            self.configs, self.datasets = self._load_enviroment_data(True, self.config, self.BATCH_SIZE, scene_idx=scene_idx, shuffle=True, dataset_mode="TRAIN")

            num_envs = len(self.configs)
            self.num_envs = num_envs

            env_fn_args = tuple(zip(self.configs, self.datasets, range(num_envs), [self.train_mode for tmp in range(num_envs)] ))
            ####################################################################

            multiprocessing_start_method="forkserver"

            self.envs = habitat.VectorEnv(
                make_env_fn=make_follower_env,
                env_fn_args=env_fn_args,
                multiprocessing_start_method=multiprocessing_start_method,
            )
            ####################################################################

            observations = self.envs.reset()
            batch = batch_obs(observations, device=self.device)
            batch['sseg_dict'] = self.get_sseg_dict()
            # self.agent.planner.reset_batch()

            for tmp_env_idx in range(self.BATCH_SIZE):
                self.agent.reset(env_idx=tmp_env_idx)

            losses = []
            pth_time = time.time()
            print("self.N_ITERATIONS",self.N_ITERATIONS)
            for iteration in range(self.N_ITERATIONS):
            # for iteration in range(1):
                object_goal = batch['objectgoal'].long().unsqueeze(-1)
                batch_new = self._collect_rollout_step(batch)
                loss = self.train_step(self.agent.global_input, self.agent.orientation, object_goal)
                losses.append(loss)
                batch = batch_new

                # ################################################################
                '''
                LOGGING
                '''
                pth_time = time.time()-pth_time
                if ((((scene_idx*self.N_ITERATIONS)+iteration) % self.PRINT_EVERY) == 0):

                    mean_loss = np.mean(np.array(losses))
                    print("iteration",int(self.iteration_idx),"mean loss:",np.mean(np.array(losses)),"total time:",pth_time, )

                    self.writer.add_scalars(
                        "losses",
                        {k: l for l, k in zip([mean_loss], ["GLOBAL_POLICY_training_loss",]) },
                        int(self.iteration_idx),
                    )
                    losses=[]
                # ################################################################
                '''
                SAVE MODEL AT REGULAR INTERVAL
                '''
                if(self.iteration_idx % self.SAVE_EVERY == 0):
                    filename_model = self.CHECKPOINT+"GLOBAL_POLICY_"+'{}_rollout_iterations.pt'.format(int(self.iteration_idx))

                    print("saving model",filename_model)
                    torch.save(self.agent.g_policy.state_dict(
                    ), filename_model)
                ################################################################
                self.iteration_idx+=1
            #end for

            ################################################################
            '''
            SCHEDULER
            '''
            if(self.scheduler):
                local_lr = self.scheduler.get_last_lr()
                self.scheduler.step()

                if(not self.scheduler2):
                    print("lr scheduler:",local_lr)
                    self.writer.add_scalars(
                        "lr",
                        {k: l for l, k in zip(local_lr, ["GLOBAL_POLICY_training_loss",]) },
                        int(self.iteration_idx),
                    )
            ####################################################################
            '''
            SAVE AT END OF EPOCH
            '''
            filename_model = self.CHECKPOINT+"GLOBAL_POLICY_"+'{}_rollout_iterations.pt'.format(int(self.iteration_idx))
            print("saving model",filename_model)
            torch.save(self.agent.g_policy.state_dict(
            ), filename_model)
            ####################################################################
            print("ending scene", scene_idx)
            self.envs.close()
            torch.cuda.empty_cache()
            ####################################################################
            '''
            FINE tune hyperparameters with eval
            '''
            ####################################################################
            '''
            Perform eval
            '''
            # self.agent.reset_map(self.EVAL_BATCH_SIZE)
            # eval_metric = self.eval()
            #
            # ####################################################################
            #
            # if(self.scheduler2):
            #     # print("lr scheduler2:",self.scheduler2.get_last_lr())
            #     self.scheduler2.step(eval_metric)
            #
            # self.agent.reset_map(self.BATCH_SIZE)
            #
            # if(self.scheduler):
            #     local_lr = self.scheduler.get_last_lr()
            #     print("lr scheduler:",local_lr)
            #
            #     self.writer.add_scalars(
            #         "lr",
            #         {k: l for l, k in zip(local_lr, ["GLOBAL_POLICY_training_loss",]) },
            #         int(self.iteration_idx),
            #     )
            ####################################################################
        #end for
        print("EXITING DEBUG")
        exit()
    ############################################################################
    def _collect_rollout_step(self, observations):
        #TODO
        actions = self.agent.act(observations)

        ############################################################################
        '''
        Pass the action to the env and grab outputs
        '''
        POSSIBLE_ACTIONS = self.config.TASK.POSSIBLE_ACTIONS

        for tmp_env_idx in range(self.BATCH_SIZE):
            # if(best_actions[tmp_env_idx]==0):
            #     tmp_action = {"action":{"action": POSSIBLE_ACTIONS[2] }}
            # else:
            #     tmp_action = {"action":{"action": POSSIBLE_ACTIONS[best_actions[tmp_env_idx]] }}

            # tmp_action = {"action":{"action": POSSIBLE_ACTIONS[best_actions[tmp_env_idx]] }}
            tmp_action = {"action":{"action": POSSIBLE_ACTIONS[actions[tmp_env_idx]] }}
            outputs_tmp = self.envs.step_at(tmp_env_idx, tmp_action)
            self.agent.ep_iteration[tmp_env_idx]+=1

            if(tmp_env_idx==0):
                outputs = outputs_tmp
            else:
                outputs = outputs + outputs_tmp

        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        batch = batch_obs(observations, device=self.device)
        batch['sseg_dict'] = self.get_sseg_dict()
        ############################################################################
        '''
        Check if episode finished
        '''
        for i,done in enumerate(dones):
            if(done):
                self.agent.reset(env_idx=i)


        ############################################################################
        '''
        GT GOAL
        '''
        _states = self.get_starting_state()
        _goals = self.get_goals_gt()
        gt_goal_in_world3d_coo = mapper.batch_original_3d_to_episodic_3d(_states, _goals)
        ####################################################################
        gt_goal_in_map_coo = mapper.worldpointgoal_to_map(gt_goal_in_world3d_coo, self.agent.mapper_wrapper.mapper.map_size_meters, self.agent.mapper_wrapper.mapper.map_cell_size)

        gt_goal_in_map_coo_perc = gt_goal_in_map_coo.float()/mapper.get_map_size_in_cells(self.agent.mapper_wrapper.mapper.map_size_meters, self.agent.mapper_wrapper.mapper.map_cell_size)
        self.gt_goal_in_map_coo_perc = gt_goal_in_map_coo_perc.to(self.device)
        ####################################################################

        return batch



'''
    Here should be the GreedyFollower Logic and helper functions
'''
##########################################################################
class GreedyFollowerEnv(habitat.RLEnv):
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
        self.dict_map = self.create_mapping_dict()

        self.object_goal = None
        self.agent_starting_state = None
        self.shortest_path = None
        self._follower_idx = 0
        self.smallest_dist = -1

        self._follower_step_size = self._env._config.SIMULATOR.FORWARD_STEP_SIZE
        self._follower_max_visible_dis = self._env._config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH



    def create_mapping_dict(self):
        # dict_map = np.zeros(1024)
        # dict_map = np.zeros(16384)
        dict_map = torch.zeros(16384,device=self._env._config.BEYOND_TRAINER.DEVICE)

        for cat in self._env._dataset.goals_by_category:
            for obj in self._env._dataset.goals_by_category[cat]:
                dict_map[obj.object_id]=self._updated_dict[obj.object_category]

        return dict_map

    def get_sseg_dict(self):
        return self.dict_map

    def get_object_goal(self):
        return self.object_goal

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

    def get_info(self, observations):
        # return self._env.get_metrics()
        infos= self._env.get_metrics()
        infos['object_goal']= self._env._current_episode.object_category
        infos['episode_id']= self._env._current_episode.episode_id
        infos['agent_starting_state']= {'position': self.agent_starting_state.position, 'rotation': self.agent_starting_state.rotation}
        infos['geodesic_distance']= self.smallest_dist
        # infos['object_goal']=self.object_goal
        # infos['object_goal']=self.object_goal

        # if(self._env.episode_over):
        #     print("infos",infos)
        #     # print("#END-EP####################################")

        # infos['episode_id']=self._env._current_episode.episode_id
        # infos['mark']=self._env._current_episode.info['mark']


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

        self.object_goal = self._updated_dict[self._env._current_episode.object_category]
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

def make_follower_env(config, dataset, rank: int = 0, is_train=True):
    r"""Constructor for default habitat Env.
    :param config: configurations for environment
    :param dataset: dataset for environment
    :param rank: rank for setting seeds for environment
    :return: constructed habitat Env
    """
    env = GreedyFollowerEnv(config=config, dataset=dataset, is_train=is_train)
    env.seed(config.SEED + rank)
    return env

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

##########################################################################


'''
    End of file and call for main function
'''
##########################################################################
def main():
    '''
        Parser args and config files and call trainer!
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str,
                        required=True, default="config.yaml")
    parser.add_argument(
        "--input-type", default="blind", choices=["blind", "rgb", "depth", "rgbd"]
    )
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )
    parser.add_argument("--model-path", default="", type=str)

    args = parser.parse_args()

    # config = habitat.get_config(args.config_path)

    config_paths = args.config_path
    # config_paths = os.environ["CHALLENGE_CONFIG_FILE"]

    config = get_config(
        ["configs/beyond_trainer.yaml", "configs/beyond.yaml","configs/ddppo_pointnav.yaml",config_paths], ["BASE_TASK_CONFIG_PATH", config_paths]
    ).clone()

    config.defrost()
    config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)

    config.TORCH_GPU_ID = 0
    config.INPUT_TYPE = args.input_type
    config.MODEL_PATH = args.model_path

    # config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = True

    config.RANDOM_SEED = 7
    config.freeze()

    print("-------------------------------------------------------------")

    # print(config)

    trainer = MyTrainer(config)
    # print("EXITING DEBUG")
    # exit()
    if(trainer.train_mode):
        trainer.train()
    else:
        trainer.eval()

    #everything went fine so exit with 0
    return 0

if __name__ == "__main__":
    main()
##########################################################################
