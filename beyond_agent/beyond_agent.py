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
from habitat_baselines.common.utils import batch_obs
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ppo import PointNavBaselinePolicy, Policy

from habitat_baselines.rl.ddppo.policy.resnet_policy import (  # isort:skip noqa
    PointNavResNetPolicy,
)

#beyond stuff
import mapper
import model

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



##########################################################################
'''
Placeholder for the Agent
'''
class BeyondAgent(Agent):
    def __init__(self, device, config, batch_size):

        self.device=device
        self.config=config
        self.batch_size=batch_size
        self.input_shape = config.BEYOND.GLOBAL_POLICY.INPUT_SHAPE

        self.orientation=torch.zeros(self.batch_size,1, device=self.device)
        self.array_of_preds=[[] for _ in range(self.batch_size)]
        self.ep_iteration=torch.zeros(self.batch_size)
        # self.ep_iteration=0

        # print(self.config)

        if(self.config.BEYOND.GLOBAL_POLICY.USE_MATTERPORT_TO_GIBSON):
            self.m3pd_to_gibson = {"__background__": -1,
                       "person": -1,
                       "bicycle": -1,
                       "car": -1,
                       "motorcycle": -1,
                       "airplane": -1,
                       "bus": -1,
                       "train": -1,
                       "truck": -1,
                       "boat": -1,
                       "traffic light": -1,
                       "fire hydrant": -1,
                       "stop sign": -1,
                       "parking meter": -1,
                       "bench": -1,
                       "bird": -1,
                       "cat": -1,
                       "dog": -1,
                       "horse": -1,
                       "sheep": -1,
                       "cow": -1,
                       "elephant": -1,
                       "bear": -1,
                       "zebra": -1,
                       "giraffe": -1,
                       "backpack": -1,
                       "umbrella": -1,
                       "handbag": -1,
                       "tie": -1,
                       "suitcase": -1,
                       "frisbee": -1,
                       "skis": -1,
                       "snowboard": -1,
                       "sports ball": -1,
                       "kite": -1,
                       "baseball bat": -1,
                       "baseball glove": -1,
                       "skateboard": -1,
                       "surfboard": -1,
                       "tennis racket": -1,
                       "bottle": -1,
                       "wine glass": -1,
                       "cup": -1,
                       "fork": -1,
                       "knife": -1,
                       "spoon": -1,
                       "bowl": -1,
                       "banana": -1,
                       "apple": -1,
                       "sandwich": -1,
                       "orange": -1,
                       "broccoli": -1,
                       "carrot": -1,
                       "hot dog": -1,
                       "pizza": -1,
                       "donut": -1,
                       "cake": -1,
                       "chair": 0,
                       "couch": 5,
                       "potted plant": 8,
                       "bed": 6,
                       "dining table": 1,
                       "toilet": 10,
                       "tv": 13,
                       "laptop": -1,
                       "mouse": -1,
                       "remote": -1,
                       "keyboard": -1,
                       "cell phone": -1,
                       "microwave": -1,
                       "oven": -1,
                       "toaster": -1,
                       "sink": -1,
                       "refrigerator": 3,
                       "book": -1,
                       "clock": -1,
                       "vase": -1,
                       "scissors": -1,
                       "teddy bear": -1,
                       "hair drier": -1,
                       "toothbrush": -1
                       }
            self.m3pd_to_gibson_inverted = {v: k for k, v in self.m3pd_to_gibson.items()}

        #internal dict for the COCO classes to map the 30 classes in the semantic map, similar classes in matterport3D map to the same number
        self._internal_dict = {'__background__': 0, 'person': 0, 'bicycle': 26, 'car': 0, 'motorcycle': 0, 'airplane': 0, 'bus': 0, 'train': 0, 'truck': 0, 'boat': 0, 'traffic light': 0, 'fire hydrant': 0, 'stop sign': 0, 'parking meter': 0, 'bench': 17, 'bird': 0, 'cat': 0, 'dog': 0, 'horse': 0, 'sheep': 0, 'cow': 0, 'elephant': 0, 'bear': 0, 'zebra': 0, 'giraffe': 0, 'backpack': 0, 'umbrella': 21, 'handbag': 25, 'tie': 19, 'suitcase': 20, 'frisbee': 0, 'skis': 0, 'snowboard': 0, 'sports ball': 27, 'kite': 0, 'baseball bat': 0, 'baseball glove': 0, 'skateboard': 0, 'surfboard': 0, 'tennis racket': 0, 'bottle': 0, 'wine glass': 24, 'cup': 16, 'fork': 0, 'knife': 0, 'spoon': 0, 'bowl': 15, 'banana': 0, 'apple': 0, 'sandwich': 0, 'orange': 0, 'broccoli': 0, 'carrot': 0, 'hot dog': 0, 'pizza': 0, 'donut': 0, 'cake': 23, 'chair': 1, 'couch': 6, 'sofa': 6, 'potted plant': 2, 'plant': 2, 'bed': 7, 'dining table': 9, 'table': 9, 'toilet': 10, 'tv': 12, 'tv_monitor': 12, 'laptop': 29, 'mouse': 0, 'remote': 28, 'keyboard': 0, 'cell phone': 0, 'microwave': 18, 'oven': 14, 'toaster': 0, 'sink': 3, 'refrigerator': 11, 'cabinet': 11, 'book': 5, 'clock': 13, 'vase': 4, 'scissors': 0, 'teddy bear': 22, 'hair drier': 0, 'toothbrush': 0}
        # self._internal_dict_inverted = {v: k for k, v in self._internal_dict.items()}

        self.YOLACT_COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                        'scissors', 'teddy bear', 'hair drier', 'toothbrush')


        '''
        load/create all necessary modules
        '''
        self.planner = DDPPOAgent_PointNav(config)
        # self.planner = None


        ########################################################################
        model_path = SavePath.from_str(config.BEYOND.SSEG.CHECKPOINT)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args_config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args_config)
        set_cfg(args_config)

        print('Loading model...', end='')
        self.sseg = Yolact()
        if(config.BEYOND.SSEG.LOAD_CHECKPOINT):
            print(config.BEYOND.SSEG.CHECKPOINT, end='')
            self.sseg.load_weights(config.BEYOND.SSEG.CHECKPOINT)
        else:
            print("using random values for the SSEG")
        #freeze weights
        self.sseg.eval()
        self.sseg.to(self.device)
        print(' Done.')
        ########################################################################

        '''
        So far we only will use the Global_Policy
        '''

        self.g_policy = model.Global_Policy(input_shape=config.BEYOND.GLOBAL_POLICY.INPUT_SHAPE, hidden_size=config.BEYOND.GLOBAL_POLICY.HIDDEN_SIZE, is_mp3d=False)
        self.g_policy.to(self.device)

        print("loading model to device",self.device)
        if(config.BEYOND.GLOBAL_POLICY.LOAD_CHECKPOINT):
            checkpoint_filepath = config.BEYOND.GLOBAL_POLICY.CHECKPOINT
            print("loading model...",checkpoint_filepath)

            model_dict = self.g_policy.state_dict()
            pretrained_dict = torch.load(checkpoint_filepath)


            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

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

        # print("GLOBAL_POLICY's state_dict:")
        # print("-----------------------------------------")
        # for param_tensor in self.g_policy.state_dict():
        #     print(param_tensor, "\t", self.g_policy.state_dict()[param_tensor].size())
        # print("-----------------------------------------")

        self.reset_map(self.batch_size)

    def reset_map(self, batch_size):
        self.batch_size = batch_size
        self.mapper_wrapper = mapper.MapperWrapper(map_size_meters=self.config.BEYOND.GLOBAL_POLICY.MAP_SIZE_METERS, map_cell_size=self.config.BEYOND.GLOBAL_POLICY.MAP_CELL_SIZE,config=self.config,n_classes=self.config.BEYOND.GLOBAL_POLICY.N_CLASSES, batch_size=self.batch_size)

    def reset_orientation(self,observations):
        #update orientation of episode start
        for i in range(self.batch_size):
            pose,phi = mapper.pose_from_angle_and_position(observations['compass'][i].cpu(),observations['gps'][i].cpu())
            if(phi>2*np.pi):
                phi=phi-(2*np.pi)

            orientation_new = (phi/2*np.pi).unsqueeze(-1)
            self.orientation[i]=orientation_new
        self.orientation.to(self.device)

    def reset(self, env_idx=0):
        # print("reset")
        # self.array_of_preds=[[] for _ in range(self.batch_size)]
        # self.ep_iteration=torch.zeros(self.batch_size)
        #by idx
        self.array_of_preds[env_idx]=[]
        self.ep_iteration[env_idx]=0
        # self.reset_map_batch_idx(env_idx)
        self.mapper_wrapper.reset_map_batch_idx(env_idx)
        self.planner.reset()

    def evalimage(self,frame):
        # print("frame.shape",frame.shape)
        #preprocess
        frame_transformed = FastBaseTransform_rgb_2_rgb()(frame)
        #feed forward
        preds = self.sseg(frame_transformed)
        #postprocess
        # y_pred_scores = torch.zeros(self.batch_size,frame.shape[1],frame.shape[2],self.config.BEYOND.GLOBAL_POLICY.N_CLASSES,device=self.device)
        y_pred_scores = torch.zeros(self.batch_size,self.config.BEYOND.GLOBAL_POLICY.N_CLASSES,frame.shape[1],frame.shape[2],device=self.device)
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
            ########################################################################
            #masks is per instance. In this case we want per class
            #we want to merge masks of same class
            #classes follow coco default ordering

            #first group instances with same class
            unique_classes, inverse_indices, counts = torch.unique(classes,sorted=True,return_inverse=True,return_counts=True)

            # unique_classes x counts
            unique_classes_set = [[] for _ in range(unique_classes.shape[0])]
            for instance in range(masks.shape[0]):
                unique_classes_set[ inverse_indices[instance].item() ].append(masks[instance])

            #we want a single image per class
            for j in range(len(unique_classes_set)):
                unique_classes_set[j] = torch.stack( unique_classes_set[j] ).permute(1,2,0)
                # unique_classes_set[j], _ = torch.max( unique_classes_set[j], dim=-1 )
                tmp, _ = torch.max( unique_classes_set[j], dim=-1 )
                #then we need to order thoses images following the _internal_dict order
                # print("j",j,"unique_classes[j]",unique_classes[j],"self.YOLACT_COCO_CLASSES[unique_classes[j] ]",self.YOLACT_COCO_CLASSES[unique_classes[j] ],"self._internal_dict[ self.YOLACT_COCO_CLASSES[unique_classes[j] ] ]",self._internal_dict[ self.YOLACT_COCO_CLASSES[unique_classes[j] ] ])
                y_pred_scores[i][ self._internal_dict[ self.YOLACT_COCO_CLASSES[unique_classes[j] ] ] ] = tmp

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

        return y_pred_scores

    def compute_pointgoal_with_pred(self,observations, y_pred_reprojected_goal):
        ###############################################
        '''
        Convert the point goal prediction to pointgoal_with_gps_compass
        '''
        '''
        I just need to convert from cartesian_to_polar
        '''
        # observations['pointgoal_with_gps_compass']=torch.zeros(self.batch_size,2,device=self.device)
        # for i in range(self.batch_size):
        #     rho, phi = cartesian_to_polar(y_pred_reprojected_goal[i][0].item(),y_pred_reprojected_goal[i][1].item())
        #     observations['pointgoal_with_gps_compass'][i][0] = rho
        #     observations['pointgoal_with_gps_compass'][i][1] = phi
        # print("observations['pointgoal_with_gps_compass']",observations['pointgoal_with_gps_compass'])

        # observations['pointgoal_with_gps_compass'] = [cartesian_to_polar(y_pred_reprojected_goal[i][0],y_pred_reprojected_goal[i][1]) for i in range(self.batch_size) ]

        #fake for now
        # observations['pointgoal_with_gps_compass']=torch.zeros(self.batch_size,2,device=self.device)
        # print("observations['gps']",observations['gps'],"observations['compass']",observations['compass'],"y_pred_reprojected_goal",y_pred_reprojected_goal)

        '''
        This is important gps is in episodic zx notation

        pointgoal_with_gps_compass should be in world xyz, we will attempt to use episodic xyz
        '''
        #so zx to xyz, y=0 since it is episodic
        agent_position = torch.zeros(self.batch_size,3)
        agent_position[:,0] = observations['gps'][:,1]
        agent_position[:,2] = observations['gps'][:,0]

        #prediction is in episodic ZX meters so do the same
        goal_position = torch.zeros(self.batch_size,3)
        goal_position[:,0] = y_pred_reprojected_goal[:,1]
        goal_position[:,2] = y_pred_reprojected_goal[:,0]

        # agent_position = observations['gps']
        '''
        Here this is important as well, compass represent the agent angle. The angle is 0 at state t=0.
        rotation_world_agent should be a quaternion representing the the true rotation betwen the agent
        and the world. Since we will attempt to use episodic coo instead of world coo.

        We will adapt this. We will convert the angle to a quaternion
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)
        '''
        observations['pointgoal_with_gps_compass'] = torch.zeros(self.batch_size,2,device=self.device)

        # rotation_world_agent = torch.zeros(self.batch_size,4,device=self.device)
        for i in range(self.batch_size):
            phi = observations['compass'][i].cpu()
            q_r,q_i,q_j,q_k = mapper.angle_to_quaternion_coeff(phi, [0.,1.,0.])
            rotation_world_agent = np.quaternion(q_r,q_i,q_j,q_k)

            # print("agent_position",agent_position,"rotation_world_agent",rotation_world_agent,"goal_position",goal_position)
            observations['pointgoal_with_gps_compass'][i] = torch.from_numpy(mapper._compute_pointgoal(agent_position[i].detach().numpy(), rotation_world_agent, goal_position[i].detach().numpy())).to(self.device)
            # rotation_world_agent[i] = mapper.quaternion_from_coeff(observations['compass'][i])

        # observations['pointgoal_with_gps_compass'] = mapper._compute_pointgoal(agent_position, rotation_world_agent, goal_position)
        # print("observations['pointgoal_with_gps_compass']",observations['pointgoal_with_gps_compass'])

        return observations

    def act(self, observations_pre, env_idx=0):
        with torch.no_grad():
            observations = batch_obs([observations_pre], device=self.device)
            del observations_pre
            observations['pointgoal_with_gps_compass'] = torch.zeros(self.batch_size,2,device=self.device)
            ####################################################################
            '''
            DEBUG
            '''
            # obj_i = self.m3pd_to_gibson_inverted[ observations['objectgoal'][0][0].item() ]
            # im_debug = PIL.Image.fromarray(observations['rgb'][0].cpu().numpy().astype(np.uint8))
            # im_debug.save("habitat-challenge-data/debug/"+str(obj_i)+"_"+str(int(self.ep_iteration[0].item()))+".jpeg")
            #
            ####################################################################
            '''
                Set episodic orientation
            '''
            if(self.ep_iteration[env_idx]==0):
                self.reset_orientation(observations)
            ####################################################################
            '''
            Deal with object goal
            '''
            object_goal = torch.zeros(self.batch_size, device=self.device).long()
            if(self.config.BEYOND.GLOBAL_POLICY.USE_MATTERPORT_TO_GIBSON):
                for i in range(self.batch_size):
                    obj_i = self.m3pd_to_gibson_inverted[ observations['objectgoal'][i][0].item() ]
                    object_goal[i] = self._internal_dict[obj_i]
            object_goal = object_goal.unsqueeze(-1)

            ####################################################################
            '''
                First call yolact and segment the rgb
            '''
            ####################################################################
            y_pred_scores = self.evalimage(observations['rgb'])
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
            self.ep_iteration[env_idx]+=1

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

##########################################################################


##########################################################################
class DDPPOAgent_PointNav(Agent):
    def __init__(self, config: Config, batch_size=1):
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
        observation_spaces = Dict(spaces)

        action_space = Discrete(len(config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS))

        self.device = torch.device("cuda:{}".format(config.TORCH_GPU_ID))
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

        if config.MODEL_PATH:
            ckpt = torch.load(config.MODEL_PATH, map_location=self.device)
            print(f"Checkpoint loaded: {config.MODEL_PATH}")
            print("ckpt",ckpt)
            del ckpt['config']

            # torch.save(ckpt, config.MODEL_PATH+"_new")
            torch.save(ckpt, "/habitat-challenge-data/ddppo_pointnav_habitat2020_challenge_baseline_v1.pth_new")

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

    def reset(self):
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

        with torch.no_grad():
            _, action, _, self.test_recurrent_hidden_states = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=False,
            )
            #  Make masks not done till reset (end of episode) will be called
            self.not_done_masks.fill_(1.0)
            self.prev_actions.copy_(action)

        return action
##########################################################################
