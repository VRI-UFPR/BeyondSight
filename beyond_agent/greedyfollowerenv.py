'''
    load libraries
'''
##########################################################################

#load pytorch stuff
import torch
import numpy as np
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.core.env import RLEnv
################################################################################
from habitat.core.logging import logger

################################################################################
from PIL import Image
import json
import pycocotools.mask
import os
################################################################################

def mask2bbox(mask):
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    rmin, rmax = torch.where(rows)[0][[0, -1]]
    cmin, cmax = torch.where(cols)[0][[0, -1]]

    return cmin, rmin, cmax - cmin, rmax - rmin

class GreedyFollowerEnv(RLEnv):
    def __init__(self, config, dataset=None, env_ind=0, is_train=True, internal_idx=0):
        super(GreedyFollowerEnv, self).__init__(config, dataset)

        '''
            Wrapper logic
        '''
        self._epoch_counter = 0
        self._my_ep_counter = 0
        self._internal_env_idx = internal_idx
        self._epoch_perc = int(max(np.ceil(len(self._env.episodes)/100),1))

        self._SLACK_REWARD = -1e-4
        # self._SUCCESS_REWARD = 2.5
        #best result for eval so far
        # self._follower_goal_radius = self._env._config.TASK.SUCCESS.SUCCESS_DISTANCE
        self._follower_goal_radius = self._env._config.TASK.SUCCESS.SUCCESS_DISTANCE

        # self.is_train = False
        self.is_train = is_train

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
        self._updated_dict= {k: v+1 for k, v in self._updated_dict.items()}
        self._updated_dict_reverse = {v: k for k, v in self._updated_dict.items()} #this reverse key with value

        '''
        WARNING UNCOMMENT IT WHEN USING matterport3D
        '''
        self.dict_map = self.create_mapping_dict()

        self.object_goal = None
        self.agent_starting_state = None
        self.shortest_path = None
        self._follower_idx = 0
        self.smallest_dist = -1

        self._follower_step_size = self._env._config.SIMULATOR.FORWARD_STEP_SIZE
        self._follower_max_visible_dis = self._env._config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH

        # self._target_seen_this_step = 0
        self._target_seen = 0
        self._exp_reward = 0
        # self._has_given_first_obs_reward = 0

        self._scene_id = None
        self._previous_measure = 0.0
        # self._predict_is_in_target_cell = 0

        # max_value_axis = 256
        # self._max_diagonal_value = np.hypot(max_value_axis,max_value_axis)
        # self._reward_scaling = 10*(1/self._max_diagonal_value)
        self._reward_scaling = (10*(1/(64*64))) *100

        #for coco gt
        self.ann_id = 1
        self.image_id = 1
        self.annotations = []
        self.images = []
        self.closest_goal = 0
    @property
    def episode_over(self):
        return self._env.episode_over

    def create_coco_mask(self, raw_semantic_output):
        ########################################################################
        # raw_semantic_output = torch.from_numpy(raw_semantic_output.astype(np.int32)).to("cuda")

        u_sem = torch.unique(raw_semantic_output)

        # oid_to_semantic_class = self.get_class_labels(u_sem.cpu())
        # oid_to_semantic_class = torch.from_numpy(oid_to_semantic_class).to("cuda")

        oid_to_semantic_class = self.dict_map[self._scene_id][u_sem.long()].long()
        oid_to_semantic_class_useful = torch.nonzero(oid_to_semantic_class)

        # print("u_sem",u_sem.shape,flush=True)
        # print("oid_to_semantic_class",oid_to_semantic_class.shape,flush=True)
        # print("oid_to_semantic_class_useful",oid_to_semantic_class_useful.shape,flush=True)

        masks = []
        for i in oid_to_semantic_class_useful:
            for j in i:
                mask = torch.where(raw_semantic_output == u_sem[j], 1.0, 0.0 )
                masks.append({'mask':mask,'semantic_class':oid_to_semantic_class[j] },)
        ########################################################################
        return masks

    def create_mapping_dict(self):
        '''
        create a mapping per scene of object ID to class ID, class 0 is background
        '''
        max_objs = 4096

        dict_scene = {}
        keys = list(self._env._dataset.goals_by_category.keys())
        for key in keys:
            key = key.split("_")[0]
            if not(key in dict_scene):
                dict_scene[key]=torch.zeros(max_objs,device="cuda")


        for cat in self._env._dataset.goals_by_category:
            key = cat.split("_")[0]
            for obj in self._env._dataset.goals_by_category[cat]:
                dict_scene[key][obj.object_id]=self._updated_dict[obj.object_category]

        return dict_scene

    def get_sseg_dict(self):
        return self.dict_map

    def get_distance_to_goal(self):
        source_position = self._env._sim.get_agent_state().position
        target_position = self._follower_target_viewpoint_pos
        # dist = self._env._sim.geodesic_distance(source_position,target_position)

        tmp_diff = target_position - source_position
        dist = np.linalg.norm(tmp_diff, ord=2, axis=-1)

        return dist

    def randomize_env(self):
        self._env._episode_iterator._shuffle_iterator()

    def get_reward_range(self):
        # return [-1e-4, 1-1e-4]
        return [(-1e-4)-0.25, 2.5+0.25+(-1e-4)]
        # return [-1e-4-0.25-0.1, 0.1+1.5+2.5+0.25-1e-4]

    def get_reward(self, observations, oracle_action):
        # reward = (self._predict_is_in_target_cell)#[-1e-4,1-1e-4]
        reward = -1e-4

        if(self._exp_reward == oracle_action):
            reward += 1e-3
        # reward += self._exp_reward

        # if(self._has_given_first_obs_reward==0 and self._target_seen>0):
        #     self._has_given_first_obs_reward = 1.0
        #     # reward += 1.25#half of success

        infos= self._env.get_metrics()
        reward_tmp = (self._previous_measure - infos['distance_to_goal'])
        self._previous_measure = infos['distance_to_goal']

        reward_tmp += infos['success']*2.5#will give a huge bonus to successful states, since it is a sparse state
        # reward_tmp += infos['success']*3.0#will give a huge bonus to successful states, since it is a sparse state
        reward = reward + reward_tmp

        #scale it down to reduce massive loss variation
        # reward = reward/2.5

        return reward

    def get_done(self, observations):
        return self._env.episode_over

    def get_metrics(self):
        infos= self._env.get_metrics()
        infos['step']= self._follower_idx
        infos['first_obs']= self._target_seen

        elements = list(self._updated_dict.keys())
        for i in elements:
            infos["success_"+i]= 0.0

        string_name = 'success_'+self._env._current_episode.object_category
        infos[string_name]= infos['success']
        infos["ep_reward"]= self.ep_reward
        return infos

    def get_metrics_extended(self):
        infos= self._env.get_metrics()

        infos['step']= self._follower_idx
        infos['first_obs']= self._target_seen
        infos['object_goal']= self._env._current_episode.object_category
        infos['episode_id']= self._env._current_episode.episode_id
        # infos['agent_starting_state']= {'position': self.agent_starting_state.position, 'rotation': self.agent_starting_state.rotation}
        # infos['geodesic_distance']= self.smallest_dist

        return infos

    def get_info(self, observations):
        # return self._env.get_metrics()
        infos= self._env.get_metrics()
        infos['step']= self._follower_idx
        infos['first_obs']= self._target_seen

        elements = list(self._updated_dict.keys())
        for i in elements:
            infos["success_"+i]= 0.0

        string_name = 'success_'+self._env._current_episode.object_category
        infos[string_name]= infos['success']
        infos["ep_reward"]= self.ep_reward

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

    # def get_next_action_eval(self):
        # return self._follower.get_next_action(self._follower_target_viewpoint_pos)

    def get_next_action(self):
        return self._follower.get_next_action(self.closest_goal)

    # def get_next_action(self):
    #
    #     action = self._follower.get_next_action(self.shortest_path[-1])
    #     # if(self._follower_idx==self._env._config.ENVIRONMENT.MAX_EPISODE_STEPS-1):
    #     #     action = 0
    #
    #     # print("inside=\n",self.shortest_path)
    #     # exit()
    #
    #     # print(self._follower_idx,len(self._env._current_episode.shortest_paths[0]),len(self._env._current_episode.shortest_paths[0])-1,)
    #
    #     # if self._follower_idx > len(self._env._current_episode.shortest_paths[0])-1:
    #     #     action = 0
    #     # else:
    #     #     action = self._env._current_episode.shortest_paths[0][self._follower_idx].action
    #
    #     self._follower_idx+=1
    #
    #     # action = self._follower.get_next_action(self.shortest_path[self._follower_idx])
    #
    #     return action

    def reset(self):
        tmp = self._env.reset()
        self._follower_idx = 0
        self._target_seen = 0
        self._exp_reward = 0
        self.ep_reward = 0
        # self._has_given_first_obs_reward = 0

        # self._target_seen_this_step = 0
        # self._predict_is_in_target_cell = 0
        self._scene_id = self._env._current_episode.scene_id.split("/")[-1]

        self._previous_measure = self._env.get_metrics()[
            "distance_to_goal"
        ]

        self.agent_starting_state = self._env._sim.get_agent_state()
        self.update_goal()

        # self.update_json()

        # self.update_goal()
        # tmp['oracle']=torch.from_numpy(np.array([self.get_next_action()] ))
        # tmp['semantic2']=self.dict_map[self._scene_id][tmp['semantic'].long()].long()
        # tmp['semantic']=self.dict_map[self._scene_id][tmp['semantic'].long()].long()
        # tmp['distance_to_goal']=self._previous_measure
        # info = self.get_info(tmp)
        # tmp['top_down_map']=info['top_down_map']
        return tmp

    # def reset(self):
    #     tmp = self._env.reset()
    #     # print("reset",self._env._current_episode.scene_id.split("/")[-1] )
    #     self._follower_idx = 0
    #     self._target_seen = 0
    #     self.agent_starting_state = self._env._sim.get_agent_state()
    #     self.update_goal()
    #     tmp['oracle']=self.get_next_action()
    #     self._scene_id = self._env._current_episode.scene_id.split("/")[-1]
    #     tmp['semantic2']=self.dict_map[self._scene_id][tmp['semantic'].long()].long()
    #
    #     infos= self._env.get_metrics()
    #     tmp['top_down_map']=infos['top_down_map']
    #
    #
    #     #debug
    #     # print(self._updated_dict_reverse[tmp['objectgoal'][0]])
    #
    #     return tmp

    def step(self, *args, **kwargs):
        r"""Perform an action in the environment.
        :return: :py:`(observations, reward, done, info)`
        """

        # '''
        # TO AVOID CONVERGENCY TO EARLY STOP TRANSFORM STOP TO FORWARD
        # '''
        # info = self.get_info(None)
        # ########################################################################
        # action_tmp = args[0]#action
        # if action_tmp == 0:
        #     if info['distance_to_goal'] > 0.2:
        #         args = (1,)
        ########################################################################

        oracle_action = self.get_next_action()
        observations = self._env.step(*args, **kwargs)

        # observations['oracle']=self.get_next_action()
        #object id to semantic id
        # observations['semantic']=self.dict_map[self._scene_id][observations['semantic'].long()].long()

        self._follower_idx+=1

        # #######################################################################
        # #internal for reward
        '''
        env needs to receive computed_reward to properly account for
        first observation
        '''
        # #######################################################################

        reward = self.get_reward(observations, oracle_action)
        self.ep_reward += reward
        done = self.get_done(observations)
        if(done):

            # self.save_json()

            # if(self._my_ep_counter % self._epoch_perc == 0):
            #     logger.info("EPOCH {}\t in env {}\t ep {}\t".format(self._epoch_counter, self._internal_env_idx, self._my_ep_counter))

            self._my_ep_counter += 1

            if(self._my_ep_counter == len(self._env.episodes)):
                #a epoch is complete all episodes were computed at least once
                logger.info("END OF EPOCH {}\t in env {}\t".format(self._epoch_counter, self._internal_env_idx))

                self._my_ep_counter = 0
                self._epoch_counter += 1
        info = self.get_info(observations)

        # observations['distance_to_goal']=info['distance_to_goal']

        # #######################################################################
        # #internal for reward
        # if not(self._target_seen):
        #     u_sem2 = torch.unique(observations['semantic'])
        #     if(observations['objectgoal'][0]+1 in u_sem2):
        #         print("activate heuristic",flush=True)
        #         self._target_seen = 1.0
        #         #force early stop to first obs
        #         if(info['success'] != 1.0):
        #             self._env._episode_over = True
        #             reward += 2.5
        #             info['success'] = 1.0
        #             done = True
        #             # self._env._task.measurements.measures['success'] = 1.0
        # #######################################################################

        # observations['top_down_map']=info['top_down_map']
        return observations, reward, done, info

    def create_coco_gt(self, sample):
        if(sample['masks_per_instance']):
            # #######################################################################
            for j,mask in enumerate(sample['masks_per_instance']):
                # rle = pycocotools.mask.encode( np.asfortranarray( np.expand_dims(mask['mask'],axis=-1).astype(np.uint8) ) )
                rle = pycocotools.mask.encode( np.asfortranarray( mask['mask'].unsqueeze(-1).byte().cpu().numpy() ) )
                rle = rle[0]
                rle['counts'] = rle['counts'].decode('ascii')

                img = pycocotools.mask.decode(rle)

                self.annotations.append({
                    'id': int(self.ann_id),
                    'image_id': int(self.image_id),
                    'category_id': int(mask['semantic_class'].cpu().item()),
                    'segmentation': rle,
                    'area': float( torch.sum(mask['mask']).cpu().item() ),
                    'bbox': [int(x) for x in mask2bbox(mask['mask'].bool())],
                    'iscrowd': int(0)
                })

                self.ann_id+=1
            ############################################################################

            img_name="./imgs/"+str(self.image_id)+".jpg"
            img = sample['rgb']

            self.images.append({
                'id': int(self.image_id),
                'width': int(img.shape[1]),
                'height': int(img.shape[0]),
                'file_name': img_name
            })

            im = Image.fromarray(img.cpu().numpy())
            path = "/habitat-challenge-data/mp3d_2021_coco_style/val_mini"
            im.save(path+"/"+img_name, quality=95, subsampling=0)

            self.image_id += 1
            ####################################################################

    def save_json(self):
        path = "/habitat-challenge-data/mp3d_2021_coco_style/val_mini"
        info = {
            'year': int(2021),
            'version': int(1),
            'description': 'Matterport3D-Habitat-challenge-2021',
        }

        categories = [{'id': x+1} for x in range(21)]

        filename=path+"/"+'mp3d_2021_coco_style.json'
        # print("saving",filename)

        if os.path.isfile(filename):
            # print("updating existent file",flush=True)

            with open(filename, 'r') as f:
                tmp_dict = json.load(f)

            tmp_dict['images'] = tmp_dict['images']+self.images
            tmp_dict['annotations'] = tmp_dict['annotations']+self.annotations

            with open(filename, 'w') as f:
                json.dump(tmp_dict, f)
        else:
            # print("creating new file",flush=True)
            with open(filename, 'w') as f:
                json.dump({
                    'info': info,
                    'images': self.images,
                    'annotations': self.annotations,
                    'licenses': {},
                    'categories': categories
                }, f)

    def update_json(self):
        path = "/habitat-challenge-data/mp3d_2021_coco_style/val_mini"
        filename=path+"/"+'mp3d_2021_coco_style.json'
        self.annotations = []
        self.images = []
        if os.path.isfile(filename):
            # print("updating existent file")

            with open(filename, 'r') as f:
                tmp_dict = json.load(f)

            self.image_id = len(tmp_dict['images'])+1
            self.ann_id = len(tmp_dict['annotations'])+1
        else:
            self.image_id = 1
            self.ann_id = 1

    def step_obs_only(self, *args, **kwargs):
        observations, reward, done, info = self.step(*args, **kwargs)
        return observations

    ######################################################################
    def send_computed_reward(self, reward):
        # self._predict_is_in_target_cell = reward
        # logger.info("reward['exp_reward'] {}\t reward['seen_target'] {}\t".format(reward['exp_reward'], reward['seen_target']))

        self._exp_reward = reward['exp_reward']

        if(reward['seen_target']>0):
            self._target_seen=reward['seen_target']

        return True
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
        self.closest_goal = target_position[0]
        ########################################################################

        # smallest_dist = self._env._sim.geodesic_distance(self.agent_starting_state.position,target_position)
        # self.closest_goal = np.copy(self._env._current_episode._shortest_path_cache.points)[-1]


        # self.smallest_dist = smallest_dist
        #
        # # print("self._env._current_episode._shortest_path_cache",self._env._current_episode._shortest_path_cache)
        #
        # self.shortest_path = np.copy(self._env._current_episode._shortest_path_cache.points)
        # self.shortest_path = np.array(self.shortest_path )
        # self.shortest_path = self.shortest_path[1:]

        #debug
        # print(self.smallest_dist)
