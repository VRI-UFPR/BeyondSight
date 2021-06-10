'''
This code is intended to create new episodes, in a offline fashion, on the matterport3d dataset
using specific rules
'''
################################################################################
##imports
from habitat_sim.errors import GreedyFollowerError
from habitat.tasks.nav.object_nav_task import ObjectGoalNavEpisode
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.datasets.utils import get_action_shortest_path
from habitat.core.simulator import Simulator
from typing import Optional
from habitat.core.env import RLEnv
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

from config_default import get_config, get_task_config

import argparse
import habitat
import random
import numpy as np
import os

import quaternion

import gzip
import json
import copy
import sys
import torch
# import time

################################################################################
from PIL import Image
import pycocotools.mask
################################################################################


################################################################################
def process_file(jsonfilename):
    with gzip.GzipFile(jsonfilename, 'r') as fin:    # 4. gzip
        json_bytes = fin.read()                      # 3. bytes (i.e. UTF-8)

    json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
    data = json.loads(json_str)                      # 1. data
    return data

def write_file(jsonfilename, data):
    with gzip.GzipFile(jsonfilename, 'w') as fout:
        fout.write(data.encode('utf-8'))
################################################################################

################################################################################
# def _create_episode_objectnav(
#     episode_id,
#     scene_id,
#     start_position,
#     start_rotation,
#     shortest_paths=None,
#     radius=None,
#     info=None,
#     object_category="",
# ):
#     # goals = [NavigationGoal(position=target_position, radius=radius)]
#     goals = []
#     return ObjectGoalNavEpisode(
#         episode_id=str(episode_id),
#         goals=goals,
#         scene_id=scene_id,
#         start_position=start_position,
#         start_rotation=start_rotation,
#         shortest_paths=shortest_paths,
#         info=info,
#         object_category=object_category
#     )

def _create_episode_objectnav(
    episode_id,
    scene_id,
    start_position,
    start_rotation,
    shortest_paths=None,
    radius=None,
    info=None,
    object_category="",
):
    goals = []
    return {
        "episode_id":episode_id,
        "scene_id":scene_id,
        "start_position":(np.array(start_position)).tolist(),
        "start_rotation":start_rotation,
        "info":info,
        "goals":goals,
        "start_room": None,
        "shortest_paths":shortest_paths,
        "object_category":object_category,
    }

################################################################################
def gen_epset_per_scene(filename, scene_name, scenes, folder):
    data = process_file(filename)
    scene_id = data['episodes'][0]['scene_id']
    episodes = []
    '''
    todo load scene list
    '''

    balance = {
      'chair': 0.010768869652583848,
      'table': 0.017361933898035333,
      'picture': 0.09553557527635062,
      'cabinet': 0.030240867368939814,
      'cushion': 0.044579046304452453,
      'sofa': 0.0574279664486845,
      'bed': 0.055730260543792924,
      'chest_of_drawers': 0.0982684415974377,
      'sink': 0.16734846935296924,
      'toilet': 0.2991650923273813,
      'stool': 0.09019254795325936,
      'towel': 0.31048004553791664,
      'tv_monitor': 1.0,
      'shower': 0.18184849046119653,
      'counter': 0.07522449354868369,
      'clothes': 0.8038316685770427,
      'plant': 0.061791954080861994,
      'bathtub': 0.29891006515253,
      'gym_equipment': 0.42959656952831016,
      'seating': 0.09791758088323294,
      'fireplace': 0.1696502626486038
    }

    current_episode_id=0
    for class_name in scenes[scene_name]:
        fullclass = scene_name+".glb_"+class_name

        current_episode_in_class=0
        episodes_in_class=[]
        for i, object in enumerate(data['goals_by_category'][fullclass]):
            for j, view_points in enumerate(data['goals_by_category'][fullclass][i]['view_points']):
                source_position=data['goals_by_category'][fullclass][i]['view_points'][j]['agent_state']['position']
                source_rotation=data['goals_by_category'][fullclass][i]['view_points'][j]['agent_state']['rotation']


                episode_new = _create_episode_objectnav(
                    episode_id=int(current_episode_id),
                    scene_id=str(scene_id),
                    start_position=source_position,
                    start_rotation=source_rotation,
                    shortest_paths=[],
                    radius=0.1,
                    info=[],
                    object_category=class_name,
                )
                episodes_in_class.append(episode_new)
                current_episode_id+=1
                current_episode_in_class+=1

        #force balance
        value = int( max( np.rint(current_episode_in_class*balance[class_name]),1 ) )
        if(value):
            episodes=episodes + np.random.choice(episodes_in_class, size=value).tolist()
    ###########################################
    ##########################################
    if(episodes):
        data['episodes']=episodes
        data = json.dumps(data)
        filename_new = folder+scene_name+'.json.gz'
        print("writing",filename_new)
        write_file(filename_new,data)
################################################################################

































def mask2bbox(mask):
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    rmin, rmax = torch.where(rows)[0][[0, -1]]
    cmin, cmax = torch.where(cols)[0][[0, -1]]

    return cmin, rmin, cmax - cmin, rmax - rmin
################################################################################
class GreedyFollowerEnv(habitat.RLEnv):
    def __init__(self, config, dataset=None, env_ind=0, split=0):
        super(GreedyFollowerEnv, self).__init__(config, dataset)

        self.split=split
        # self._data = process_file(json_file)
        '''
            Wrapper logic
        '''

        self._follower_goal_radius = 0.1
        self._follower_target_viewpoint_pos = None

        self._follower = ShortestPathFollower(self._env._sim, goal_radius=self._follower_goal_radius, return_one_hot=False)

        self.iteration = 0

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

    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self._env.episode_over

    @property
    def episode_over(self):
        return self._env.episode_over

    def get_info(self, observations):
        return self._env.get_metrics()

    def get_metrics_extended(self):
        tmp= self._env.get_metrics()
        tmp['step']=self.iteration
        return tmp

    def step(self, *args, **kwargs):
        pass

    def reset_env(self):
        r"""Resets the environments and returns the initial observations.
        :return: initial observations from the environment.
        """
        self._env._reset_stats()

        assert len(self._env.episodes) > 0, "Episodes list is empty"
        # Delete the shortest path cache of the current episode
        # Caching it for the next time we see this episode isn't really worth
        # it
        if self._env._current_episode is not None:
            self._env._current_episode._shortest_path_cache = None

        self._env._current_episode = next(self._env._episode_iterator)
        self._env.reconfigure(self._env._config)

        observations = self._env.task.reset(episode=self._env.current_episode)
        # self._task.measurements.reset_measures(
        #     episode=self.current_episode, task=self.task
        # )

        return observations

    def reset(self):
        self.iteration = 0
        # tmp = self._env.reset()
        observations = self.reset_env()

        self._scene_id = self._env._current_episode.scene_id.split("/")[-1]
        self.update_json()

        ########################################################################
        '''
        create coco json and jpgs for yolact transfer learning
        '''
        #i want semantic using oid in this step
        masks = self.create_coco_mask(observations['semantic'])
        self.create_coco_gt({'masks_per_instance': masks, 'rgb': observations['rgb'] })
        ########################################################################

        self.save_json()
        return observations

    def create_coco_mask(self, raw_semantic_output):
        ########################################################################
        # raw_semantic_output = torch.from_numpy(raw_semantic_output.astype(np.int32)).to("cuda")

        u_sem = torch.unique(raw_semantic_output)
        #discart zeros!
        # u_sem = u_sem[u>0]

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
            path = "/habitat-challenge-data/mp3d_2021_coco_style/viewpoints_train/"+self.split
            im.save(path+"/"+img_name, quality=95, subsampling=0)

            self.image_id += 1
            ####################################################################

    def save_json(self):
        path = "/habitat-challenge-data/mp3d_2021_coco_style/viewpoints_train/"+self.split
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
        path = "/habitat-challenge-data/mp3d_2021_coco_style/viewpoints_train/"+self.split
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
################################################################################


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
        "--split", type=int
    )
    # parser.add_argument(
    #     "--input-type", default="blind", choices=["blind", "rgb", "depth", "rgbd"]
    # )
    # parser.add_argument(
    #     "--evaluation", type=str, required=True, choices=["local", "remote"]
    # )

    # '''
    # step one load json with scenes names
    # '''
    #
    # #load scenes_with_containing_classes
    # scenes_per_class_dict = process_file("/habitat-challenge-data/scenes_with_containing_classes.json.gz")
    #
    # '''
    # step two generate ep json per scene
    # '''
    # folder ='/habitat-challenge-data/mine_objectgoal_mp3d/epset_starting_on_viewpoints_balanced/content/'
    # for scene_name in scenes_per_class_dict:
    #     filename = '/habitat-challenge-data/objectgoal_mp3d/train/content/'+scene_name+'.json.gz'
    #     gen_epset_per_scene(filename,scene_name,scenes_per_class_dict,folder)
    #
    # '''
    # step three run sim and extract images and annotations
    # '''
    # print("end phase 2")
    # # exit()
    print("starting phase 3")

    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    # parser.add_argument("--model-path", default="", type=str)
    args = parser.parse_args()


    config = get_config(
        ["configs/beyond.yaml","configs/beyond_trainer.yaml","configs/ddppo_pointnav.yaml", config_paths],
    ).clone()


    config.defrost()
    config.TORCH_GPU_ID = 0
    # config.INPUT_TYPE = args.input_type
    # config.MODEL_PATH = args.model_path
    config.RANDOM_SEED = 7
    config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    config.TASK_CONFIG.TASK.MEASUREMENTS = []

    config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = True
    config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False

    config.DATASET.SPLIT= "epset_starting_on_viewpoints_balanced_split"+str(args.split)
    config.DATASET.DATA_PATH= "/habitat-challenge-data/mine_objectgoal_mp3d/{split}/{split}.json.gz"
    config.ENVIRONMENT.MAX_EPISODE_STEPS = 500
    config.freeze()

    print("-------------------------------------------------------------")

    ########################################################################
    _env = GreedyFollowerEnv(config=config,split=str(args.split))
    num_episodes = None
    if num_episodes is None:
        num_episodes = len(_env.episodes)
    else:
        assert num_episodes <= len(_env.episodes), (
            "num_episodes({}) is larger than number of episodes "
            "in environment ({})".format(
                num_episodes, len(_env.episodes)
            )
        )

    assert num_episodes > 0, "num_episodes should be greater than 0"

    count_episodes = 0
    infos = []
    steps = max(int(num_episodes/100),1)
    print("num_episodes",num_episodes,flush=True)
    print("steps",steps,flush=True)
    valid_episodes = []
    ########################################################################
    for count_episodes in range(num_episodes):
        observations = _env.reset()
        if(count_episodes % steps == 0):
            print(count_episodes,flush=True)

    print("EXIT SUCCESS")
    #everything went fine so exit with 0
    return 0

if __name__ == "__main__":
    main()
##########################################################################
