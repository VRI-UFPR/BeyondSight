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
# import time

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
class GreedyFollowerEnv(habitat.RLEnv):
    def __init__(self, config, dataset=None, env_ind=0, number_retries_per_target=256):
        super(GreedyFollowerEnv, self).__init__(config, dataset)

        '''
            Wrapper logic
        '''
        self._follower_goal_radius = self._env._config.TASK.SUCCESS.SUCCESS_DISTANCE*0.25
        self._follower_target_viewpoint_pos = None

        self._follower = ShortestPathFollower(self._env._sim, goal_radius=self._follower_goal_radius, return_one_hot=False)

        self.number_retries_per_target = number_retries_per_target

        self.computed_points={}

    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self._env.episode_over

    def get_info(self, observations):
        return self._env.get_metrics()

    # def get_next_action(self):
    #     return self._follower.get_next_action(self._follower_target_viewpoint_pos)

    def reset(self):
        tmp = self._env.reset()
        # self.update_goal()
        return tmp
################################################################################

################################################################################
def _create_episode_objectnav(
    episode_id,
    scene_id,
    start_position,
    start_rotation,
    shortest_paths=None,
    radius=None,
    info=None,
    object_category="",
) -> Optional[NavigationEpisode]:
    # goals = [NavigationGoal(position=target_position, radius=radius)]
    goals = []
    return ObjectGoalNavEpisode(
        episode_id=str(episode_id),
        goals=goals,
        scene_id=scene_id,
        start_position=start_position,
        start_rotation=start_rotation,
        shortest_paths=shortest_paths,
        info=info,
        object_category=object_category
    )
################################################################################
# def pick_smallest_in_list(a,b):
#
#     pass

################################################################################
def generator():
    print(os.environ["TRACK_CONFIG_FILE"])

    config = habitat.get_config(
        config_paths=os.environ["TRACK_CONFIG_FILE"])

    print("############################################")
    print("inside python script",config.DATASET.DATA_PATH)
    print("############################################")

    # scenes = process_file("/habitat-challenge-data/scenes_sorted_by_nonzero_instances_of_class.json.gz")

    filename = sys.argv[1]
    print("filename",filename)
    scenes = process_file(filename)

    # max_scenes_dict = {0: for i in len(list(scenes.keys()))}
    # max_scenes = {}
    # for class_name in scenes:
    #     if class_name in max_scenes:
    #         max_scenes[class_name]=max(len(scenes[class_name]),max_scenes[class_name])
    #     else:
    #         max_scenes[class_name]=len(scenes[class_name])

    # n_scenes = len(scenes[class_name])

    # n_episodes_o=8192
    n_episodes_o=1024
    # n_episodes_o=128
    n_episodes=n_episodes_o

    # n_episodes=4096
    # 70 ,        30
    # 5734, 2457
    # 4k to train, 4k to test

    #for testing
    # n_episodes=32
    # n_episodes=64

    jiggle_max_idx = n_episodes*21*100
    jiggle_array = np.random.rand(jiggle_max_idx+1,2)*0.1
    jiggle_idx= 0

    # dst_folder = "/habitat-challenge-data/mine_objectgoal_mp3d/lv1_train/content/"
    #
    # '''
    # check which scenes already exists
    # '''
    # # scenes_already_present = sort(os.listdir(dst_folder))

    # print("scenes=\n",scenes)
    # print("EXITING")
    # exit()

    for class_name in scenes:

        #pick the scene with the smallest number of instances
        # scene_name = scenes[class_name][0]
        scenes_per_class= len(scenes[class_name])
        scenes_perc= 1/scenes_per_class
        print("scenes_per_class",scenes_per_class)
        print("scenes_perc",scenes_perc)

        #do all scenes
        for scene_name in scenes[class_name]:

            n_episodes= max( int(np.ceil(n_episodes_o*scenes_perc)) , 2)
            print("n_episodes",n_episodes)
            ##########################################
            config.defrost()
            # config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = False
            config.DATASET.DATA_PATH = "/habitat-challenge-data/objectgoal_mp3d/train/content/"+scene_name+".json.gz"
            config.freeze()
            ##########################################
            print("############################################")
            print(class_name)
            print(config.DATASET.DATA_PATH)
            print("############################################")

            env = GreedyFollowerEnv(config=config)
            episodes = []
            ##########################################
            with env:
                '''
                find the object and its viewpoints, the viewpoints will the set of targets
                '''
                dataset_json = process_file("/habitat-challenge-data/objectgoal_mp3d/train/content/"+scene_name+".json.gz")
                fullclass = scene_name+".glb_"+class_name

                scene_id = dataset_json['episodes'][0]['scene_id']

                current_targets = []
                for i, object in enumerate(dataset_json['goals_by_category'][fullclass]):
                    current_targets.append(dataset_json['goals_by_category'][fullclass][i]['view_points'])

                points = []
                #per object
                for j in range(len(current_targets)):
                    #per viewpoint
                    for k in range(len(current_targets[j])):
                        if(current_targets[j][k]):
                            points.append(current_targets[j][k]['agent_state']['position'])
                points = np.array(points)

                ##########################################
                #2 episodes per class
                #4096 episodes per class
                print_step_size = max(np.ceil(n_episodes/100),32)
                print_step_size_retry = np.ceil((4*env.number_retries_per_target)/5)

                for current_episode_id in range(n_episodes):
                    if(current_episode_id % print_step_size == 0):
                        print("ep",current_episode_id,flush=True)
                    '''
                    start trying random points now
                    '''
                    for retry in range(env.number_retries_per_target):
                        #over 2/3
                        # if(retry+1 == print_step_size_retry):
                        #     print("retry",retry)
                        ##########################################
                        #random sampling

                        #discard some randoms
                        # for pad in range(10):
                        source_position = env._env._sim.sample_navigable_point()

                        '''
                        Insert a jiggle on the position to minimize repetead src loc
                        '''
                        # print("source_position 0",source_position)
                        source_position = np.array(source_position)
                        source_position[0] = source_position[0] + jiggle_array[jiggle_idx][0]
                        #don't apply to height
                        source_position[2] = source_position[1] + jiggle_array[jiggle_idx][1]

                        jiggle_idx = (jiggle_idx+1) % jiggle_max_idx
                        ########################################################


                        closest_pt = env._env._sim.pathfinder.snap_point(source_position)
                        if not env._env._sim.pathfinder.is_navigable(closest_pt):
                            continue
                        source_position = closest_pt
                        # print("source_position 1",source_position)
                        source_position = list(source_position)
                        # print("source_position 2",source_position)
                        # print("source_position",source_position)

                        tmp_diff = points - source_position
                        euclid_vec = np.linalg.norm(tmp_diff, ord=2, axis=-1)

                        smallest_dist_euclid = np.min(euclid_vec)
                        if(smallest_dist_euclid>10):
                            continue

                        target_positions = points[np.argsort(euclid_vec)]

                        smallest_dist = env._env._sim.geodesic_distance(source_position,target_positions)
                        if(smallest_dist==np.inf):
                            continue
                        ##########################################

                        '''
                        Discard episodes with those criteria, the bigger the ratio the more obstacles in the path.
                        '''

                        '''
                        the first 1/2 of the episodes should be lv0 and the second 1/2 should be lv1
                        '''
                        # if( (current_episode_id < n_episodes/4)or(current_episode_id < (3*n_episodes)/4)  ):
                        ratio = smallest_dist/smallest_dist_euclid
                        if(retry > print_step_size_retry):
                            if((smallest_dist<1)or(smallest_dist>20)or(ratio<1.025)):
                                continue
                        else:
                            if( (current_episode_id < n_episodes/4)or( (current_episode_id > n_episodes/2)and(current_episode_id < (3*n_episodes)/4) ) ):
                                if((smallest_dist<1)or(smallest_dist>2)):
                                    continue
                            else:
                                if((smallest_dist<2)or(smallest_dist>4)):
                                # if((smallest_dist<4)or(smallest_dist>8)or(ratio<1.1)):
                                    continue

                        #debug
                        # print("smallest_dist",smallest_dist,"ratio",ratio)
                        #found no need for further retries
                        ##########################################

                        '''
                        Here we have a issue if the episode already exist and the agent is performing then

                            ep_shortest_path = np.copy(env._env._current_episode._shortest_path_cache.points)
                            ep_shortest_path = np.array(ep_shortest_path )

                        works but here we did not start the episode...

                        Here we can pass a fake episode and then grab the cache to obtain the path
                        '''
                        angle = np.random.uniform(0, 2 * np.pi)
                        source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]

                        episode_new = _create_episode_objectnav(
                            episode_id=int(current_episode_id),
                            scene_id=str(scene_id),
                            start_position=source_position,
                            start_rotation=source_rotation,
                            shortest_paths=[],
                            radius=float(env._follower_goal_radius),
                            info=[],
                            object_category=class_name,
                        )

                        smallest_dist = env._env._sim.geodesic_distance(source_position,target_positions,episode_new)

                        ep_shortest_path = np.copy(episode_new._shortest_path_cache.points)
                        ep_shortest_path = np.array(ep_shortest_path )

                        target_viewpoint_pos = ep_shortest_path[-1].tolist()

                        ############################################################
                        '''
                        make sure that the target is valid
                        '''
                        _tmp_diff = points - target_viewpoint_pos
                        _euclid_vec = np.linalg.norm(_tmp_diff, ord=2, axis=-1)

                        _target_positions = points[np.argsort(_euclid_vec)]
                        target_viewpoint_pos = _target_positions[0]
                        ############################################################


                        tmp_diff = np.array(target_viewpoint_pos) - np.array(source_position)
                        euclid_dist = np.linalg.norm(tmp_diff, ord=2, axis=-1)

                        geo_dist=smallest_dist

                        shortest_paths =  None
                        try:
                            shortest_paths = get_action_shortest_path(
                                    env._env._sim,
                                    source_position=source_position,
                                    source_rotation=source_rotation,
                                    goal_position=target_viewpoint_pos,
                                    success_distance=float(env._follower_goal_radius),
                                    max_episode_steps=env._env._config.ENVIRONMENT.MAX_EPISODE_STEPS,
                                )
                        # Throws an error when it can't find a path
                        except GreedyFollowerError:
                            print("error greedy")
                            continue

                        if(not shortest_paths):
                            print("error shortest_paths empty")
                            continue
                        elif(len(shortest_paths)>499):
                            print("path too long")
                            continue

                        # print("path len=",len(shortest_paths))
                        shortest_paths = [shortest_paths]

                        episode_new = _create_episode_objectnav(
                            episode_id=int(current_episode_id),
                            scene_id=str(scene_id),
                            start_position=source_position,
                            start_rotation=source_rotation,
                            shortest_paths=shortest_paths,
                            radius=float(env._follower_goal_radius),
                            info={"geodesic_distance": float(geo_dist), "euclidean_distance": float(euclid_dist),
                                  "target_viewpoint_pos": target_viewpoint_pos},
                            object_category=class_name,
                        )

                        episodes.append(episode_new)
                        ##########################################
                        # points.pop(target_viewpoint_pos)
                        ##########################################
                        break
                    #end for retry
                    # print("no valid sample found")
                ##########################################

                '''
                create a json with the new episodes
                '''
                ###########################################
                if(episodes):
                    ###########################################
                    #sanity check
                    dataset = env._env._dataset
                    dataset.episodes = episodes

                    json_new = dataset.to_json()
                    parsed_json = json.loads(json_new)

                    dataset_json['episodes'] = parsed_json['episodes']
                    ###########################################
                    #inplace op
                    # np.random.shuffle(dataset_json['episodes'])
                    # dataset_json['episodes'] = list(dataset_json['episodes'])
                    ###########################################

                    n_episodes_out = len(episodes)
                    print("n_episodes_out",n_episodes_out,flush=True)
                    separation = max( int(np.floor(n_episodes_out*0.7)), 1)
                    #TRAIN
                    dataset_json_train = copy.deepcopy(dataset_json)
                    dataset_json_train['episodes'] = dataset_json_train['episodes'][:separation]

                    dst_folder = "/habitat-challenge-data/mine_objectgoal_mp3d/lv2_train_small/content/"
                    filename = dst_folder+scene_name+".json.gz"

                    ###########################################
                    '''
                    check if the file already exists, if so update it
                    '''
                    for idx in range(len(dataset_json_train['episodes'])):
                        dataset_json_train['episodes'][idx]['episode_id']= idx
                    if os.path.isfile(filename):
                        oldfile = process_file(filename)
                        pad = len(oldfile['episodes'])
                        #update every episode with the pad
                        ########################################################
                        for w in range(len(dataset_json_train['episodes'])):
                            dataset_json_train['episodes'][w]['episode_id']= str(int(dataset_json_train['episodes'][w]['episode_id'])+int(pad))
                        ########################################################
                        oldfile['episodes'] = oldfile['episodes']+dataset_json_train['episodes']
                        oldfile = json.dumps(oldfile)
                        print("updating",filename,flush=True)
                        write_file(filename,oldfile)
                    else:
                        dataset_json_train = json.dumps(dataset_json_train)
                        print("writing to",filename,flush=True)
                        write_file(filename,dataset_json_train)
                    ###########################################

                    #TEST
                    dataset_json_test = copy.deepcopy(dataset_json)
                    # dataset_json_test['episodes'] = [dataset_json_test['episodes'][1]]
                    dataset_json_test['episodes'] = dataset_json_test['episodes'][separation:]

                    dst_folder = "/habitat-challenge-data/mine_objectgoal_mp3d/lv2_test_small/content/"
                    filename = dst_folder+scene_name+".json.gz"

                    ###########################################
                    '''
                    check if the file already exists, if so update it
                    '''
                    for idx in range(len(dataset_json_test['episodes'])):
                        dataset_json_test['episodes'][idx]['episode_id']= idx
                    if os.path.isfile(filename):
                        oldfile = process_file(filename)
                        pad = len(oldfile['episodes'])
                        ########################################################
                        for w in range(len(dataset_json_test['episodes'])):
                            dataset_json_test['episodes'][w]['episode_id']= str(int(dataset_json_test['episodes'][w]['episode_id'])+int(pad))
                        ########################################################
                        oldfile['episodes'] = oldfile['episodes']+dataset_json_test['episodes']
                        oldfile = json.dumps(oldfile)
                        print("updating",filename,flush=True)
                        write_file(filename,oldfile)
                    else:
                        dataset_json_test = json.dumps(dataset_json_test)
                        print("writing to",filename,flush=True)
                        write_file(filename,dataset_json_train)
                else:
                    print("episodes empty",episodes)
            #end with env
                    ###########################################

                ###########################################
        #END for
    #END for
    print("EXITING")
    exit()

################################################################################
##main
################################################################################
def main():
    generator()

if __name__ == "__main__":
    main()
