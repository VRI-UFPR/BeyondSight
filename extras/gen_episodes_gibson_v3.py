from habitat_sim.errors import GreedyFollowerError
from habitat.tasks.nav.object_nav_task import ObjectGoalNavEpisode
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.datasets.utils import get_action_shortest_path
from habitat.core.simulator import Simulator
from typing import Optional
from habitat.core.env import RLEnv
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import argparse
import habitat
import random
import numpy as np
import os

import quaternion

from habitat.core.utils import try_cv2_import
cv2 = try_cv2_import()

import gzip
import json

def process_file(jsonfilename):
    with gzip.GzipFile(jsonfilename, 'r') as fin:    # 4. gzip
        json_bytes = fin.read()                      # 3. bytes (i.e. UTF-8)

    json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
    data = json.loads(json_str)                      # 1. data
    return data

def write_file(jsonfilename, data):
    with gzip.GzipFile(jsonfilename, 'w') as fout:
        fout.write(data.encode('utf-8'))

class GreedyFollowerEnv(habitat.RLEnv):
    def __init__(self, config, dataset=None, env_ind=0, number_retries_per_target=2048):
        super(GreedyFollowerEnv, self).__init__(config, dataset)

        '''
            Wrapper logic
        '''
        self._follower_goal_radius = self._env._config.TASK.SUCCESS.SUCCESS_DISTANCE
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

    def get_next_action(self):
        return self._follower.get_next_action(self._follower_target_viewpoint_pos)

    def reset(self):
        tmp = self._env.reset()
        # self.update_goal()
        return tmp

    ######################################################################
    def update_goal_with_start(self, i, source_position, source_rotation):
        obj = self._env._sim.semantic_annotations().objects[i]
        #divide by two because it is diameter
        sizes = obj.aabb.sizes/2.

        target_position = obj.aabb.center
        # radius = np.max(np.array(sizes*1.25))
        # radius = np.max(np.array(sizes)+1.0)
        radius = np.max(np.array(sizes)+0.36)
        # print("radius",radius)

        # if(self._env._sim._sim.pathfinder.is_navigable(target_position)):
        #     print("center is navigatable")
        # else:
        #     print("center is not navigatable")

        shortest_paths =  None

        try:
            shortest_paths = [
                get_action_shortest_path(
                    self._env._sim,
                    source_position=source_position,
                    source_rotation=source_rotation,
                    goal_position=target_position,
                    success_distance=radius,
                    max_episode_steps=self._env._config.ENVIRONMENT.MAX_EPISODE_STEPS,
                )
            ]
        # Throws an error when it can't find a path
        except GreedyFollowerError:
            print("error greedy")
            exit()
            # return
            # continue

        if(not shortest_paths):
            print("error shortest_paths empty")
            exit()
            # return
        else:
            if( (len(shortest_paths[0])==0) or (len(shortest_paths[0])>200) ):
                print("error shortest_paths == 0 or > 200",)
                dist, euclid_dist, source_position, source_rotation, target_position, shortest_paths = self.update_goal(i, radius)
                return dist, euclid_dist, source_position, source_rotation, target_position, shortest_paths, radius


        dist = self._env._sim.geodesic_distance(source_position,target_position)


        tmp_diff = target_position - source_position
        euclid_dist = np.linalg.norm(tmp_diff, ord=2, axis=-1)

        print("path",len(shortest_paths[0]), "radius",radius, "euclid_dist",euclid_dist)


        return dist, euclid_dist, source_position, source_rotation, target_position, shortest_paths, radius
    ######################################################################

    ######################################################################
    def update_goal(self, i, radius):



        if(not (i in self.computed_points)):
            '''
            object for test
            '''
            obj = self._env._sim.semantic_annotations().objects[i]
            #divide by two because it is diameter
            sizes = obj.aabb.sizes/2.

            #test all combinations at the edge of bbox
            # list_pts = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1],[-1,0,0],[0,-1,0],[0,0,-1],[-1,-1,0],[0,-1,-1],[-1,0,-1],[-1,-1,-1]]
            possible_points = [obj.aabb.center]


            # for point in list_pts:
            #     tmp_point = obj.aabb.center + sizes*point
            #     closest_pt = self._env._sim._sim.pathfinder.snap_point(tmp_point)
            #
            #     if(self._env._sim._sim.pathfinder.is_navigable(closest_pt)):
            #         possible_points.append(closest_pt)

            self.computed_points[i]=possible_points
        else:
            possible_points=self.computed_points[i]

        ##################################################################
        flag = False
        ##################################################################

        for retry in range(self.number_retries_per_target):
            source_position = self._env._sim.sample_navigable_point()

            angle = np.random.uniform(0, 2 * np.pi)
            source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]

            for point in possible_points:
                target_position = point

                shortest_paths =  None
                try:
                    shortest_paths = [
                        get_action_shortest_path(
                            self._env._sim,
                            source_position=source_position,
                            source_rotation=source_rotation,
                            goal_position=target_position,
                            success_distance=radius,
                            max_episode_steps=self._env._config.ENVIRONMENT.MAX_EPISODE_STEPS,
                        )
                    ]
                # Throws an error when it can't find a path
                except GreedyFollowerError:
                    continue

                if(not shortest_paths):
                    continue

                dist = self._env._sim.geodesic_distance(source_position,target_position)
                if ( not((dist == np.inf) or (dist < 1.0) or (dist > 25.0)) ):
                    flag=True
                else:
                    flag=False

                if(flag):
                    break
            if(flag):
                break


        tmp_diff = target_position - source_position
        euclid_dist = np.linalg.norm(tmp_diff, ord=2, axis=-1)

        return dist, euclid_dist, source_position, source_rotation, target_position, shortest_paths

def my_repeat(arr, arr_max):
    diff = arr_max - len(arr)

    if( arr_max>len(arr) ):
        aux = np.random.choice(arr,diff)
        arr = np.concatenate((arr,aux))
    else:
        arr = np.repeat(arr, np.round( arr_max/len(arr)) )

    return arr

def _create_episode(
    episode_id,
    scene_id,
    start_position,
    start_rotation,
    target_position,
    shortest_paths=None,
    radius=None,
    info=None,
) -> Optional[NavigationEpisode]:
    goals = [NavigationGoal(position=target_position, radius=radius)]
    return NavigationEpisode(
        episode_id=str(episode_id),
        goals=goals,
        scene_id=scene_id,
        start_position=start_position,
        start_rotation=start_rotation,
        shortest_paths=shortest_paths,
        info=info,
    )

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

def sample_sphere_points(center,radius,step_size):
    start = np.array(center) - radius
    end = np.array(center) + radius


    # sample points inside the desired region
    x = np.arange(start=start[0], stop=end[0]+step_size, step=step_size)
    y = np.arange(start=start[1], stop=end[1]+step_size, step=step_size)
    z = np.arange(start=start[2], stop=end[2]+step_size, step=step_size)


    n_samples = min(x.shape[0], y.shape[0], z.shape[0])

    points = np.stack([x[:n_samples], y[:n_samples], z[:n_samples]], axis=-1)


    tmp = np.sum((points - center)**2, axis=-1)
    final = points[tmp  < radius**2].reshape(-1,3)

    return final

def quaternion_from_two_vectors(v0: np.array, v1: np.array) -> np.quaternion:
    EPSILON = 1e-8
    r"""Computes the quaternion representation of v1 using v0 as the origin."""
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    c = v0.dot(v1)
    # Epsilon prevents issues at poles.
    if c < (-1 + EPSILON):
        c = max(c, -1)
        m = np.stack([v0, v1], 0)
        _, _, vh = np.linalg.svd(m, full_matrices=True)
        axis = vh.T[:, 2]
        w2 = (1 + c) * 0.5
        w = np.sqrt(w2)
        axis = axis * np.sqrt(1 - w2)
        return np.quaternion(w, *axis)

    axis = np.cross(v0, v1)
    s = np.sqrt((1 + c) * 2)
    return np.quaternion(s * 0.5, *(axis / s))

def generator():
    config = habitat.get_config(
        config_paths=os.environ["CHALLENGE_CONFIG_FILE"])

    print("############################################")
    print("inside python script",config.DATASET.DATA_PATH)
    print("############################################")

    path = config.DATASET.DATA_PATH
    folder = path.split('/')
    # folder = folder[0]+'/'+folder[1][:-1]+'1'+'/'+folder[2]
    scene_name = folder[2].split('.')[0]
    print("scene_name",scene_name)
    # folder = folder[0]+'/'+folder[1][:-1]+'4'+'/'+folder[2]
    # folder = folder[0]+'/'+folder[1][:-1]+'4'+'/'+folder[2]
    folder = path
    print("folder",folder)
    # exit()

    ###########################################
    # folder = folder+"delete"
    ###########################################

    config.defrost()
    config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = False
    config.freeze()

    dataset_json = process_file(config.DATASET.DATA_PATH)
    scene_id = dataset_json['episodes'][0]['scene_id']
    episodes = []
    episodes_idx = []
    episode_count = 0

    env = GreedyFollowerEnv(config=config)

    '''
    This is change only to garantee everthing is correct fix should not be hardcoded
    '''
    ###########################################
    '''
    Force one episode per object from dataset_json['target_objs]
    '''
    # dataset_json['target_objs]
    current_episode_id = 0
    for _class in dataset_json['target_objs']:
        fullclass=scene_name+".glb_"+_class

        '''
        find the object and its viewpoints, the viewpoints will the set of targets
        '''
        current_targets = []
        for i, object in enumerate(dataset_json['goals_by_category'][fullclass]):
            current_targets.append(dataset_json['goals_by_category'][fullclass][i]['view_points'])

        points = []
        #per objct
        for j in range(len(current_targets)):
            #per viewpoint
            for k in range(len(current_targets[j])):
                if(current_targets[j][k]):
                    points.append(current_targets[j][k]['agent_state']['position'])
        points = np.array(points)

        #1024 episodes per class
        for i in range(1024):
            '''
            start trying random points now
            '''
            for retry in range(env.number_retries_per_target):
                source_position = env._env._sim.sample_navigable_point()

                tmp_diff = points - source_position
                euclid_vec = np.linalg.norm(tmp_diff, ord=2, axis=-1)


                target_positions = points[np.argsort(euclid_vec)]

                smallest_dist = env._env._sim.geodesic_distance(source_position,target_positions)
                if(smallest_dist==np.inf):
                    continue

                '''
                Discard episodes with those criteria, the bigger the ratio the more obstacles in the path.
                '''
                ratio = smallest_dist/np.min(euclid_vec)
                # if((smallest_dist<1)or(smallest_dist>2)):
                # if((smallest_dist<2)or(smallest_dist>4)or(ratio<1.1)):
                if((smallest_dist<4)or(smallest_dist>8)or(ratio<1.1)):
                    continue

                # print("smallest_dist",smallest_dist)

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
                    radius=float(config.TASK.SUCCESS_DISTANCE),
                    info=[],
                    object_category=_class,
                )

                smallest_dist = env._env._sim.geodesic_distance(source_position,target_positions,episode_new)

                ep_shortest_path = np.copy(episode_new._shortest_path_cache.points)
                ep_shortest_path = np.array(ep_shortest_path )

                target_viewpoint_pos = ep_shortest_path[-1].tolist()

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
                            success_distance=float(config.TASK.SUCCESS_DISTANCE),
                            max_episode_steps=env._env._config.ENVIRONMENT.MAX_EPISODE_STEPS,
                        )
                # Throws an error when it can't find a path
                except GreedyFollowerError:
                    print("error greedy")
                    continue

                if(not shortest_paths):
                    print("error shortest_paths empty")
                    continue

                shortest_paths = [shortest_paths]

                episode_new = _create_episode_objectnav(
                    episode_id=int(current_episode_id),
                    scene_id=str(scene_id),
                    start_position=source_position,
                    start_rotation=source_rotation,
                    shortest_paths=shortest_paths,
                    radius=float(config.TASK.SUCCESS_DISTANCE),
                    info={"geodesic_distance": float(geo_dist), "euclidean_distance": float(euclid_dist),
                          "closest_goal_object_id": -1,
                          "target_viewpoint_pos": target_viewpoint_pos},
                    object_category=_class,
                )

                # print("episode_new",episode_new)

                episodes.append(episode_new)
                current_episode_id+=1
                if(current_episode_id%128==0):
                    print("current_episode_id",current_episode_id)
                # print("EXITING FOR DEBUG")
                # exit()
                break
                #end for
            #end if
        #end for
    #end for



    ###########################################
    if(episodes):
        #sanity check
        dataset = env._env._dataset
        dataset.episodes = episodes

        json_new = dataset.to_json()
        parsed_json = json.loads(json_new)

        # dataset_json['episodes'] = dataset_json['episodes'] + parsed_json['episodes']
        dataset_json['episodes'] = parsed_json['episodes']
        dataset_json = json.dumps(dataset_json)

        print("writing to",folder)
        write_file(folder,dataset_json)
        print("fixed")
    ###########################################
    print("END")

def main():
    generator()

if __name__ == "__main__":
    main()
