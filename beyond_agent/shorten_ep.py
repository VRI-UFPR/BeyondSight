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
# import time

#yolact stuff
import torch
from yolact.utils.functions import SavePath
from yolact.data import cfg, set_cfg
from yolact.yolact import Yolact
from yolact.utils.augmentations import BaseTransform, FastBaseTransform_rgb_2_rgb, Resize
from yolact.layers.output_utils import postprocess, undo_image_transformation
####


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
    def __init__(self, config, dataset=None, env_ind=0, json_file=None):
        super(GreedyFollowerEnv, self).__init__(config, dataset)

        self.config = config
        self.device = self.config.BEYOND.DEVICE
        self._data = process_file(json_file)
        '''
            Wrapper logic
        '''

        self._follower_goal_radius = 0.1
        self._follower_target_viewpoint_pos = None

        self._follower = ShortestPathFollower(self._env._sim, goal_radius=self._follower_goal_radius, return_one_hot=False)

        self.iteration = 0
        self.first_obs = 0
        self.batch_size = 1

        ########################################################################
        # print("using gt sseg instead of YOLACT++")
        ########################################################################
        args_config = "yolact_plus_resnet50_mp3d_2021_config"
        set_cfg(args_config)

        # print('Loading model...', end='')
        self.sseg = Yolact()
        if(config.BEYOND.SSEG.LOAD_CHECKPOINT):
            print(config.BEYOND.SSEG.CHECKPOINT, end='')
            self.sseg.load_weights(config.BEYOND.SSEG.CHECKPOINT)
        else:
            # pass
            print("using random values for the SSEG")
        #freeze weights
        self.sseg.eval()
        self.sseg.to(self.device)
        ########################################################################

    def evalimage(self,frame,target):
        # print("frame.shape",frame.shape)
        #preprocess
        frame_transformed = FastBaseTransform_rgb_2_rgb()(frame.float())
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

            if target.to("cpu") in unique_classes.to("cpu"):
                self.first_obs = self.iteration

            return None

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

        return y_pred_scores
    ############################################################################

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
        tmp['first_obs']=self.first_obs
        return tmp

    def step(self, *args, **kwargs):
        self.iteration+= 1
        observations = self._env.step(*args, **kwargs)

        #ignore first_obs from earlier steps
        if self.iteration >= len(self._env._current_episode.shortest_paths[0])-self.max_step:
            sem_pred = self.evalimage(observations['rgb'].unsqueeze(0),torch.from_numpy(observations['objectgoal']) )

        if(self.iteration < len(self._env._current_episode.shortest_paths[0]) ):
            if self._env._current_episode.shortest_paths[0][self.iteration].action:
                observations['oracle'] = int(self._env._current_episode.shortest_paths[0][self.iteration].action)
            else:
                observations['oracle'] = 0
        else:
            observations['oracle'] = 0

        #
        metrics= self._env.get_metrics()

        # if not self._env.episode_over:
        #     if self._env._current_episode.shortest_paths[0][self.iteration].action:
        #         observations['oracle'] = int(self._env._current_episode.shortest_paths[0][self.iteration].action)
        #     else:
        #         observations['oracle'] = 0
        # else:
        #     observations['oracle'] = 0

        # if( (observations['oracle'] ==0) and (tmp['distance_to_goal']>0.1) ):
        #     action = self._follower.get_next_action(self.shortest_path[-1])
        #     observations['oracle'] = action

        # action = self._follower.get_next_action(self.shortest_path[-1])
        # observations['oracle'] = action

        # max_step = 10
        max_step = self.max_step
        # print("self._env._current_episode.shortest_paths[0]",len(self._env._current_episode.shortest_paths[0]),"self.iteration",self.iteration)
        if( (len(self._env._current_episode.shortest_paths[0])>max_step)and(self.iteration == len(self._env._current_episode.shortest_paths[0])-max_step) ):
            # agent_state = self._env._sim.get_agent_state()
            # print("id",self._env._current_episode.episode_id,"agent_state",agent_state,flush=True)

            # agent_state.position
            # agent_state.rotation

            # self._data['episodes'][int(self._env._current_episode.episode_id)]['start_position'] = agent_state.position.tolist()
            # self._data['episodes'][int(self._env._current_episode.episode_id)]['start_rotation'] = np.array( quaternion.as_float_array(agent_state.rotation) ).tolist()
            # print("self.iteration",self.iteration,"len",len(self._data['episodes'][int(self._env._current_episode.episode_id)]['shortest_paths'][0]))

            # if()

            self._data['episodes'][int(self._env._current_episode.episode_id)]['start_position'] = self._data['episodes'][int(self._env._current_episode.episode_id)]['shortest_paths'][0][self.iteration]['position']
            self._data['episodes'][int(self._env._current_episode.episode_id)]['start_rotation'] = self._data['episodes'][int(self._env._current_episode.episode_id)]['shortest_paths'][0][self.iteration]['rotation']
            self._data['episodes'][int(self._env._current_episode.episode_id)]['shortest_paths'] = (np.array(self._data['episodes'][int(self._env._current_episode.episode_id)]['shortest_paths'])[:,-max_step:]).tolist()
            self._data['episodes'][int(self._env._current_episode.episode_id)]['info']['geodesic_distance'] = metrics['distance_to_goal']
            self._data['episodes'][int(self._env._current_episode.episode_id)]['info']['euclidean_distance'] = metrics['distance_to_goal']

            # print("self._env._current_episode",self._env._current_episode,flush=True)


        return observations

    def reset(self):
        self.iteration = 0
        self.first_obs = 0
        # self.max_step = 30+np.random.randint(low=0,high=5)
        # self.max_step = 35
        self.max_step = 10
        tmp = self._env.reset()
        tmp['oracle'] = int(self._env._current_episode.shortest_paths[0][self.iteration].action)

        self.shortest_path = np.copy(self._env._current_episode._shortest_path_cache.points)
        self.shortest_path = np.array(self.shortest_path )
        # self.update_goal()
        return tmp
################################################################################

##########################################################################
class OracleAgent(habitat.Agent):
    def __init__(self):
        pass
    def act(self, observations):
        # return  {"action": observations['oracle']}
        return  observations['oracle']

############################################################################
# def load_checkpoint(checkpoint_path: str, *args, **kwargs) -> Dict:
#     r"""Load checkpoint of specified path as a dict.
#     Args:
#         checkpoint_path: path of target checkpoint
#         *args: additional positional args
#         **kwargs: additional keyword args
#     Returns:
#         dict containing checkpoint info
#     """
#     return torch.load(checkpoint_path, *args, **kwargs)

############################################################################
def eval_checkpoint(
    config,
    checkpoint_path: str,
    num_episodes = None,
    # writer: TensorboardWriter,
    # checkpoint_index: int = 0,
) -> None:
    r"""Evaluates a single checkpoint.
    Args:
        checkpoint_path: path of checkpoint
        writer: tensorboard writer object for logging to tensorboard
        checkpoint_index: index of cur checkpoint for logging
    Returns:
        None
    """

    # Map location CPU is almost always better than mapping to a CUDA device.
    # Models will be loaded Separate later on
    # ckpt_dict = load_checkpoint(checkpoint_path, map_location="cpu")
    # print("ckpt_dict",ckpt_dict)

    #use the config saved
    # self.config = ckpt_dict["config"].clone()

    config.defrost()
    config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = True
    config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False

    # config.DATASET.SPLIT= config.BEYOND_TRAINER.ENVIRONMENT_LIST.TRAIN.NAMES[0]
    # config.DATASET.DATA_PATH= config.BEYOND_TRAINER.ENVIRONMENT_LIST.TRAIN.PATH+"/{split}/{split}.json.gz"


    config.DATASET.SPLIT= config.BEYOND_TRAINER.ENVIRONMENT_LIST.TRAIN.NAMES[0]
    config.DATASET.DATA_PATH= "/habitat-challenge-data/mine_objectgoal_mp3d/single_scene_all_classes_train4_balanced/content/ac26ZMwG7aT_"+config.BEYOND.SPLIT+".json.gz"

    config.ENVIRONMENT.MAX_EPISODE_STEPS = 500
    # self.config.TASK.MEASUREMENTS.append('TOP_DOWN_MAP')

    config.freeze()

    print(config.DATASET.SPLIT, config.DATASET.DATA_PATH, config.DATASET)

    # self.agent.actor_critic.load_state_dict(ckpt_dict["state_dict"])
    # self.agent.eval()
    # self.agent.planner.eval()

    agent = OracleAgent()
    ########################################################################
    _env = GreedyFollowerEnv(config=config,json_file="/habitat-challenge-data/mine_objectgoal_mp3d/single_scene_all_classes_train4_balanced/content/ac26ZMwG7aT_"+config.BEYOND.SPLIT+".json.gz")

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
    print("steps",steps)
    valid_episodes = []
    ########################################################################
    while count_episodes < num_episodes:
        # agent.reset()
        observations = _env.reset()

        while not _env.episode_over:
            action = agent.act(observations)
            observations = _env.step(action)

        metrics = _env.get_metrics_extended()
        if ( (metrics['success']==1.0) and (metrics['first_obs']>0) ):
            valid_episodes.append(_env._data['episodes'][count_episodes])
        # print("metrics=\n",metrics,"\n\n")

        if(not infos):
            infos = [metrics]
        else:
            infos.append(metrics)

        ####################################################################
        if(count_episodes%steps==0):
            print("count_episodes",count_episodes)
            ################################################################
            # batch = {'distance_to_goal':[],'success':[],'softspl':[],'spl':[]}
            batch = {'distance_to_goal':[],'success':[],'softspl':[],'spl':[], 'step':[], 'first_obs':[]}
            for i in infos:
                for key in batch:
                    if(key=='collisions'):
                        batch[key].append(i[key]['count'])
                    else:
                        batch[key].append(i[key])

            for key in batch:
                batch[key]=np.mean(np.array(batch[key]))
            ################################################################
            means = batch
            print(means, flush=True)
        ####################################################################
        count_episodes += 1
    ########################################################################
    # batch = {'distance_to_goal':[],'success':[],'softspl':[],'spl':[]}
    batch = {'distance_to_goal':[],'success':[],'softspl':[],'spl':[], 'step':[], 'first_obs':[]}

    for i in infos:
        for key in batch:
            if(key=='collisions'):
                batch[key].append(i[key]['count'])
            else:
                batch[key].append(i[key])

    for key in batch:
        batch[key]=np.mean(np.array(batch[key]))

    means = batch
    print(means, flush=True)
    ########################################################################
    # _env.close()
    print("writing")
    _env._data['episodes']=valid_episodes
    for i in range(len(_env._data['episodes'])):
        _env._data['episodes'][i]['episode_id']=i
    newfile = json.dumps(_env._data)
    write_file("/habitat-challenge-data/mine_objectgoal_mp3d/single_scene_all_classes_train4_balanced/content/ac26ZMwG7aT_3_"+config.BEYOND.SPLIT+".json.gz",newfile)

    print("EXIT SUCCESS")

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

    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    # parser.add_argument("--model-path", default="", type=str)
    args = parser.parse_args()

    # config = get_config(
    #     ["configs/beyond.yaml","configs/beyond_trainer.yaml","configs/ddppo_pointnav.yaml"], ["BASE_TASK_CONFIG_PATH", config_paths]
    # ).clone()

    config = get_config(
        ["configs/beyond.yaml","configs/beyond_trainer.yaml","configs/ddppo_pointnav.yaml", config_paths],
    ).clone()

    # config_ori = copy.deepcopy(config)
    # config = config.TASK_CONFIG

    # config.defrost()
    #
    # config.TORCH_GPU_ID = 0
    # config.INPUT_TYPE = args.input_type
    # config.MODEL_PATH = args.model_path
    # config.RANDOM_SEED = 7
    # config.TENSORBOARD_DIR = "habitat-challenge-data/log/"
    #
    # # config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)
    # config.freeze()

    config.defrost()
    config.TORCH_GPU_ID = 0
    # config.INPUT_TYPE = args.input_type
    # config.MODEL_PATH = args.model_path
    config.RANDOM_SEED = 7
    config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    config.BEYOND.SPLIT = str(args.split)
    config.freeze()

    print("-------------------------------------------------------------")
    # print(config)
    # print("-------------------------------------------------------------")

    eval_checkpoint(config, config.BEYOND_TRAINER.RESUME_EVAL_CHEKPOINT)

    # trainer = PPO(config)
    #
    # if(config.BEYOND_TRAINER.TRAIN_MODE):
    #     print("TRAIN MODE")
    #     trainer.train()
    # else:
    #     print("EVAL MODE: checkpoint", config.BEYOND_TRAINER.RESUME_EVAL_CHEKPOINT)
    #     trainer.eval_checkpoint(config.BEYOND_TRAINER.RESUME_EVAL_CHEKPOINT)
    # # print("EXITING DEBUG")
    # # exit()

    #everything went fine so exit with 0
    return 0

if __name__ == "__main__":
    main()
##########################################################################
