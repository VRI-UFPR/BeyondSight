#!/usr/bin/env python3

import argparse
import os
import random
from collections import OrderedDict

import numba
import numpy as np
import PIL
import torch

import habitat
from habitat import Config
from habitat_baselines.config.default import get_config

from habitat.core.utils import try_cv2_import
cv2 = try_cv2_import()
from PIL import Image

from habitat.utils.visualizations import maps

#beyond stuff
from beyond_agent import BeyondAgent
####

@numba.njit
def _seed_numba(seed: int):
    random.seed(seed)
    np.random.seed(seed)

##########################################################################
class OracleAgent(habitat.Agent):
    def __init__(self, task_config: habitat.Config):

        self.d3_269_colors_rgb : np.ndarray = np.array([[0, 0, 0], [255, 255, 0], [28, 230, 255], [255, 52, 255], [255, 74, 70], [0, 137, 65], [0, 111, 166], [163, 0, 89], [255, 219, 229], [122, 73, 0], [0, 0, 166], [99, 255, 172], [183, 151, 98], [0, 77, 67], [143, 176, 255], [153, 125, 135], [90, 0, 7], [128, 150, 147], [254, 255, 230], [27, 68, 0], [79, 198, 1], [59, 93, 255], [74, 59, 83], [255, 47, 128], [97, 97, 90], [186, 9, 0], [107, 121, 0], [0, 194, 160], [255, 170, 146], [255, 144, 201], [185, 3, 170], [209, 97, 0], [221, 239, 255], [0, 0, 53], [123, 79, 75], [161, 194, 153], [48, 0, 24], [10, 166, 216], [1, 51, 73], [0, 132, 111], [55, 33, 1], [255, 181, 0], [194, 255, 237], [160, 121, 191], [204, 7, 68], [192, 185, 178], [194, 255, 153], [0, 30, 9], [0, 72, 156], [111, 0, 98], [12, 189, 102], [238, 195, 255], [69, 109, 117], [183, 123, 104], [122, 135, 161], [120, 141, 102], [136, 85, 120], [250, 208, 159], [255, 138, 154], [209, 87, 160], [190, 196, 89], [69, 102, 72], [0, 134, 237], [136, 111, 76], [52, 54, 45], [180, 168, 189], [0, 166, 170], [69, 44, 44], [99, 99, 117], [163, 200, 201], [255, 145, 63], [147, 138, 129], [87, 83, 41], [0, 254, 207], [176, 91, 111], [140, 208, 255], [59, 151, 0], [4, 247, 87], [200, 161, 161], [30, 110, 0], [121, 0, 215], [167, 117, 0], [99, 103, 169], [160, 88, 55], [107, 0, 44], [119, 38, 0], [215, 144, 255], [155, 151, 0], [84, 158, 121], [255, 246, 159], [32, 22, 37], [114, 65, 143], [188, 35, 255], [153, 173, 192], [58, 36, 101], [146, 35, 41], [91, 69, 52], [253, 232, 220], [64, 78, 85], [0, 137, 163], [203, 126, 152], [164, 232, 4], [50, 78, 114], [106, 58, 76], [131, 171, 88], [0, 28, 30], [209, 247, 206], [0, 75, 40], [200, 208, 246], [163, 164, 137], [128, 108, 102], [34, 40, 0], [191, 86, 80], [232, 48, 0], [102, 121, 109], [218, 0, 124], [255, 26, 89], [138, 219, 180], [30, 2, 0], [91, 78, 81], [200, 149, 197], [50, 0, 51], [255, 104, 50], [102, 225, 211], [207, 205, 172], [208, 172, 148], [126, 211, 121], [1, 44, 88], [122, 123, 255], [214, 142, 1], [53, 51, 57], [120, 175, 161], [254, 178, 198], [117, 121, 124], [131, 115, 147], [148, 58, 77], [181, 244, 255], [210, 220, 213], [149, 86, 189], [106, 113, 74], [0, 19, 37], [2, 82, 95], [10, 163, 247], [233, 129, 118], [219, 213, 221], [94, 188, 209], [61, 79, 68], [126, 100, 5], [2, 104, 78], [150, 43, 117], [141, 133, 70], [150, 149, 197], [231, 115, 206], [216, 106, 120], [62, 137, 190], [202, 131, 78], [81, 138, 135], [91, 17, 60], [85, 129, 59], [231, 4, 196], [0, 0, 95], [169, 115, 153], [75, 129, 96], [89, 115, 138], [255, 93, 167], [247, 201, 191], [100, 49, 39], [81, 58, 1], [107, 148, 170], [81, 160, 88], [164, 91, 2], [29, 23, 2], [226, 0, 39], [231, 171, 99], [76, 96, 1], [156, 105, 102], [100, 84, 123], [151, 151, 158], [0, 106, 102], [57, 20, 6], [244, 215, 73], [0, 69, 210], [0, 108, 49], [221, 182, 208], [124, 101, 113], [159, 178, 164], [0, 216, 145], [21, 160, 138], [188, 101, 233], [255, 255, 254], [198, 220, 153], [32, 59, 60], [103, 17, 144], [107, 58, 100], [245, 225, 255], [255, 160, 242], [204, 170, 53], [55, 69, 39], [139, 180, 0], [121, 120, 104], [198, 0, 90], [59, 0, 10], [200, 98, 64], [41, 96, 124], [64, 35, 52], [125, 90, 68], [204, 184, 124], [184, 129, 131], [170, 81, 153], [181, 214, 195], [163, 132, 105], [159, 148, 240], [167, 69, 113], [184, 148, 166], [113, 187, 140], [0, 180, 51], [120, 158, 201], [109, 128, 186], [149, 63, 0], [94, 255, 3], [228, 255, 252], [27, 225, 119], [188, 177, 229], [118, 145, 47], [0, 49, 9], [0, 96, 205], [210, 0, 150], [137, 85, 99], [41, 32, 29], [91, 50, 19], [167, 111, 66], [137, 65, 46], [26, 58, 42], [73, 75, 90], [168, 140, 133], [244, 171, 170], [163, 243, 171], [0, 198, 200], [234, 139, 102], [149, 138, 159], [189, 201, 210], [159, 160, 100], [190, 71, 0], [101, 129, 136], [131, 164, 133], [69, 60, 35], [71, 103, 93], [58, 63, 0], [6, 18, 3], [223, 251, 113], [134, 142, 126], [152, 208, 88], [108, 143, 125], [215, 191, 194], [60, 62, 110], [216, 61, 102], [47, 93, 155], [108, 94, 70], [210, 91, 136], [91, 101, 108], [0, 181, 127], [84, 92, 70], [134, 96, 151], [54, 93, 37], [37, 47, 153], [0, 204, 255], [103, 78, 96], [252, 0, 156], [146, 137, 107]],
        dtype=np.uint8,
        )

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
        # pass

    def reset(self):
        pass

    def colorized(self, img):
        semantic_img = Image.new("P", (img.shape[1], img.shape[0]))
        semantic_img.putpalette(self.d3_269_colors_rgb[:256].flatten())
        semantic_img.putdata((img.flatten()).astype(np.uint8))
        semantic_img = semantic_img.convert("RGB")

        semantic_img = np.array(semantic_img)
        perm = np.array([2,1,0])
        semantic_img = semantic_img[:,:,perm]#BGR

        return semantic_img

    def act(self, observations):

        print(observations['top_down_map'])
        if(observations['top_down_map']):
            # map = maps.colorize_draw_agent_and_fit_to_height(observations['top_down_map'], output_height=1024)
            map = maps.colorize_draw_agent_and_fit_to_height(observations['top_down_map'], output_height=256)
            cv2.imshow('top_down_map',map)
            cv2.waitKey()

        perm = torch.LongTensor([2,1,0])
        rgb = observations['rgb'][:,:,perm].byte().cpu().numpy()

        print("sem",torch.unique(observations['semantic']))
        u_sem2 = torch.unique(observations['semantic2']).cpu().numpy()
        print("sem2", u_sem2 )
        tmp = [self._updated_dict_reverse[i] for i in u_sem2]
        print("sem2", tmp )
        print()

        sem = self.colorized(observations['semantic'].cpu().numpy())
        sem2 = self.colorized(observations['semantic2'].cpu().numpy())

        img = np.hstack( (rgb,sem,sem2) )
        cv2.imshow('img',img)
        cv2.waitKey()

        # print("action",observations['oracle'])
        return  {"action": observations['oracle']}
##########################################################################

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


    agent = BeyondAgent(device=config.BEYOND.DEVICE, config=config, batch_size=1, is_batch_internal=True)


    if args.evaluation == "local":
        challenge = habitat.Challenge(eval_remote=False)
        challenge._env.seed(config.RANDOM_SEED)
    else:
        challenge = habitat.Challenge(eval_remote=True)

    challenge.submit(agent)


if __name__ == "__main__":
    main()
##########################################################################
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--evaluation", type=str, required=True, choices=["local", "remote"]
#     )
#     args = parser.parse_args()
#
#     config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
#     config = habitat.get_config(config_paths)
#
#     agent = BeyondAgent(device=config.BEYOND.DEVICE, config=config, batch_size=1, is_batch_internal=True)
#     # agent = RandomAgent(task_config=config)
#
#     if args.evaluation == "local":
#         challenge = habitat.Challenge(eval_remote=False)
#     else:
#         challenge = habitat.Challenge(eval_remote=True)
#
#     challenge.submit(agent)
#
#
# if __name__ == "__main__":
#     main()
