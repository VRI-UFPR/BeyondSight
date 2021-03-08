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

#beyond stuff
from beyond_agent import BeyondAgent
# from beyond_agent_matterport_only import BeyondAgentMp3dOnly
####

@numba.njit
def _seed_numba(seed: int):
    random.seed(seed)
    np.random.seed(seed)

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

    # config = get_config(
    #     "configs/ddppo_pointnav.yaml", ["BASE_TASK_CONFIG_PATH", config_paths]
    # ).clone()
    config.defrost()
    config.TORCH_GPU_ID = 0
    config.INPUT_TYPE = args.input_type
    config.MODEL_PATH = args.model_path

    config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = True

    config.RANDOM_SEED = 7
    config.freeze()

    args.evaluation = "local"

    agent = BeyondAgent(device=config.BEYOND.DEVICE, config=config, batch_size=1)
    # agent = BeyondAgentMp3dOnly(device=config.BEYOND.DEVICE, config=config, batch_size=1, is_batch_internal=True)

    if args.evaluation == "local":
        challenge = habitat.Challenge(eval_remote=False)
        challenge._env.seed(config.RANDOM_SEED)
    else:
        challenge = habitat.Challenge(eval_remote=True)

    challenge.submit(agent)


if __name__ == "__main__":
    main()
##########################################################################
