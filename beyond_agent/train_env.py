import torch
import numpy as np
# from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast
import argparse
import os
import numba
import random

from model import ChannelPool
# from beyond_agent import BeyondAgent
from beyond_agent_without_internal_mapper import BeyondAgentWithoutInternalMapper
from fog_of_war import reveal_fog_of_war

from baselines_config_default import get_config

from datetime import datetime

from train import PPO, RolloutStorage

from gym_env_semantic_map import VectorEnv_without_habitat
from beyond_agent import batch_obs
from collections import defaultdict, deque
import time
from tensorboard_utils import TensorboardWriter

# from train import RolloutStorage

# class PPO_without_habitat(PPO):
#     def __init__(self, config):
#         super(PPO, self).__init__(config)
#         print("EXITING")
#         exit()

class PPO_without_habitat(PPO):
    def __init__(self, config):
        self.config = config
        ##################################################################

        self.device = config.BEYOND_TRAINER.DEVICE
        print("DEVICE=",self.device)
        self.batch_size = config.BEYOND_TRAINER.BATCH_SIZE if config.BEYOND_TRAINER.TRAIN_MODE else config.BEYOND_TRAINER.EVAL_BATCH_SIZE
        print("self.batch_size",self.batch_size)

        self.train_mode = True if config.BEYOND_TRAINER.TRAIN_MODE else False
        self.rollout_size = config.BEYOND_TRAINER.ROLLOUT_SIZE
        self.rollouts = None

        self.RESUME_TRAIN = config.BEYOND_TRAINER.RESUME_TRAIN
        self.RESUME_TRAIN_CHEKPOINT = config.BEYOND_TRAINER.RESUME_TRAIN_CHEKPOINT

        self.configs = None
        self.datasets = None
        self.num_envs = None
        self.envs = None

        self.smallest_value_loss = np.inf
        self.smallest_action_loss = np.inf

        self.flush_secs = 30

        self.LOG_INTERVAL = config.BEYOND_TRAINER.LOG_INTERVAL

        self.NUM_UPDATES=config.BEYOND_TRAINER.NUM_UPDATES
        self.ppo_epoch = self.config.BEYOND_TRAINER.PPO.ppo_epoch
        self.num_mini_batch = min(self.batch_size, self.config.BEYOND_TRAINER.PPO.num_mini_batch)
        self.num_steps_done=0
        self.num_updates_done=0
        self._last_checkpoint_percent=0

        self.use_linear_lr_decay = self.config.BEYOND_TRAINER.PPO.use_linear_lr_decay
        self.use_clipped_value_loss = True
        self.use_normalized_advantage = self.config.BEYOND_TRAINER.PPO.use_normalized_advantage

        self.clip_param = self.config.BEYOND_TRAINER.PPO.clip_param
        self.value_loss_coef = self.config.BEYOND_TRAINER.PPO.value_loss_coef
        self.entropy_coef = self.config.BEYOND_TRAINER.PPO.entropy_coef
        self.max_grad_norm = self.config.BEYOND_TRAINER.PPO.max_grad_norm
        self.reward_window_size = self.config.BEYOND_TRAINER.PPO.reward_window_size


        self.NUM_CHECKPOINTS = self.config.BEYOND_TRAINER.NUM_CHECKPOINTS
        self.CHECKPOINT_INTERVAL = self.config.BEYOND_TRAINER.CHECKPOINT_INTERVAL

        ########################################################################
        '''
            Logging stuff
        '''
        now = datetime.now()

        if config.BEYOND_TRAINER.TRAIN_MODE:
            EXPERIMENT_NAME = config.BEYOND_TRAINER.EXPERIMENT_NAME
            if EXPERIMENT_NAME!="":
                self.LOG_DIR = config.BEYOND_TRAINER.LOG_DIR+EXPERIMENT_NAME+now.strftime("%d-%m-%Y_%H-%M-%S")+"/"
            else:
                self.LOG_DIR = config.BEYOND_TRAINER.LOG_DIR+now.strftime("%d-%m-%Y_%H-%M-%S")+"/"
            if not os.path.exists(self.LOG_DIR):
                os.makedirs(self.LOG_DIR)


            self.CHECKPOINT_FOLDER = config.BEYOND_TRAINER.CHECKPOINT_FOLDER
            if EXPERIMENT_NAME!="":
                self.CHECKPOINT_FOLDER = self.CHECKPOINT_FOLDER+EXPERIMENT_NAME+now.strftime("%d-%m-%Y_%H-%M-%S")+"/"
            else:
                self.CHECKPOINT_FOLDER = self.CHECKPOINT_FOLDER+now.strftime("%d-%m-%Y_%H-%M-%S")+"/"
            if not os.path.exists(self.CHECKPOINT_FOLDER):
                os.makedirs(self.CHECKPOINT_FOLDER)

        ########################################################################

        # self.agent = BeyondAgent(
        #     device=self.device,
        #     config=self.config,
        #     batch_size=self.batch_size,
        # )
        self.agent = BeyondAgentWithoutInternalMapper(
            device=self.device,
            config=self.config,
            batch_size=self.batch_size,
            is_batch_internal=True
        )
    ############################################################################
#     # def train(self):
#     #     ########################################################################
#     #     '''
#     #     load interrupted state
#     #     '''
#     #     if(self.RESUME_TRAIN):
#     #         print("RESUMING TRAINING FROM", self.RESUME_TRAIN_CHEKPOINT)
#     #         # Map location CPU is almost always better than mapping to a CUDA device.
#     #         # Models will be loaded Separate later on
#     #         ckpt_dict = self.load_checkpoint(self.RESUME_TRAIN_CHEKPOINT, map_location="cpu")
#     #
#     #         # self.agent.actor_critic.load_state_dict(ckpt_dict["state_dict"])
#     #         self.agent.actor_critic.load_actor_critic_weights(ckpt_dict["state_dict"])
#     #         # self.agent.actor_critic.load_feature_net_weights(ckpt_dict["state_dict"])
#     #         del ckpt_dict
#     #         print("",flush=True)
#     #     ########################################################################
#     #
#     #     '''
#     #     Setup logging
#     #
#     #     Load weights if resuming
#     #
#     #     Perform train_step
#     #     '''
#     #     ########################################################################
#     #     self.setup_envs()
#     #     ########################################################################
#     #
#     #     count_checkpoints = 0
#     #     prev_time = 0
#     #
#     #     ########################################################################
#     #     with (
#     #         TensorboardWriter(
#     #             self.LOG_DIR, flush_secs=self.flush_secs
#     #         )
#     #     ) as writer:
#     #         while (not self.is_done()):
#     #             self.agent.eval()
#     #             self.agent.planner.eval()
#     #             count_steps_delta = 0
#     #
#     #             '''
#     #             Then start rollout buffer loop,
#     #             '''
#     #             for i in range(self.rollout_size):
#     #                 self._compute_actions()
#     #                 count_steps_delta+=self._do_step_and_collect()
#     #             ################################################################
#     #             (
#     #                 value_loss,
#     #                 action_loss,
#     #                 dist_entropy,
#     #             ) = self._update_agent()
#     #
#     #             # if self.use_linear_lr_decay:
#     #             #     lr_scheduler.step()  # type: ignore
#     #
#     #             self.num_updates_done += 1
#     #
#     #             losses = self._coalesce_post_step(
#     #                 dict(value_loss=value_loss, action_loss=action_loss),
#     #                 count_steps_delta,
#     #             )
#     #
#     #             self._training_log(writer, losses, prev_time)
#     #
#     #             # checkpoint model
#     #             if self.should_checkpoint():
#     #                 self.save_checkpoint(
#     #                     f"ckpt.{count_checkpoints}.pth",
#     #                     dict(
#     #                         step=self.num_steps_done,
#     #                         wall_time=(time.time() - self.t_start) + prev_time,
#     #                     ),
#     #                 )
#     #                 count_checkpoints += 1
#     #     ########################################################################
#     #     self.save_checkpoint(
#     #         f"ckpt.last.pth",
#     #         dict(
#     #             step=self.num_steps_done,
#     #             wall_time=(time.time() - self.t_start) + prev_time,
#     #         ),
#     #     )
#     #     print("---------------------------------------------------------------")
#     #     print("End of train")
#     #     print("---------------------------------------------------------------")
#     #     self.envs.close()
#     #     torch.cuda.empty_cache()
#     #     return
#     ############################################################################
    ############################################################################
    def setup_envs(self):

        self.envs = VectorEnv_without_habitat(
            config=self.config,
            device=self.device,
            batch_size=self.batch_size,
            n_ep=21,
        )
        num_envs = self.batch_size
        self.num_envs = num_envs

        # print("EXITING")
        # exit()
        # self.configs, self.datasets = self._load_environment_data()
        #
        # num_envs = len(self.configs)
        # self.num_envs = num_envs
        #
        # env_fn_args = tuple(zip(self.configs, self.datasets, range(num_envs), [self.train_mode for tmp in range(num_envs)], [i for i in range(num_envs)] ))
        # ########################################################################
        #
        # multiprocessing_start_method="forkserver"
        #
        # self.envs = habitat.VectorEnv(
        #     make_env_fn=make_follower_env,
        #     env_fn_args=env_fn_args,
        #     multiprocessing_start_method=multiprocessing_start_method,
        # )
        ########################################################################
        ########################################################################

        observations = self.envs.reset()
        # print("observations=\n",observations)
        # print("EXITING")
        # exit()

        for i in range(num_envs):
            self.agent.reset(i)

        batch = batch_obs(observations, device=self.device)

        sensors = []
        for sensor_name in batch:
            sensor_dict = {'name':None,'shape':None,'dtype':None}
            sensor_dict['name'] = sensor_name
            sensor_dict['shape'] = batch[sensor_name][0].shape
            sensor_dict['dtype'] = batch[sensor_name][0].dtype
            sensors.append(sensor_dict)

        # print("sensors=\n",sensors)
        # print("batch=\n",batch)

        # print("EXITING")
        # exit()

        '''
        store inital obs in rollout
        '''
        # print("self.envs.action_spaces[0]",self.envs.action_spaces[0])

        self.rollouts = RolloutStorage(
                            numsteps=self.rollout_size,
                            num_envs=self.num_envs,
                            sensors=sensors,
                            recurrent_hidden_state_size=self.agent.actor_critic.feature_net.feature_out_size,
                            num_recurrent_layers=self.agent.actor_critic.feature_net.rnn_layers
                        )

        self.rollouts.to(self.device)

        # print("EXITING")
        # exit()

        # print("self.rollout=\n",self.rollout)

        ########################################################################
        #STEP X BATCH X FEATURES
        self.rollouts.buffers["observations"][0] = batch

        self.current_episode_reward = torch.zeros(self.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.num_envs, 1),
            reward=torch.zeros(self.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=self.reward_window_size)
        )



        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()
        # print("EXITING")
        # exit()
    ############################################################################
    def send_computed_reward(self, reward):
        self.envs.send_computed_reward(reward)

    def train(self):
        ########################################################################
        '''
        load interrupted state
        '''
        if(self.RESUME_TRAIN):
            print("RESUMING TRAINING FROM", self.RESUME_TRAIN_CHEKPOINT)
            # Map location CPU is almost always better than mapping to a CUDA device.
            # Models will be loaded Separate later on
            ckpt_dict = self.load_checkpoint(self.RESUME_TRAIN_CHEKPOINT, map_location="cpu")
            # print("ckpt_dict",ckpt_dict,flush=True)

            # self.agent.actor_critic.load_state_dict(ckpt_dict["state_dict"])
            if(self.config.BEYOND_TRAINER.RESUME_ONLY_FEATURE_NET):
                print("load_feature_net_weights")
                self.agent.actor_critic.load_feature_net_weights(ckpt_dict["state_dict"])
            else:
                print("load_actor_critic_weights")
                self.agent.actor_critic.load_actor_critic_weights(ckpt_dict["state_dict"])


            del ckpt_dict
            print("",flush=True)
        ########################################################################

        '''
        Setup logging

        Load weights if resuming

        Perform train_step
        '''
        ########################################################################
        self.setup_envs()
        ########################################################################

        count_checkpoints = 0
        prev_time = 0

        ########################################################################
        with (
            TensorboardWriter(
                self.LOG_DIR, flush_secs=self.flush_secs
            )
        ) as writer:
            while (not self.is_done()):
                self.agent.eval()
                self.agent.planner.eval()
                count_steps_delta = 0

                '''
                Then start rollout buffer loop,
                '''
                for i in range(self.rollout_size):
                    self._compute_actions()
                    count_steps_delta+=self._do_step_and_collect()
                ################################################################
                (
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent()

                # if self.use_linear_lr_decay:
                #     lr_scheduler.step()  # type: ignore

                self.num_updates_done += 1

                losses = self._coalesce_post_step(
                    dict(value_loss=value_loss, action_loss=action_loss),
                    count_steps_delta,
                )

                self._training_log(writer, losses, prev_time)

                # checkpoint model
                if self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    count_checkpoints += 1
        ########################################################################
        self.save_checkpoint(
            f"ckpt.last.pth",
            dict(
                step=self.num_steps_done,
                wall_time=(time.time() - self.t_start) + prev_time,
            ),
        )
        print("---------------------------------------------------------------")
        print("End of train")
        print("---------------------------------------------------------------")
        self.envs.close()
        torch.cuda.empty_cache()
        return
    ############################################################################
################################################################################

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
        ["configs/beyond.yaml","configs/beyond_trainer.yaml","configs/ddppo_pointnav.yaml"], ["BASE_TASK_CONFIG_PATH", config_paths]
    ).clone()

    config.defrost()
    config.TORCH_GPU_ID = 0
    config.INPUT_TYPE = args.input_type
    config.MODEL_PATH = args.model_path
    config.RANDOM_SEED = 7
    config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    config.freeze()

    device = config.BEYOND.DEVICE


    # agent = BeyondAgentWithoutInternalMapper(device=device, config=config, batch_size=1, is_batch_internal=True)
    #
    # # env = SemanticEnv(device=device, n_ep=256)
    # env = SemanticEnv(device=device, n_ep=21)
    # env.seed(config.RANDOM_SEED)
    # env.init_eps()


    print("-------------------------------------------------------------")


    trainer = PPO_without_habitat(config)
    # print("EXITING")
    # exit()

    if(config.BEYOND_TRAINER.TRAIN_MODE):
        print("TRAIN MODE")
        trainer.train()
    else:
        print("EVAL MODE: checkpoint", config.BEYOND_TRAINER.RESUME_EVAL_CHEKPOINT)
        trainer.eval_checkpoint(config.BEYOND_TRAINER.RESUME_EVAL_CHEKPOINT)

if __name__ == "__main__":
    os.environ["CHALLENGE_CONFIG_FILE"] = "configs/challenge_objectnav2021.local.rgbd.yaml"
    main()
