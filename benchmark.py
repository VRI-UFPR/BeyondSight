#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""Implements evaluation of ``habitat.Agent`` inside ``habitat.Env``.
``habitat.Benchmark`` creates a ``habitat.Env`` which is specified through
the ``config_env`` parameter in constructor. The evaluation is task agnostic
and is implemented through metrics defined for ``habitat.EmbodiedTask``.
"""

import os
from collections import defaultdict
from typing import Dict, Optional

from habitat.config.default import get_config
from habitat.core.agent import Agent
from habitat.core.env import Env

import numpy as np

from greedyfollowerenv import GreedyFollowerEnv

class Benchmark:
    r"""Benchmark for evaluating agents in environments.
    """

    def __init__(
        self, config_paths: Optional[str] = None, eval_remote=False
    ) -> None:
        r"""..

        :param config_paths: file to be used for creating the environment
        :param eval_remote: boolean indicating whether evaluation should be run remotely or locally
        """
        config_env = get_config(config_paths)
        self.config_internal = config_env
        self._eval_remote = eval_remote

        if self._eval_remote is True:
            self._env = None
        else:
            if(self.config_internal.ENVIRONMENT.USE_GT_SSEG):
                self._env = GreedyFollowerEnv(config=config_env)
            else:
                self._env = Env(config=config_env)
            # self._env = Env(config=config_env)


    def remote_evaluate(
        self, agent: Agent, num_episodes: Optional[int] = None
    ):
        # The modules imported below are specific to habitat-challenge remote evaluation.
        # These modules are not part of the habitat-api repository.
        import evaluation_pb2
        import evaluation_pb2_grpc
        import evalai_environment_habitat
        import grpc
        import pickle
        import time

        time.sleep(60)

        def pack_for_grpc(entity):
            return pickle.dumps(entity)

        def unpack_for_grpc(entity):
            return pickle.loads(entity)

        def remote_ep_over(stub):
            res_env = unpack_for_grpc(
                stub.episode_over(evaluation_pb2.Package()).SerializedEntity
            )
            return res_env["episode_over"]

        env_address_port = os.environ.get("EVALENV_ADDPORT", "localhost:8085")
        channel = grpc.insecure_channel(env_address_port)
        stub = evaluation_pb2_grpc.EnvironmentStub(channel)

        base_num_episodes = unpack_for_grpc(
            stub.num_episodes(evaluation_pb2.Package()).SerializedEntity
        )
        num_episodes = base_num_episodes["num_episodes"]

        agg_metrics: Dict = defaultdict(float)

        count_episodes = 0

        while count_episodes < num_episodes:
            agent.reset()
            res_env = unpack_for_grpc(
                stub.reset(evaluation_pb2.Package()).SerializedEntity
            )

            while not remote_ep_over(stub):
                obs = res_env["observations"]
                action = agent.act(obs)

                res_env = unpack_for_grpc(
                    stub.act_on_environment(
                        evaluation_pb2.Package(
                            SerializedEntity=pack_for_grpc(action)
                        )
                    ).SerializedEntity
                )

            metrics = unpack_for_grpc(
                stub.get_metrics(
                    evaluation_pb2.Package(
                        SerializedEntity=pack_for_grpc(action)
                    )
                ).SerializedEntity
            )

            for m, v in metrics["metrics"].items():
                agg_metrics[m] += v
            count_episodes += 1

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        stub.evalai_update_submission(evaluation_pb2.Package())

        return avg_metrics

    # def _convert_list_to_means(infos):
    #     batch = {'distance_to_goal':[],'success':[],'softspl':[],'spl':[],'collisions':[],'step':[]}
    #
    #     for i in infos:
    #         for key in batch:
    #             batch[key].append(i[key])
    #
    #     for key in batch:
    #         batch[key]=np.mean(np.array(batch[key]))
    #
    #     return batch

    def local_evaluate(self, agent: Agent, num_episodes: Optional[int] = None):
        if num_episodes is None:
            num_episodes = len(self._env.episodes)
        else:
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(
                    num_episodes, len(self._env.episodes)
                )
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        # num_episodes = 1
        # num_episodes = 4
        agg_metrics: Dict = defaultdict(float)
        infos = []

        count_episodes = 0
        # steps = max(int(num_episodes/100),1)
        steps = max(int(num_episodes/100),32)
        print("steps",steps)

        # scene_name = self.config_internal.DATASET.DATA_PATH.split("/")[-1]
        # scene_name = scene_name.split(".")[0]
        scene_name = self.config_internal.DATASET.SPLIT
        print(scene_name)
        print(self.config_internal.DATASET)
        print("-----------------------------------------")

        while count_episodes < num_episodes:
            agent.reset()
            observations = self._env.reset()

            while not self._env.episode_over:
                if(self.config_internal.ENVIRONMENT.USE_GT_SSEG):
                    # observations['sseg_dict'] = self._env.get_sseg_dict()
                    action = agent.act(observations)
                    observations = self._env.step_obs_only(action)
                else:
                    action = agent.act(observations)
                    observations = self._env.step(action)

            if(self.config_internal.ENVIRONMENT.USE_GT_SSEG):
                metrics = self._env.get_metrics_extended()
            else:
                metrics = self._env.get_metrics()

            if(not infos):
                infos = [metrics]
            else:
                infos.append(metrics)




            if(count_episodes%steps==0):
                print("count_episodes",count_episodes)
                ###########################################################
                # batch = {'distance_to_goal':[],'success':[],'softspl':[],'spl':[],'collisions':[],'step':[]}
                batch = {'distance_to_goal':[],'success':[],'softspl':[],'spl':[],'collisions':[]}
                for i in infos:
                    for key in batch:
                        if(key=='collisions'):
                            batch[key].append(i[key]['count'])
                        else:
                            batch[key].append(i[key])

                for key in batch:
                    batch[key]=np.mean(np.array(batch[key]))
                ###########################################################
                # means = _convert_list_to_means(infos)
                means = batch
                print(means)
                # avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}
                # print("avg_metrics",avg_metrics)
                print("saving", "habitat-challenge-data/"+scene_name+"_tmp.npy")
                # np.save("habitat-challenge-data/"+scene_name+".npy", agg_metrics)
                np.save("habitat-challenge-data/"+scene_name+"_tmp.npy", infos)

            count_episodes += 1

        # agg_metrics
        ###########################################################
        # batch = {'distance_to_goal':[],'success':[],'softspl':[],'spl':[],'collisions':[],'step':[]}
        batch = {'distance_to_goal':[],'success':[],'softspl':[],'spl':[],'collisions':[]}
        for i in infos:
            for key in batch:
                if(key=='collisions'):
                    batch[key].append(i[key]['count'])
                else:
                    batch[key].append(i[key])

        for key in batch:
            batch[key]=np.mean(np.array(batch[key]))
        ###########################################################
        # means = _convert_list_to_means(infos)
        means = batch
        print(means)
        print("saving", "habitat-challenge-data/"+scene_name+".npy")
        # np.save("habitat-challenge-data/"+scene_name+".npy", agg_metrics)
        np.save("habitat-challenge-data/"+scene_name+".npy", infos)
        # avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        return means



    def evaluate(
        self, agent: Agent, num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        r"""..

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the
            evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """

        if self._eval_remote is True:
            return self.remote_evaluate(agent, num_episodes)
        else:
            return self.local_evaluate(agent, num_episodes)
