#FROM objectnav_ddppo_base AS sim_image
FROM objectnav_base AS sim_image
LABEL stage=builderddppo

#two stage to avoid compilation everytime

#TODO: Remove once base image is updated.
RUN /bin/bash -c ". activate habitat; git clone http://github.com/facebookresearch/habitat-lab.git habitat-lab2; cd habitat-lab2; git fetch; git checkout 814dc04715561aeeb7f3113723b66ec53188eb41; cd .."
RUN /bin/bash -c ". activate habitat; mv habitat-api habitat-lab ; cp -r habitat-lab2/habitat_baselines habitat-lab/.; cp -r habitat-lab2/habitat/tasks habitat-lab/habitat/.;cp -r habitat-lab2/habitat/core/embodied_task.py habitat-lab/habitat/core/.; cp -r habitat-lab2/habitat/core/dataset.py habitat-lab/habitat/core/.; cp -r habitat-lab2/habitat/sims habitat-lab/habitat/."

ADD benchmark.py habitat-lab/habitat/core/benchmark.py

RUN /bin/bash -c ". activate habitat ; cd habitat-lab ; python setup.py develop build_ext --parallel 5 install; cd .." AS v0.1.5

ADD ddppo_agents.py agent.py
ADD submission.sh submission.sh
ADD configs/challenge_objectnav2020.local.rgbd.yaml /challenge_objectnav2020.local.rgbd.yaml
ADD configs/ configs/
ADD checkpoints/ddppo_objectnav_habitat2020_challenge_baseline_v1.pth demo.ckpt.pth

ENV AGENT_EVALUATION_TYPE remote

ENV TRACK_CONFIG_FILE "/challenge_objectnav2020.local.rgbd.yaml"

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash submission.sh --model-path demo.ckpt.pth --input-type rgbd"]
