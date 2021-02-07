FROM objectnav_submission_beyond_pre AS sim_image
LABEL stage=builderBeyond

ADD beyond_agent/ beyond_agent/

ADD submission.sh submission.sh
ADD configs/challenge_objectnav2020.local.rgbd.yaml /challenge_objectnav2020.local.rgbd.yaml
ADD configs/ configs/

################################################################################
ADD benchmark.py habitat-lab/habitat/core/benchmark.py
ADD beyond_agent/greedyfollowerenv.py habitat-lab/habitat/core/greedyfollowerenv.py
RUN /bin/bash -c ". activate habitat ; cd habitat-lab ; python setup.py develop build_ext --parallel 5 install; cd .." AS v0.1.5
################################################################################

ENV AGENT_EVALUATION_TYPE remote

ENV TRACK_CONFIG_FILE "/challenge_objectnav2020.local.rgbd.yaml"

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash submission.sh --model-path beyond_agent/checkpoints/ddppo_pointnav_habitat2020_challenge_baseline_v1.pth --input-type rgbd"]
