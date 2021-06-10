FROM mine_objectnav_beyond_stage1
LABEL stage=builderBeyondStage2

ADD beyond_agent/ beyond_agent/
#ADD extras/ extras/

ADD submission.sh submission.sh
ADD configs/challenge_objectnav2021.local.rgbd.yaml /challenge_objectnav2021.local.rgbd.yaml
ADD configs/ configs/

#uncomment to perform training or local evaluation using train.py
#ADD mine_test_env.sh submission.sh


RUN /bin/bash -c ". activate habitat ; cd habitat-lab ; python setup.py develop build_ext --parallel 5 install; cd .."
################################################################################

ENV AGENT_EVALUATION_TYPE remote

ENV TRACK_CONFIG_FILE "/challenge_objectnav2021.local.rgbd.yaml"

ADD mine_compile_yolact_plus.sh mine_compile_yolact_plus.sh

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash submission.sh --model-path beyond_agent/checkpoints/ddppo_pointnav_habitat2020_challenge_baseline_v1.pth --input-type rgbd"]
