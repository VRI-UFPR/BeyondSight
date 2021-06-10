# python beyond_agent/gym_env_semantic_map_old.py --model-path beyond_agent/checkpoints/ddppo_pointnav_habitat2020_challenge_baseline_v1.pth --input-type rgbd --evaluation local
# python beyond_agent/gym_env_semantic_map_old.py --model-path beyond_agent/checkpoints/gibson-0plus-mp3d-train-val-test-blind.pth --input-type blind --evaluation local
# python beyond_agent/gym_env_semantic_map.py --model-path beyond_agent/checkpoints/gibson-0plus-mp3d-train-val-test-blind.pth --input-type blind --evaluation local
# export CUDA_VISIBLE_DEVICES=1
# python beyond_agent/train_env.py --model-path beyond_agent/checkpoints/ddppo_pointnav_habitat2020_challenge_baseline_v1.pth --input-type rgbd --evaluation local
# python beyond_agent/train_env.py --model-path beyond_agent/checkpoints/gibson-0plus-mp3d-train-val-test-blind.pth --input-type blind --evaluation local

source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash mine_compile_yolact_plus.sh

# python beyond_agent/train_env.py --model-path beyond_agent/checkpoints/gibson-0plus-mp3d-train-val-test-blind.pth --input-type blind --evaluation local
# python beyond_agent/shorten_ep.py --model-path beyond_agent/checkpoints/ddppo_pointnav_habitat2020_challenge_baseline_v1.pth --input-type rgbd --evaluation local

# bash extras/shorten_parallel.sh

python beyond_agent/train.py --model-path beyond_agent/checkpoints/ddppo_pointnav_habitat2020_challenge_baseline_v1.pth --input-type rgbd --evaluation local


# python beyond_agent/grab_viewpoints.py --model-path beyond_agent/checkpoints/ddppo_pointnav_habitat2020_challenge_baseline_v1.pth --input-type rgbd --evaluation local

# bash extras/yolact_matterport_trainer_parallel2.sh
