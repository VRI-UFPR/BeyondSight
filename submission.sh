#!/usr/bin/env bash

#sanity check
if [ -d "/habitat-challenge-data/DCNv2" ]; then
  echo "DCNv2 copying then running setup"
  cp -r /habitat-challenge-data/DCNv2 /beyond_agent/yolact/external/;
  cd beyond_agent/yolact/external/DCNv2; export CUDA_HOME=/usr/local/cuda-10.1; python setup.py build develop >/dev/null 2>&1; cd /
else
  echo "compiling DCNv2"
  cd beyond_agent/yolact/external/DCNv2; export CUDA_HOME=/usr/local/cuda-10.1; python setup.py build develop >/dev/null 2>&1; cp -r /beyond_agent/yolact/external/DCNv2 /habitat-challenge-data/ ; cd /
fi

# echo "compiling DCNv2"
cd beyond_agent/yolact/external/DCNv2; export CUDA_HOME=/usr/local/cuda-10.1; python setup.py build develop >/dev/null 2>&1; cd /
# cd beyond_agent/yolact/external/DCNv2; export CUDA_HOME=/usr/local/cuda-10.1; python setup.py build develop; cd /

python beyond_agent/eval.py --evaluation $AGENT_EVALUATION_TYPE $@


# REPLACEABLE_CHECK={REPLACE}
# for filename in `ls -v /beyond_agent/checkpoints/beyond_with_sseg_oracle_habitat_challenge09-12-2020_11-15-57/*.pt`; do
#     filename=$(basename -- "$filename")
#     filename="${filename%%.*}"
#     echo ${filename}
#     sed -i "s/${REPLACEABLE_CHECK}/${filename}/" configs/beyond.yaml
#     sed '15q;d' configs/beyond.yaml
#     REPLACEABLE_CHECK=${filename}
#     python beyond_agent/eval.py --evaluation $AGENT_EVALUATION_TYPE $@
# done

# python beyond_agent/train.py --config_path challenge_objectnav2020.local.rgbd.yaml --evaluation $AGENT_EVALUATION_TYPE $@

# python agent.py --evaluation $AGENT_EVALUATION_TYPE $@

#
# function processing_mine ()
# {
# #
# FILE=$(</challenge_objectnav2020.local.rgbd.yaml)$'\n'
# read -r -d '' DATASET_SUFFIX << EOM
# DATASET:
#   TYPE: ObjectNav-v1
#   SPLIT: val
#   DATA_PATH: habitat-challenge-data/val_gibson_v4_1_2m_mp3d_to_gibson/val/content/${1}.json.gz
#   SCENES_DIR: "habitat-challenge-data/data/scene_datasets/"
# EOM
# FILE+=$'\n'${DATASET_SUFFIX}
# echo "$FILE" > challenge_objectnav2020.local.rgbd.yaml
#
# # python agent.py --evaluation $AGENT_EVALUATION_TYPE $@
# # python agent.py --evaluation $AGENT_EVALUATION_TYPE --model-path demo.ckpt.pth --input-type rgbd
# # python beyond_agent/eval.py --evaluation $AGENT_EVALUATION_TYPE $@
# python beyond_agent/eval.py  --evaluation $AGENT_EVALUATION_TYPE --model-path beyond_agent/checkpoints/ddppo_pointnav_habitat2020_challenge_baseline_v1.pth --input-type rgbd
# }
#
# # processing_mine Churchton
#
# IDX=0
# processing_mine Churchton &
# res=$!
# pids[${IDX}]=$res
# IDX=$((IDX + 1))
#
# sleep 120
#
# processing_mine Emmaus &
# res=$!
# pids[${IDX}]=$res
# IDX=$((IDX + 1))
#
# sleep 120
#
# processing_mine Gravelly &
# res=$!
# pids[${IDX}]=$res
# IDX=$((IDX + 1))
#
# sleep 120
#
# processing_mine Micanopy &
# res=$!
# pids[${IDX}]=$res
# IDX=$((IDX + 1))
#
#
# len=${#pids[@]}
# echo ${len}
#
# echo "will wait ${pids[*]}"
# for pid in ${pids[*]}; do
#     # wait $pid
#     tail --pid=$pid -f /dev/null
# done
