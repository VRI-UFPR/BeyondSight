#!/usr/bin/env bash

source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash mine_compile_yolact_plus.sh
# python agent.py --evaluation $AGENT_EVALUATION_TYPE $@
# cd beyond_agent
python beyond_agent/eval.py --evaluation $AGENT_EVALUATION_TYPE $@
