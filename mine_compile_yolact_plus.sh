# sanity check
# if [ -d "/habitat-challenge-data/DCNv2" ]; then
#   echo "DCNv2 copying then running setup"
#   cp -r /habitat-challenge-data/DCNv2 /beyond_agent/yolact/external/;
#   cd beyond_agent/yolact/external/DCNv2; export CUDA_HOME=/usr/local/cuda-10.1; python setup.py build develop ; cd /
#   # cd beyond_agent/yolact/external/DCNv2; export CUDA_HOME=/usr/local/cuda-10.1; python setup.py build develop >/dev/null 2>&1; cd /
# else
#   echo "compiling DCNv2"
#   # cd beyond_agent/yolact/external/DCNv2; export CUDA_HOME=/usr/local/cuda-10.1; python setup.py build develop >/dev/null 2>&1; cp -r /beyond_agent/yolact/external/DCNv2 /habitat-challenge-data/ ; cd /
#   cd beyond_agent/yolact/external/DCNv2; export CUDA_HOME=/usr/local/cuda-10.1; python setup.py build develop ; cp -r /beyond_agent/yolact/external/DCNv2 /habitat-challenge-data/ ; cd /
# fi

# echo "compiling DCNv2"
cd beyond_agent/yolact/external/DCNv2; export CUDA_HOME=/usr/local/cuda-10.2; python setup.py build develop >/dev/null 2>&1; cd /

# echo "compiling DCNv2"
# cd beyond_agent/yolact/external/DCNv2; export CUDA_HOME=/usr/local/cuda-10.2; python setup.py build develop ; cd /

# python beyond_agent/eval.py --evaluation $AGENT_EVALUATION_TYPE $@

# echo "WILL START TRAIN"
# cd beyond_agent
# bash train_yolact.sh
