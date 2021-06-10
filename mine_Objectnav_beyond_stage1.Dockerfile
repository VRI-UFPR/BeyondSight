FROM mine_objectnav_beyond_stage0
LABEL stage=builderBeyondStage1

#two stage to avoid compilation everytime

#for baseline yolact++
RUN /bin/bash -c ". activate habitat; pip install pycocotools git+git://github.com/hjweide/pyastar.git@master#egg=pyastar"

ENV CUDA_HOME /usr/local/cuda-10.2
ADD beyond_agent/yolact/external/DCNv2 DCNv2
#compile DCNv2
RUN /bin/bash -c ". activate habitat; cd DCNv2; export CUDA_HOME=/usr/local/cuda-10.2; python setup.py build develop ; cd /"
