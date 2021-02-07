FROM objectnav_base AS sim_image
LABEL stage=builderBeyondPre

# RUN /bin/bash -c ". activate habitat; nvidia-smi"

#two stage to avoid compilation everytime
RUN /bin/bash -c ". activate habitat; pip install pycocotools"

#TODO: Remove once base image is updated.
RUN /bin/bash -c ". activate habitat; git clone http://github.com/facebookresearch/habitat-lab.git habitat-lab2; cd habitat-lab2; git fetch; git checkout 814dc04715561aeeb7f3113723b66ec53188eb41; cd .."
RUN /bin/bash -c ". activate habitat; mv habitat-api habitat-lab ; cp -r habitat-lab2/habitat_baselines habitat-lab/.; cp -r habitat-lab2/habitat/tasks habitat-lab/habitat/.;cp -r habitat-lab2/habitat/core/embodied_task.py habitat-lab/habitat/core/.; cp -r habitat-lab2/habitat/core/dataset.py habitat-lab/habitat/core/.; cp -r habitat-lab2/habitat/sims habitat-lab/habitat/."

ADD benchmark.py habitat-lab/habitat/core/benchmark.py

RUN /bin/bash -c ". activate habitat ; cd habitat-lab ; python setup.py develop build_ext --parallel 5 install; cd .." AS v0.1.5

################################################################################
COPY --from=nvidia/cuda:10.1-devel-ubuntu18.04 /usr/include/cublas* /usr/include/

#building the DCNv2 is super tricky during build so we will do it during docker run
#RUN /bin/bash -c ". activate habitat; cd beyond_agent/yolact/external/DCNv2; export CUDA_HOME=/usr/local/cuda-10.1; python setup.py build develop"
###############################################################################
