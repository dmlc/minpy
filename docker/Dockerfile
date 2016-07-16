FROM nvidia/cuda:7.5-cudnn4-devel
MAINTAINER Ziheng Jiang <jzhtomas@gmail.com>
# following dmlc/mxnet/docker

# install the core library
RUN apt-get update && apt-get install -y build-essential git libopenblas-dev libopencv-dev wget
RUN mkdir dmlc

# install mxnet
RUN cd /dmlc && git clone --recursive https://github.com/dmlc/mxnet/ && \
    cd mxnet && cp make/config.mk . && \
    echo "USE_CUDA=1" >>config.mk && \
    echo "USE_CUDA_PATH=/usr/local/cuda" >>config.mk && \
    echo "USE_CUDNN=1" >>config.mk && \
    echo "USE_BLAS=openblas" >>config.mk && \
    make -j32 ADD_LDFLAGS=-L/usr/local/cuda/lib64/stubs
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH 

# python pakcage
RUN wget https://bootstrap.pypa.io/ez_setup.py -O - | python
RUN apt-get install -y python-numpy python-scipy unzip
RUN cd /dmlc/mxnet/python && python setup.py install

# install minpy
RUN cd /dmlc && git clone --recursive https://github.com/dmlc/minpy.git
RUN cd /dmlc/minpy && python setup.py install
