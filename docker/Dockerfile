FROM nvidia/cuda:7.5-cudnn4-devel
MAINTAINER Ziheng Jiang <jzhtomas@gmail.com>
# Install MXNet with CUDA support.
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    apt-get update --yes && \
    apt-get install --yes software-properties-common && \
    add-apt-repository ppa:git-core/ppa && \
    apt-get update --yes && \
    apt-get install --yes python python-dev python-pip build-essential git libatlas-base-dev libopencv-dev vim curl wget unzip && \
    apt-get dist-upgrade --yes
RUN pip install --upgrade pip
RUN pip install --upgrade numpy scipy matplotlib ipython jupyter cpplint pylint
RUN mkdir -p /dmlc
WORKDIR /dmlc
RUN git clone --recursive https://github.com/dmlc/mxnet.git && \
    cd mxnet && cp make/config.mk . && \
    sed -i -e 's#^USE_CUDA =.*#USE_CUDA = 1#g' \
        -e 's#^USE_CUDA_PATH =.*#USE_CUDA_PATH = /usr/local/cuda#g' \
        -e 's#^USE_CUDNN =.*#USE_CUDNN = 1#g' config.mk && \
    make -j && \
    cd python && python setup.py install

# Install MinPy from GitHub directly. Easy!
RUN git clone --recursive https://github.com/dmlc/minpy.git && \
    cd minpy && python setup.py install
