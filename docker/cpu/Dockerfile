FROM ubuntu:14.04
MAINTAINER Ziheng Jiang <jzhtomas@gmail.com>
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
    make -j && \
    cd python && python setup.py install

# Install MinPy from GitHub directly. Easy!
RUN git clone --recursive https://github.com/dmlc/minpy.git && \
    cd minpy && python setup.py install
