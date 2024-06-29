# This docker is based on:
# 1. https://github.com/dusty-nv/jetson-containers/blob/master/Dockerfile.ml for building OpenCV
# 2. https://github.com/microsoft/onnxruntime/blob/master/dockerfiles/Dockerfile.jetson for building onnxruntime

# Follow the below instructions before and after building this docker image:
# https://github.com/microsoft/onnxruntime/tree/master/dockerfiles#nvidia-jetson-tx1tx2nanoxavier

# syntax=docker/dockerfile:experimental
#
# This Dockerfile just installs pre-built ONNX Runtime wheel inside the image.
# Please make sure you have nvidia-runtime enabled in docker config and then build like:
#
# sudo -H DOCKER_BUILDKIT=1 nvidia-docker build --build-arg WHEEL_FILE=<path> -f Dockerfile.jetson
#

ARG BASE_IMAGE=nvcr.io/nvidia/l4t-base:r32.4.3
FROM ${BASE_IMAGE} as onnxruntime

ARG WHEEL_FILE
RUN echo "Building ONNX Runtime Docker image with ${WHEEL_FILE}..."

#
# setup environment
#
# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV LLVM_CONFIG="/usr/bin/llvm-config-9"
ARG MAKEFLAGS=-j

RUN printenv

#
# apt packages
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip \
		  python3-dev \
      libpython3.6-dev \
      python3-matplotlib \
		  build-essential \
      software-properties-common \
      unattended-upgrades \
		  gfortran \
		  git \
		  cmake \
		  curl \
		  libopenblas-dev \
		  liblapack-dev \
		  libblas-dev \
		  libhdf5-serial-dev \
		  hdf5-tools \
		  libhdf5-dev \
		  zlib1g-dev \
		  zip \
		  libjpeg8-dev \
		  libopenmpi2 \
          openmpi-bin \
          openmpi-common \
		  protobuf-compiler \
          libprotoc-dev \
		llvm-9 \
          llvm-9-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean


RUN unattended-upgrade
RUN pip3 install --upgrade pip
RUN pip3 install setuptools
RUN pip3 install wheel pybind11 pytest

WORKDIR /onnxruntime

# copy previously built wheel into the container
COPY ${WHEEL_FILE} .

RUN basename ${WHEEL_FILE} | xargs pip3 install 


#
# OpenCV
#
ARG L4T_APT_KEY
ARG L4T_APT_SOURCE

COPY jetson-ota-public.asc /etc/apt/trusted.gpg.d/jetson-ota-public.asc

RUN echo "$L4T_APT_SOURCE" > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list && \
    cat /etc/apt/sources.list.d/nvidia-l4t-apt-source.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
            libopencv-dev \
		  libopencv-python \
    && rm /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && rm -rf /var/lib/apt/lists/*

# downgrade numpy to resolve this issue https://github.com/numpy/numpy/issues/18131 and install scipy as it is needed for inference
RUN pip3 install numpy==1.19.4 scipy