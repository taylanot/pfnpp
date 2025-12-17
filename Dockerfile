# syntax=docker/dockerfile:1
ARG CUDA_TAG=11.8.0-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive
ENV LIBTORCH_URL="https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcu118.zip"
ENV LIBTORCH_DIR=/opt/libtorch
ENV CMAKE_PREFIX_PATH=${LIBTORCH_DIR}
ENV LD_LIBRARY_PATH=${LIBTORCH_DIR}/lib:${LD_LIBRARY_PATH}

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    wget \
    unzip \
    cmake \
    git \
    python3 \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN wget -O libtorch.zip ${LIBTORCH_URL} \
 && unzip libtorch.zip -d /opt \
 && rm libtorch.zip

WORKDIR /workspace
CMD ["/bin/bash"]

