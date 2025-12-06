FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # Build tools
        build-essential \
        cmake \
        git \
        # C++ dependencies
        libeigen3-dev \
        # Python
        python3 \
        python3-pip \
        python3-dev \
        python3-pybind11 \
        pybind11-dev \
        # Utilities
        sudo \
        curl \
        ca-certificates \
        wget && \
    rm -rf /var/lib/apt/lists/*

# Create build directory
RUN mkdir -p /workspaces/CRM_ML/build

ENV SHELL=/bin/bash
ENV PYTHONPATH=/workspaces/CRM_ML
WORKDIR /workspaces/CRM_ML

# Install Python dependencies (including tensorboard for SB3 logging)
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt

CMD ["/bin/bash"]
