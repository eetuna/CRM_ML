FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# --- New: user args ---
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

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
#RUN mkdir -p /workspaces/CRM_ML/build


# --- New: create non-root user matching host UID/GID ---
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash && \
    # Optional: allow sudo inside container
    echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME}

# Create workspace/build directory and give ownership to user
RUN mkdir -p /workspaces/CRM_ML/build && \
    chown -R ${USERNAME}:${USER_GID} /workspaces


ENV SHELL=/bin/bash
ENV PYTHONPATH=/workspaces/CRM_ML
WORKDIR /workspaces/CRM_ML

# Install Python dependencies (including tensorboard for SB3 logging)
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt

# --- New: switch default user ---
USER ${USERNAME}

CMD ["/bin/bash"]
