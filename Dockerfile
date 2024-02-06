# Use the official Ubuntu 22.04 base image
FROM nvidia/cuda:12.0.0-base-ubuntu22.04

# Set the working directory in the container
WORKDIR /TumFlow

# Update package lists and install necessary packages
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    unzip \
    vim \
    nano \
    tar \
    gzip \
    bzip2 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Set the conda environment path
ENV PATH=/opt/conda/bin:$PATH

# Copy the conda environment file into the container
COPY tumflow_env.yml .

# Create a conda environment
RUN conda env create -f tumflow_env.yml

# Activate the conda environment
SHELL ["conda", "run", "-n", "tumflow", "/bin/bash", "-c"]

# Command to run your application
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "tumflow"]