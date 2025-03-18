# Use Ubuntu 24.04 as the base image
FROM ubuntu:24.04

# Set non-interactive mode for apt to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Set Conda in PATH
ENV PATH="/opt/conda/bin:$PATH"

# Copy all contents of the working directory (adjust for exceptions later if needed)
COPY . .

# Set working directory
WORKDIR .

# Create and activate the Conda environment
RUN conda env create -f environment.yml

# Activate the environment and set it as default
ENV CONDA_DEFAULT_ENV=insight_core
ENV PATH="/opt/conda/envs/insight_core/bin:$PATH"

# Set working directory to cleanrl
WORKDIR /cleanrl

# Run the Python script
CMD ["python", "train_then_distill_policy_hackatari.py"]
