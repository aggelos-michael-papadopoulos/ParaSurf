# Use NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    nano \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Set Conda environment path
ENV PATH="/opt/conda/bin:$PATH"
SHELL ["/bin/bash", "-c"]

# Set workspace directory
WORKDIR /workspace

# Clone ParaSurf repository
RUN git clone https://github.com/aggelos-michael-papadopoulos/ParaSurf.git

# Install DMS (which is inside the ParaSurf directory)
RUN cd ParaSurf/dms && make install && cd ..

# Create Conda environment and install dependencies from requirements.txt
RUN conda init bash && \
    conda create -n ParaSurf python=3.10 -y && \
    conda run -n ParaSurf pip install --no-cache-dir -r ParaSurf/requirements.txt && \
    conda run -n ParaSurf conda install -c conda-forge openbabel -y && \
    conda run -n ParaSurf pip install gdown && \
    conda clean --all -y

# Set PYTHONPATH for ParaSurf
RUN echo "export PYTHONPATH=$PYTHONPATH:/workspace/ParaSurf" >> ~/.bashrc

# Expose necessary ports (if needed)
EXPOSE 5000

# **Set ENTRYPOINT to Automatically Activate Conda Environment**
ENTRYPOINT ["/bin/bash", "-c", "source /opt/conda/bin/activate ParaSurf && exec bash"]
