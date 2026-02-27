# Base image
FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:${PATH}"
ENV SRC_DIR /usr/local/src


# Install system packages
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        apt-utils \
        wget \
        curl \
        bzip2 \
        ca-certificates \
        openjdk-11-jre \
        git \
        libc6 \
        gnupg2 \
        lsb-release \
        vim \
        gcc \
        g++ \
        build-essential \
        libfreetype6-dev \
        libpng-dev \
        pkg-config \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# Install Miniconda with Python 3.9
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    conda clean -tipsy

# Update conda
RUN conda update -n base -c defaults conda -y

# Copy the environment.yml file
COPY environment.yaml /tmp/environment.yml

# Create conda environment
RUN conda env create -f /tmp/environment.yml

# Activate environment
ENV CONDA_DEFAULT_ENV=myenv
ENV PATH="/opt/conda/envs/myenv/bin:${PATH}"

# Install EnergyPlus 9.6.0
ENV ENERGYPLUS_INSTALL_VERSION=9-6-0
ENV ENERGYPLUS_DOWNLOAD_URL=https://github.com/NREL/EnergyPlus/releases/download/v9.6.0/EnergyPlus-9.6.0-f420c06a69-Linux-Ubuntu20.04-x86_64.sh
ENV ENERGYPLUS_DOWNLOAD_FILENAME=EnergyPlus-9.6.0-f420c06a69-Linux-Ubuntu20.04-x86_64.sh

RUN curl -SLO $ENERGYPLUS_DOWNLOAD_URL && \
    chmod +x $ENERGYPLUS_DOWNLOAD_FILENAME && \
    echo "y" | ./$ENERGYPLUS_DOWNLOAD_FILENAME && \
    rm $ENERGYPLUS_DOWNLOAD_FILENAME && \
    rm -rf /usr/local/EnergyPlus-$ENERGYPLUS_INSTALL_VERSION/{DataSets,Documentation,ExampleFiles,WeatherData,MacroDataSets,PostProcess/convertESOMTRpgm,PostProcess/EP-Compare,PreProcess/FMUParser,PreProcess/ParametricPreProcessor}

# #Define E+ version 
ENV ENERGYPLUS_INSTALL_VERSION 9-4-0
# # Download from github
ENV ENERGYPLUS_DOWNLOAD_URL https://github.com/NREL/EnergyPlus/releases/download/v9.4.0/EnergyPlus-9.4.0-998c4b761e-Linux-Ubuntu20.04-x86_64.sh
ENV ENERGYPLUS_DOWNLOAD_FILENAME  EnergyPlus-9.4.0-998c4b761e-Linux-Ubuntu20.04-x86_64.sh
RUN rm -rf /var/lib/apt/lists/* 
RUN curl -SLO $ENERGYPLUS_DOWNLOAD_URL 
RUN chmod +x $ENERGYPLUS_DOWNLOAD_FILENAME 
RUN echo "y\r" | ./$ENERGYPLUS_DOWNLOAD_FILENAME 
RUN rm $ENERGYPLUS_DOWNLOAD_FILENAME 
RUN cd /usr/local/EnergyPlus-$ENERGYPLUS_INSTALL_VERSION 
RUN rm -rf DataSets Documentation ExampleFiles WeatherData MacroDataSets PostProcess/convertESOMTRpgm 
RUN rm -rf PostProcess/EP-Compare PreProcess/FMUParser PreProcess/ParametricPreProcessor PreProcess/IDFVersionUpdater

# Install EnergyPlus 9.0.1
ENV ENERGYPLUS_VERSION_901=9-0-1
ENV ENERGYPLUS_URL_901=https://github.com/NREL/EnergyPlus/releases/download/v9.0.1/EnergyPlus-9.0.1-bb7ca4f0da-Linux-x86_64.sh
ENV ENERGYPLUS_FILENAME_901=EnergyPlus-9.0.1-bb7ca4f0da-Linux-x86_64.sh

RUN wget $ENERGYPLUS_URL_901 -O $ENERGYPLUS_FILENAME_901 && \
    chmod +x $ENERGYPLUS_FILENAME_901 && \
    echo "y" | ./$ENERGYPLUS_FILENAME_901 && \
    rm $ENERGYPLUS_FILENAME_901 && \
    rm -rf /usr/local/EnergyPlus-${ENERGYPLUS_VERSION_901}/DataSets \
           /usr/local/EnergyPlus-${ENERGYPLUS_VERSION_901}/Documentation \
           /usr/local/EnergyPlus-${ENERGYPLUS_VERSION_901}/ExampleFiles \
           /usr/local/EnergyPlus-${ENERGYPLUS_VERSION_901}/WeatherData \
           /usr/local/EnergyPlus-${ENERGYPLUS_VERSION_901}/MacroDataSets \
           /usr/local/EnergyPlus-${ENERGYPLUS_VERSION_901}/PostProcess/convertESOMTRpgm \
           /usr/local/EnergyPlus-${ENERGYPLUS_VERSION_901}/PostProcess/EP-Compare \
           /usr/local/EnergyPlus-${ENERGYPLUS_VERSION_901}/PreProcess/FMUParser \
           /usr/local/EnergyPlus-${ENERGYPLUS_VERSION_901}/PreProcess/ParametricPreProcessor \
           /usr/local/EnergyPlus-${ENERGYPLUS_VERSION_901}/PreProcess/IDFVersionUpdater

# Copy and install epw-master
#WORKDIR /
#COPY ./epw-master /app/epw-master
#WORKDIR /app/epw-master
#RUN python -m pip install . 
#setup.py install 
# Expose port for Jupyter
EXPOSE 8888

# Set the working directory
WORKDIR /

# Launch Jupyter Notebook
CMD ["conda", "run", "-n", "myenv", "jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
