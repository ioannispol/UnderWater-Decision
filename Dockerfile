FROM python:3.10-slim-bullseye

# Install miniconda
ENV CONDA_DIR /opt/miniconda/

RUN apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y wget  \
   # apt-get install -y libgl1-mesa-glx \
    git -y && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/miniconda

# Add conda to path
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda create -n uw-decision python=3.10

RUN rm -rf /workspace/*
WORKDIR /workspace/uw-decision

COPY requirements.txt requirements.txt
RUN echo "source activate uw-decision" >~/.bashrc && \
    /opt/miniconda/envs/uw-decision/bin/pip install -r requirements.txt

COPY . .

CMD [ "python3", "-m" , "UnderWater-Decision" ]
