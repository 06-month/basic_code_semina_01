FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

RUN apt-get update
RUN apt-get install -y wget curl unzip sudo vim libgtk2.0-dev

RUN pip install --upgrade pip
RUN pip install ipdb wandb opencv-python gdown
RUN apt-get install -y libgl1-mesa-glx
