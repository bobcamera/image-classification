#!/bin/bash
sudo docker run -it --gpus=all --rm -v ${PWD}/notebooks:/tf/notebooks -v ${PWD}/models:/tf/models -v ${PWD}/data:/tf/data -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter
