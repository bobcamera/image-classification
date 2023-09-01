#!/bin/bash

# Initialize a variable to hold the GPU-specific options
GPU_OPTIONS=""

# Check if "--gpu" argument is passed
if [[ "$1" == "--gpu" ]]; then
    GPU_OPTIONS="--gpus=all"
    IMAGE="tensorflow/tensorflow:latest-gpu-jupyter"
else
    IMAGE="tensorflow/tensorflow:latest-jupyter"
fi

# Run the docker command
docker run -it $GPU_OPTIONS --rm \
    -v ${PWD}/notebooks:/tf/notebooks \
    -v ${PWD}/models:/tf/models \
    -v ${PWD}/data:/tf/data \
    -p 8888:8888 \
    $IMAGE
