#!/bin/bash
set -e

# docker names
DOCKER_REGISTRY=tiffanyyk
IMAGE_NAME=${DOCKER_REGISTRY}/tiffanyyk
TAG=16.412-learning-for-planning
CONTAINER_NAME="412_env"

## mount paths
# local paths (in ubuntu)
WORKSPACE_LOCAL="${HOME}/Documents/courses/16.412"
DATA_LOCAL=/data

# paths in container
WORKSPACE_CONTAINER=/data/workspace
DATA_CONTAINER=/data/datasets

# resources
MEMORY_LIMIT=30g
NUM_CPUS=4
INTERACTIVE=1
VM_PORT=10025
GPU_DEVICE=0
# In the container run: `tensorboard --logdir [logs folder] --host 0.0.0.0` then go to http://localhost:[TENSORBOARD_PORT]/
TENSORBOARD_PORT=6008
# In the container run: `jupyter notebook --ip 0.0.0.0` then go to http://localhost:[JUPYTER_NOTEBOOK_PORT]/ (and paste in the token)
JUPYTER_NOTEBOOK_PORT=8888

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key=$1
  case $key in
  -i | --interactive)
    INTERACTIVE=$2
    shift
    shift
    ;;
  -m | --memory_limit)
    MEMORY_LIMIT=$2
    shift
    shift
    ;;
  -cn | --container_name)
    CONTAINER_NAME=$2
    shift
    shift
    ;;
  -im | --image)
    IMAGE_NAME=${DOCKER_REGISTRY}/$2
    shift
    shift
    ;;
  -t | --tag)
    TAG=$2
    shift
    shift
    ;;
  -nc | --cpus)
    NUM_CPUS=$2
    shift
    shift
    ;;
  -gd | --gpu_device)
    GPU_DEVICE=$2
    shift
    shift
    ;;
  -vp | --map_vm_port)
    VM_PORT=$2
    shift
    shift
    ;;
  -tbp | --tensorboard_port)
    TENSORBOARD_PORT=$2
    shift
    shift
    ;;
  esac
done

if [[ INTERACTIVE -eq 1 ]]; then
  echo "Running docker in interactive mode"
  IT=-it
else
  IT=-it  # hard code as interactive for now
fi


NV_GPU=${GPU_DEVICE} docker run --gpus '"device='"${GPU_DEVICE}"'"' --rm ${IT:-} \
--mount type=bind,source=${WORKSPACE_LOCAL},target=${WORKSPACE_CONTAINER} \
--mount type=bind,source=${DATA_LOCAL},target=${DATA_CONTAINER} \
--mount type=bind,source=/home/$USER/.ssh,target=/home/dev/.ssh \
-m ${MEMORY_LIMIT} \
-w ${WORKSPACE_CONTAINER} \
--name ${CONTAINER_NAME} \
--cpus ${NUM_CPUS} \
--shm-size ${MEMORY_LIMIT} \
-p ${VM_PORT}:22 \
-p ${TENSORBOARD_PORT}:6006 \
-p ${JUPYTER_NOTEBOOK_PORT}:8888 \
${IMAGE_NAME}:${TAG}
