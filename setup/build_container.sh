#!/bin/bash
readonly SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE}"c)")
DOCKER_REGISTRY=tiffanyyk
IMAGE_NAME=${DOCKER_REGISTRY}/tiffanyyk
TAG=16.412-learning-for-planning
BASE_IMG=pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime
# BASE_IMG=nvidia/cuda:11.2.1-cudnn8-runtime-ubuntu20.04

DOCKERFILE=Dockerfile
DOCKER_USERNAME=dev
PASSWORD=dev

docker build -t $IMAGE_NAME:$TAG \
    -f $DOCKERFILE \
    --network host \
    --build-arg BASE_IMG=$BASE_IMG \
    --build-arg USERNAME=$DOCKER_USERNAME \
    --build-arg PASSWORD=$PASSWORD \
    --build-arg GROUP_ID=$(id -g ${USER}) \
    --build-arg USER_ID=$(id -u ${USER}) \
    ${SCRIPT_DIR}
