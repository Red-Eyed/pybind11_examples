#!/bin/bash

cd $(dirname $(readlink -f $0))

sudo apt install --yes \
    g++ \
    cmake \
    python3-opencv \
    libopencv-dev

pip3 install numpy

