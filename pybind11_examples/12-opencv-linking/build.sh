#!/bin/bash

cd $(dirname $(readlink -f $0))

rm -rf build
mkdir build
cd build

cmake ..
cmake --build .

cd -

