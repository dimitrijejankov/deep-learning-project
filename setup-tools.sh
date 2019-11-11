#!/usr/bin/env bash

# this script install the manipulation tools if needed
git clone https://github.com/openai/retro.git gym-retro
cd gym-retro
pip3 install -e .

sudo apt-get install capnproto libcapnp-dev libqt5opengl5-dev qtbase5-dev
cmake . -DBUILD_UI=ON -UPYLIB_DIRECTORY
make -j$(grep -c ^processor /proc/cpuinfo)
./gym-retro-integration