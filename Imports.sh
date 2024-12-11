#!/bin/bash

# Update the package list
sudo apt update

# Install Python3 and pip (if not already installed)
sudo apt install -y python3 python3-pip

# Install OpenCV
pip3 install opencv-python-headless

# Install NumPy
pip3 install numpy

# Install PyCryptodome (for Blowfish, get_random_bytes, and Counter)
pip3 install pycryptodome

# Install dlib (ensure required dependencies are installed first)
sudo apt install -y build-essential cmake libgtk-3-dev libboost-all-dev
pip3 install dlib

# Verify installations
pip3 show opencv-python-headless numpy pycryptodome dlib