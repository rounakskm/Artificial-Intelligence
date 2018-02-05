# Artificial-Intelligence

The objective is to build an AI for playing games

We will be using:-
 
Deep-Q-Learning 				- for building AI simulation of the self driving car 
Deep-Convolutional-Q-Learning	- for building AI that can play DOOM
A3C								- for buliding AI that plays Breakout


Installing pytorch:

    For macOS or linux:-
    pip install -c pytorch

OR

    cd pytorch
    python build setup.py
    python install setup.py

Installing OpenAI Gym:

    git clone https://github.com/openai/gym
    cd gym
    pip install -e . # minimal install

Installing other dependencies on Ubuntu:

    apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

Isntalling ppaquette:
   
    pip install ppaquette-gym-doom

Install ffmpeg to get videos of AI playing the game:
 
    conda install -c conda-forge ffmpeg=3.2.4


For common debugging solutions refer to the debug doc. The file has been taken from the Artificail-Intelligence A-Z course on Udemy.

