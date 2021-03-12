#!/usr/bin/env bash

PY="/home/esetstore/blackjack/anaconda3/bin/python"
CUDA_VISIBLE_DEVICES=2 $PY eval_ofa_stereo.py --path /datasets/SceneFlow

