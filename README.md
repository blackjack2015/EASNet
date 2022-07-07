# EASNet
EASNet: Searching Elastic and Accurate Network Architecture for Stereo Matching

PyTorch implementation of searching an efficient network architecture for stereo matching.

in ECCV 2022

## TL;DR quickstart

## Setup

Python 3 dependencies:

* PyTorch 1.8+
* matplotlib
* numpy
* imageio
* configargparse

You will also need the [LLFF code](http://github.com/fyusion/llff) (and COLMAP) set up to compute poses if you want to run on your own real data.

## What is a NeRF?

A neural radiance field is a simple fully connected network (weights are ~5MB) trained to reproduce input views of a single scene using a rendering loss. The network directly maps from spatial location and viewing direction (5D input) to color and opacity (4D output), acting as the "volume" so we can use volume rendering to differentiably render new views.

Optimizing a NeRF takes between a few hours and a day or two (depending on resolution) and only requires a single GPU. Rendering an image from an optimized NeRF takes somewhere between less than a second and ~30 seconds, again depending on resolution.


## Searching EASNet

## Finetuning on KITTI 2012/2015

## Pretrained Models

## Inference

## Citation

