# PoseMono

## Introduction

This is a repo of monocular 3D pose estimation. Here we provide a full pipeline of 3D human pose estimation in monocular videos. 

With this tool (maybe it's a tool), you can estimate 3D poses in any videos. 

This repo is still in progress, and will be constantly upgraded.

Let's start with a naive version and make progress step by step. 

## Requirement

*Note: this requirement may be changed from version to version*

- TorchSUL

- Detectron2

Pre-trained models and sample video can be downloaded from [here](https://www.dropbox.com/sh/zu80ojnx9l1r1ei/AABIVaCT5MIw3CAIw6Pkf73ua?dl=0-)

## TODO

-[ ] Finish the first version of pipeline. Estimating single person, easy scene.

-[ ] Add conversion from skeleton to MMD format.

-[ ] Add support for occluded poses, make the system more robust to different occlusions

-[ ] Add pose trackers for estimating multi-persons in the video 

-[ ] Add camera coordinate estimation, add support for multi-person 3D pose estimation 

## Versions

TBD

