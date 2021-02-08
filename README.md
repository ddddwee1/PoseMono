# PoseMono

*This repo is under development and will update frequently.*

## Introduction

This is a repo of monocular 3D pose estimation. Here we provide a full pipeline of 3D human pose estimation in monocular videos. 

With this tool (maybe it's a tool), you can estimate 3D poses in any videos. 

This repo is still in progress, and will be constantly upgraded.

## Requirement

*Note: this requirement may be changed from version to version*

- TorchSUL

```
pip install torchsul 
```

- Detectron2

- Pretrained models and sample video 

Pre-trained models and sample video can be downloaded from [here](https://www.dropbox.com/sh/zu80ojnx9l1r1ei/AABIVaCT5MIw3CAIw6Pkf73ua?dl=0-)

## Instruction for MMD motion (VMD) converter 

1. Load any MMD model, turn off the IK point in the model manipulation section (モデル操作)

2. Save the empty motion data (file -> save motion data / ファイル -> モーションデータ保存)

3. Configure your template vmd path and your desired output vmd path in config.py 

4. Run the "run.py", then you can load this motion file into your MMD

*You can see the demo video along with the pre-trained model and video: mmd_vid1t.mp4*

## TODO

- [x] Finish the first version of pipeline. Estimating single person, easy scene.

- [x] Add conversion from skeleton to MMD format.

- [x] Add support for occluded poses, make the system more robust to different occlusions

- [ ] Add pose trackers for estimating multi-persons in the video 

- [ ] Add camera coordinate estimation, add support for multi-person 3D pose estimation 

- [ ] Add more modules to improve the robustness and performance, such as graph and post processing. 

## Versions

V1.0: (25-Jan-2021) Basic version of the pipeline. Support single person 3D pose detection. 

V1.1: (27-Jan-2021) Now it can produce quite accurate and smooth poses. 

V1.2 (30-Jan-2021) Add converter to MMD motion file (VMD file) 

V1.3 (31-Jan-2021) Change the training strategy, modify the network structure, for better sensitivity to high-frequency motions. (now still slight jittering)

V1.4 (2-Feb-2021) Finalize the 2D->3D network. Now the estimations are stable and can be applied to clear videos. (Will add more examples in the future version)
