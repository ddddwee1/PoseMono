import numpy as np 

# headnet 
head_layernum = 1
head_chn = 32

# upsmaple 
upsample_layers = 1
upsample_chn = 32

# size
inp_size = 384 
out_size = 192
base_sigma = 3.5
num_pts = 17

# checkpoint
estimator_path = './models/model_estimator/'
video_name = 'vid1t.mp4'
visualization_path = './output/'
