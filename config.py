import numpy as np 

video_name = 'vid1t.mp4'

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
visualization_path = './output/'

# MMD
template_vmd_path = 'kagamine_template.vmd'
vmd_output_path = 'output.vmd'
