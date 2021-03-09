import numpy as np 

video_name = 'vid1t.mp4'

#### Normal pose estimator ####
# headnet 
head_layernum = 1
head_chn = 32

# upsmaple 
upsample_layers = 1
upsample_chn = 32

#### heatmap-guided network ####
# headnet 
head_layernum_guided = 1
head_chn_guided = 32

# upsmaple 
upsample_layers_guided = 1
upsample_chn_guided = 32

#### Tracker ####
source_inp_size = 128
source_out_size = 64
target_inp_size = 256
target_out_size = 128

#### sizes ####
# estimator 
inp_size = 384 
out_size = 192
base_sigma = 3.5
num_pts = 17
# guided-estimator 
inp_size_guided = 384 
out_size_guided = 192
# tracker 
num_pts_tracker = 12

# checkpoint
estimator_path = './models/model_estimator/'
guided_estimator_path = './models/model_estimator_guided/'
tracker_path = './models/model_tracker/'
TCN_path = './models/model_tcn/'
visualization_path = './output/'

# MMD
template_vmd_path = 'kagamine_template.vmd'
vmd_output_path = 'output.vmd'
