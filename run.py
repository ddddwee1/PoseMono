import os 
import config 
import pickle 

import util.paths
import util.detector
import util.estimator
import util.TCN
import util.visutil
import util.mmdconvert_pose 

from TorchSUL import sul_tool
import numpy as np 

# step1: extract frames 
util.paths.makedir('./temp/imgs/')
sul_tool.extract_frames(config.video_name, './temp/imgs/vid', skip=1)

# step 2: run human detection
util.detector.run_frames('./temp/')

# step 3: run pose detection
pts = util.estimator.run_frames('./temp/')
pts = pickle.load(open('temp/points.pkl','rb'))
pts = np.float32(pts)
print(pts.shape)
conf = pts[:,:,2:3]
pts = pts[:,:,:2]
pts = pts - pts[:,0:1]
conf[conf<0.7] = 0
conf[conf>=0.7] = 1
pts = pts * conf 
pts = pts / 2000

# step 4: run TCN
pred = util.TCN.run_points(pts)
print(pred.shape)
pickle.dump(pred, open('temp/pts3d.pkl','wb'))

# step 5: visualization 
pred = pickle.load(open('temp/pts3d.pkl','rb'))
util.paths.makedir(config.visualization_path)
util.visutil.plot_sequence(pred, config.visualization_path)

# step 6: convert to VMD file (MMD Motion file)
util.mmdconvert_pose.convert_vmd(config.template_vmd_path, 'temp/pts3d.pkl', config.vmd_output_path)
