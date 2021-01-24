import os 
import config 
import pickle 

import util.paths
import util.detector
import util.estimator
import util.TCN
import util.visutil

from TorchSUL import sul_tool
import numpy as np 

# step1: extract frames 
util.paths.makedir('./temp/imgs/')
sul_tool.extract_frames(config.video_name, './temp/imgs/vid', skip=5)

# step 2: run human detection
util.detector.run_frames('./temp/')

# step 3: run pose detection
pts = util.estimator.run_frames('./temp/')
pts = pickle.load(open('temp/points.pkl','rb'))
pts = np.float32(pts) / 1000
# print(pts[0])

# step 4: run TCN
pred = util.TCN.run_points(pts)

# step 5: visualization 
util.paths.makedir(config.visualization_path)
util.visutil.plot_sequence(pred, config.visualization_path)
