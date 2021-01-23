import os 
import config 

import util.paths
import util.detector
import util.estimator
import util.TCN
import util.visutil

from TorchSUL import sul_tool

# step1: extract frames 
util.paths.makedir('./temp/imgs/')
sul_tool.extract_frames(config.video_name, './temp/imgs/vid', skip=5)

# step 2: run human detection
util.detector.run_frames('./temp/')

# step 3: run pose detection
pts = util.estimator.run_frames('./temp/')

# step 4: run TCN
# TODO: Need pts transform
pred = util.TCN.run_points(pts)

# step 5: visualization 
util.visutil.plot_sequence(pred, config.visualization_path)
