from . import Siamese 
import numpy as np 
from TorchSUL import Model as M 
import config 

model = Siamese.Siamese()
x = np.float32(np.random.random(size=[1,3, config.source_inp_size, config.source_inp_size]))
y = np.float32(np.random.random(size=[1,3, config.target_inp_size, config.target_inp_size]))
b = np.float32([[1,1,3,3],]*config.num_pts)
x = torch.from_numpy(x)
y = torch.from_numpy(y)
b = [torch.from_numpy(b)]
with torch.no_grad():
	outs = model(x, y, b)
saver = M.Saver(model)
saver.restore(config.tracker_path)
