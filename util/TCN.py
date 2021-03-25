from . import networktcn 
import torch 
import numpy as np 
import torch.nn.functional as F 
from TorchSUL import Model as M 
import config 

seq_len = 129
nettcn = networktcn.NetBundle(17, seq_len)
x_dumb = torch.zeros(2,seq_len, 17*2)
nettcn(x_dumb)
M.Saver(nettcn).restore(config.TCN_path)
nettcn.eval()
nettcn.cuda()

def run_points(pts):
	pts = np.float32(pts)[:,:,:2]
	pts = torch.from_numpy(pts).cuda()
	pts = pts.unsqueeze(0).unsqueeze(0)
	pts = F.pad(pts, (0,0,0,0,seq_len//2, seq_len//2), mode='replicate')
	pts = pts.squeeze()
	print('TCN padded:', pts.shape)
	with torch.no_grad():
		pred = nettcn.evaluate(pts)
	pred = pred.cpu().numpy()
	return pred
