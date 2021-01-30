from . import networktcn 
import torch 
import numpy as np 
import torch.nn.functional as F 
from TorchSUL import Model as M 

seq_len = 129
nettcn = networktcn.modelBundle(17, seq_len)
x_dumb = torch.zeros(2,seq_len, 17*2)
nettcn(x_dumb)
M.Saver(nettcn).restore('./models/model_tcn/')
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

def pre_process_pts2d(pts):
	pts = np.float32(pts)
	print(pts.shape)
	conf = pts[:,:,2:3]
	pts = pts[:,:,:2]
	pts = pts - pts[:,0:1]
	conf[conf<0.7] = 0
	conf[conf>=0.7] = 1
	pts = pts * conf 
	pts = pts / 2000
	return pts 
