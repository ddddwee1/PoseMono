from . import Siamese 
import numpy as np 
from TorchSUL import Model as M 
import config 
import torch 
import cv2 

model = Siamese.Siamese()
x = np.float32(np.random.random(size=[1,3, config.source_inp_size, config.source_inp_size]))
y = np.float32(np.random.random(size=[1,3, config.target_inp_size, config.target_inp_size]))
b = np.float32([[1,1,3,3],]*config.num_pts_tracker)
x = torch.from_numpy(x)
y = torch.from_numpy(y)
b = [torch.from_numpy(b)]
with torch.no_grad():
	outs = model(x, y, b)
saver = M.Saver(model)
saver.restore(config.tracker_path)
model.cuda()
model.eval()

def _pre_process(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = np.float32(img)
	img = img / 255 
	img = img - np.float32([0.485, 0.456, 0.406])
	img = img / np.float32([0.229, 0.224, 0.225])

	img = np.transpose(img, [2,0,1])
	return img 

def parse_ptsboxes(pts):
	res = []
	for p in pts:
		left = p - 3 
		right = p + 3 
		buf = np.concatenate([left, right], axis=1)
		buf = torch.from_numpy(buf).cuda()
		res.append(buf)
	return res 

def extract_src(imgs, pts_box):
	imgs = [cv2.resize(i, (config.source_inp_size, config.source_inp_size)) for i in imgs]
	imgs = [_pre_process(i) for i in imgs]
	imgs = torch.from_numpy(np.float32(imgs)).cuda()
	fmaps = model.extract_src(imgs, pts_box)
	return fmaps

def extract_tgt(imgs):
	imgs = [cv2.resize(i, (config.target_inp_size, config.target_inp_size)) for i in imgs]
	imgs = [_pre_process(i) for i in imgs]
	imgs = torch.from_numpy(np.float32(imgs)).cuda()
	return model.extract_tgt(imgs)

def match(kernel, tgt):
	return model.match(kernel, tgt)
