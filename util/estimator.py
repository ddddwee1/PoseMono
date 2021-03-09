import os 
import cv2 
import glob 
import config
import pickle 
import numpy as np  
from . import network 
from tqdm import tqdm
from TorchSUL import Model as M 
import torch 

model_dnet = network.get_network()
M.Saver(model_dnet).restore(config.estimator_path)
model_dnet.eval()
model_dnet.cuda()

model_guided = network.get_guided_network()
M.Saver(model_guided).restore(config.guided_estimator_path)
model_guided.eval()
model_guided.cuda()

def _pre_process(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = np.float32(img)
	img = img / 255 
	img = img - np.float32([0.485, 0.456, 0.406])
	img = img / np.float32([0.229, 0.224, 0.225])
	img = np.transpose(img, [2,0,1])
	return img 

def get_pts(img):
	img = _pre_process(img)
	img = torch.from_numpy(img[None,...]).cuda()
	with torch.no_grad():
		hmap = model_dnet(img).cpu().numpy()
	res = np.zeros([hmap.shape[1], 3], dtype=np.float32)
	for i in range(hmap.shape[1]):
		m = hmap[0, i].reshape([-1])
		idx = np.argmax(m)
		scr = np.max(m)
		xx = idx % config.out_size
		yy = idx // config.out_size
		res[i,0] = xx 
		res[i,1] = yy
		res[i,2] = scr 
	return res 

def remap_to_origin(pts, bbox):
	pts = pts.copy()
	wh = bbox[2] - bbox[0]
	corner_x = bbox[0]
	corner_y = bbox[1]
	pts[:,:2] *= wh / config.out_size 
	pts[:,0] += corner_x
	pts[:,1] += corner_y
	return pts 

def run_frames(path):
	print('Estimating 2D pose:',path)
	results = []
	paths = glob.glob(os.path.join(path, 'cropped/*'))
	paths = sorted(paths)
	for p in tqdm(paths):
		imgs = glob.glob(os.path.join(p, '*.png'))
		img = cv2.imread(imgs[0])
		img = cv2.resize(img, (config.inp_size, config.inp_size))
		pts = get_pts(img)

		bbox = p.replace('cropped', 'bboxes') + '.pkl'
		bbox = pickle.load(open(bbox, 'rb'))[0]
		pts = remap_to_origin(pts, bbox)
		results.append(pts)
	pickle.dump(results, open(os.path.join(path, 'points.pkl'), 'wb'))
	return results
