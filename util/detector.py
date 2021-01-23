from detectron2.data.detection_utils import read_image
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
import numpy as np 
import cv2 
import os 
import glob 
from tqdm import tqdm 
import pickle 

class Args():
	def __init__(self):
		self.config_file = 'util/detector_configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'
		self.opts = ['MODEL.WEIGHTS', './models/model_detector/model_final_68b088.pkl']
		self.confidence_threshold = 0.5 

def setup_cfg(args):
	# load config from file and command-line arguments
	cfg = get_cfg()
	cfg.merge_from_file(args.config_file)
	cfg.merge_from_list(args.opts)
	# Set score_threshold for builtin models
	cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
	cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
	cfg.freeze()
	return cfg

def makedir(path):
	if not os.path.exists(path):
		os.makedirs(path)

args = Args()
cfg = setup_cfg(args)
predictor = DefaultPredictor(cfg)

def detect(imgname):
	img = read_image(imgname, format="BGR")
	predictions = predictor(img)['instances'].get_fields()

	bboxes, scrs, classes = predictions['pred_boxes'].tensor.cpu().numpy(), predictions['scores'].cpu().numpy(), predictions['pred_classes'].cpu().numpy()
	bboxes, scrs, classes = list(bboxes), list(scrs), list(classes)

	boxes = []
	for bbox,scr,c in zip(bboxes,scrs,classes):
		if c==0:
			# select the person class 
			boxes.append(bbox)
	return img, boxes

def crop(img, bboxes):
	results = []
	resboxes = []
	for b in bboxes:
		wh = max(b[2]-b[0], b[3]-b[1])
		center = [0.5*(b[2]+b[0]), 0.5*(b[3]+b[1])]
		wh = wh * 1.4 
		corner = [ center[0] - wh*0.5, center[1] - wh*0.5 ]
		H = np.float32([[1,0,-corner[0]], [0,1,-corner[1]]])
		cropped = cv2.warpAffine(img, H, (int(wh),int(wh)))
		results.append(cropped)
		newbox = [corner[0], corner[1], corner[0]+wh, corner[1]+wh]
		resboxes.append(newbox)
	return results, resboxes

def run_frames(path):
	print('Detecting: ', path)
	makedir(os.path.join(path, 'bboxes/'))

	for i in tqdm(glob.glob(os.path.join(path,'imgs/*.jpg'))):
		i = i.replace('\\','/')
		img, bboxes = detect(i)
		cropped, bboxes = crop(img, bboxes)
		
		foldername = os.path.basename(i)
		foldername = os.path.splitext(foldername)[0]
		makedir(  os.path.join(path, 'cropped/%s/'%(foldername)) )
		for i,c in enumerate(cropped):
			cv2.imwrite(  os.path.join(path, 'cropped/%s/%04d.png'%(foldername,i)), c)
		pickle.dump(bboxes, open( os.path.join(path, 'bboxes/%s.pkl'%(foldername)), 'wb'))

if __name__=='__main__':
	run_frames('./data/')
