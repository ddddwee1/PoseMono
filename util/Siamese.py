import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from TorchSUL import Model as M 
from . import hrnet 
from torchvision.ops.roi_align import roi_align
import config 

class DepthToSpace(M.Model):
	def initialize(self, block_size):
		self.block_size = block_size
	def forward(self, x):
		bsize, chn, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
		assert chn%(self.block_size**2)==0, 'DepthToSpace: Channel must be divided by square(block_size)'
		x = x.view(bsize, -1, self.block_size, self.block_size, h, w)
		x = x.permute(0,1,4,2,5,3)
		x = x.reshape(bsize, -1, h*self.block_size, w*self.block_size)
		return x 

class UpSample(M.Model):
	def initialize(self, upsample_layers, upsample_chn):
		self.prevlayers = nn.ModuleList()
		#self.uplayer = M.DeConvLayer(3, upsample_chn, stride=2, activation=M.PARAM_PRELU, batch_norm=True, usebias=False)
		self.uplayer = M.ConvLayer(3, upsample_chn*4, activation=M.PARAM_PRELU, usebias=False)
		self.d2s = DepthToSpace(2)
		self.postlayers = nn.ModuleList()
		for i in range(upsample_layers):
			self.prevlayers.append(M.ConvLayer(3, upsample_chn, activation=M.PARAM_PRELU, batch_norm=True, usebias=False))
		for i in range(upsample_layers):
			self.postlayers.append(M.ConvLayer(3, upsample_chn, activation=M.PARAM_PRELU, batch_norm=True, usebias=False))
	def forward(self, x):
		for p in self.prevlayers:
			x = p(x)
		x = self.uplayer(x)
		x = self.d2s(x)
		# print('UPUP', x.shape)
		for p in self.postlayers:
			x = p(x)
		return x 

class ExtraLayer(M.Model):
	def initialize(self, outchn):
		self.c1 = M.ConvLayer(5, 256, activation=M.PARAM_PRELU)
		self.c3 = M.ConvLayer(5, 256, activation=M.PARAM_PRELU)
		self.c2 = M.ConvLayer(3, outchn, usebias=False)
	def forward(self, x):
		x = self.c1(x)
		x = self.c3(x)
		x = self.c2(x)
		return x 

class TransAfter(M.Model):
	def initialize(self):
		self.f1 = M.Dense(256, activation=M.PARAM_PRELU)
		self.f3 = M.Dense(256, activation=M.PARAM_PRELU)
		self.f2 = M.Dense(49)
	def forward(self, x):
		x = self.f1(x)
		x = self.f3(x)
		x = self.f2(x)
		return x 

class Siamese(M.Model):
	def initialize(self):
		self.backbone = hrnet.Body()
		self.upsample = UpSample(2, 64)
		self.extra1 = ExtraLayer(64)
		self.extra2 = ExtraLayer(64)
		self.post_template = TransAfter()

	def forward(self, temp, target, bboxes):
		fmap_tmp = self.backbone(temp)
		fmap_tmp = self.upsample(fmap_tmp)

		fmap_tgt = self.backbone(target)
		fmap_tgt = self.upsample(fmap_tgt)

		map_tmp = self.extra1(fmap_tmp)
		map_tgt = self.extra2(fmap_tgt)
		# print(bboxes.shape)
		# print(map_tmp.shape, map_tgt.shape)
		small_maps = roi_align(map_tmp, bboxes, (7,7), 1, -1)

		shape = small_maps.shape

		small_maps = small_maps.view(-1, shape[2]*shape[3])
		small_maps = self.post_template(small_maps)

		small_maps = small_maps.view(-1, config.num_pts_tracker, shape[1], shape[2], shape[3])
		
		res = []
		map_tgt = map_tgt.unsqueeze(1)
		for i in range(map_tgt.shape[0]):
			buff = F.conv2d(map_tgt[i], small_maps[i], None, 1, 3, 1, 1) # 3 is padding 
			res.append(buff)
		res = torch.cat(res, dim=0)

		return res 

	def extract_src(self, temp, bboxes):
		with torch.no_grad():
			fmap_tmp = self.backbone(temp)
			fmap_tmp = self.upsample(fmap_tmp)
			map_tmp = self.extra1(fmap_tmp)
			small_maps = roi_align(map_tmp, bboxes, (7,7), 1, -1)
			shape = small_maps.shape
			small_maps = small_maps.view(-1, shape[2]*shape[3])
			small_maps = self.post_template(small_maps)
			small_maps = small_maps.view(-1, config.num_pts_tracker, shape[1], shape[2], shape[3])
		return small_maps

	def extract_tgt(self, target):
		with torch.no_grad():
			fmap_tgt = self.backbone(target)
			fmap_tgt = self.upsample(fmap_tgt)
			map_tgt = self.extra2(fmap_tgt)
			# map_tgt = map_tgt.unsqueeze(1)
		return map_tgt

	def match(self, kernel, tgt):
		return F.conv2d(tgt, kernel, None, 1, 3, 1, 1)

if __name__=='__main__':
	import numpy as np 
	net = Siamese()

	bboxes = np.float32([[0,0,0,100,100]] * 14 + [[1,2,2,100,100]] * 14)
	temp = np.float32(np.zeros([2,3,384,384]))
	target = np.float32(np.zeros([2,3,384,384]))

	bboxes = torch.from_numpy(bboxes)
	temp = torch.from_numpy(temp)
	target = torch.from_numpy(target)

	res = net(temp, target, bboxes)
	print(res.shape)
