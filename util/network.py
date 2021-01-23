import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
from TorchSUL import Model as M 
import config 
from . import hrnet 

class Head(M.Model):
	def initialize(self, head_layernum, head_chn):
		self.layers = nn.ModuleList()
		for i in range(head_layernum):
			self.layers.append(M.ConvLayer(3, head_chn, activation=M.PARAM_PRELU, batch_norm=True, usebias=False))
	def forward(self, x):
		for l in self.layers:
			x = l(x)
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

class DensityNet(M.Model):
	def initialize(self, head_layernum, head_chn, upsample_layers, upsample_chn):
		self.backbone = hrnet.Body()
		self.upsample = UpSample(upsample_layers, upsample_chn)
		self.head = Head(head_layernum, head_chn)
		self.c1 = M.ConvLayer(1, config.num_pts)

	def build_forward(self, x, *args, **kwargs):
		feat = self.backbone(x)
		feat = self.upsample(feat)
		feat1 = self.head(feat)
		outs = self.c1(feat1)
		nn.init.normal_(self.c1.conv.weight, std=0.001)
		print('normal init for last conv ')
		return outs
		
	def forward(self, x, density_only=False):
		feat = self.backbone(x)
		feat = self.upsample(feat)
		h1 = self.head(feat)
		outs = self.c1(h1)
		return outs

def get_network():
	net = DensityNet(config.head_layernum, config.head_chn, config.upsample_layers, config.upsample_chn)
	x = torch.zeros(1, 3, config.inp_size, config.inp_size)
	with torch.no_grad():
		net(x)
	return x 
