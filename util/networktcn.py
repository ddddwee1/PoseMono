from TorchSUL import Model as M 
import random
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 

class ResBlock1D(M.Model):
	def initialize(self, outchn=1024, k=3):
		self.k = k 
		# self.bn = M.BatchNorm()
		self.c1 = M.ConvLayer1D(k, outchn, stride=k, activation=M.PARAM_PRELU, batch_norm=True, usebias=False, pad='VALID')
		self.c2 = M.ConvLayer1D(1, outchn, activation=M.PARAM_PRELU, batch_norm=True, usebias=False, pad='VALID')

	def forward(self, x):
		short = x

		# residual branch
		# branch = self.bn(x)
		# branch = M.activation(branch, M.PARAM_LRELU)

		branch = self.c1(x)
		branch = F.dropout(branch, 0.5, self.training, False)
		branch = self.c2(branch)
		branch = F.dropout(branch, 0.5, self.training, False)
		# slicing & shortcut
		# branch_shape = branch.shape[-1]
		# short_shape = short.shape[-1]
		# start = (short_shape - branch_shape) // 2
		short = short[:, :, self.k//2::self.k]
		res = short + branch
		# res = F.dropout(res, 0.25, self.training, False)

		return res

class Refine2dNet(M.Model):
	def initialize(self, num_kpts, temp_length, outdim):
		self.num_kpts = num_kpts
		self.outdim = outdim
		self.temp_length = temp_length
		self.c1 = M.ConvLayer1D(3, 1024, stride=3, activation=M.PARAM_PRELU, pad='VALID', batch_norm=True, usebias=False)
		self.r1 = ResBlock1D(k=3)
		# self.c2 = M.ConvLayer1D(3, 1024, activation=M.PARAM_PRELU, pad='VALID', batch_norm=True, usebias=False)
		self.r2 = ResBlock1D(k=3)
		# self.c3 = M.ConvLayer1D(3, 1024, activation=M.PARAM_PRELU, pad='VALID', batch_norm=True, usebias=False)
		self.r3 = ResBlock1D(k=3)
		# self.r4 = ResBlock1D(k=3)
		# self.r3 = ResBlock1D(k=3, dilation=3)
		# self.c5 = M.ConvLayer1D(9, 256, activation=M.PARAM_PRELU, pad='VALID', batch_norm=True, usebias=False)
		self.c4 = M.ConvLayer1D(1, num_kpts*outdim)

	def forward(self, x, drop=True):
		x = x.view(x.shape[0], x.shape[1], self.num_kpts * 2)
		x = x.permute(0,2,1)
		x = self.c1(x)
		x = self.r1(x)
		# x = self.c2(x)
		x = self.r2(x)
		# x = self.c3(x)
		x = self.r3(x)
		# x = self.r4(x)
		# x = self.r5(x)
		# x = self.c5(x)
		x = self.c4(x)
		x = x.permute(0, 2, 1)
		x = x.reshape(x.shape[0], x.shape[1], self.num_kpts, self.outdim)
		return x 

	def evaluate(self, x):
		aa = []
		# print(x.shape)
		for i in range(x.shape[0]-self.temp_length+1):
			aa.append(x[i:i+self.temp_length])
		aa = torch.stack(aa, dim=0)
		y = self(aa)
		y = y.permute(1,0,2,3)
		y = y[0]
		return y

class NetBundle(M.Model):
	def initialize(self, num_kpts, temp_length):
		self.net2d = Refine2dNet(num_kpts, temp_length, 2)
		self.net3d = Refine2dNet(num_kpts, temp_length, 1)

	def forward(self, x):
		x2d = self.net2d(x)
		x3d = self.net3d(x)
		x = torch.cat([x2d, x3d], dim=-1)
		return x 

	def evaluate(self, x):
		x2d = self.net2d.evaluate(x)
		x3d = self.net3d.evaluate(x)
		x = torch.cat([x2d, x3d], dim=-1)
		return x 
