from TorchSUL import Model as M 
import random
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 

class ResBlock1D(M.Model):
	def initialize(self, outchn=512, dilation=1, k=3):
		self.bn = M.BatchNorm()
		self.c1 = M.ConvLayer1D(k, outchn, dilation_rate=dilation, activation=M.PARAM_PRELU, batch_norm=True, usebias=False, pad='VALID')
		self.c2 = M.ConvLayer1D(3, outchn, pad='VALID')

	def forward(self, x):
		short = x

		# residual branch
		branch = self.bn(x)
		branch = M.activation(branch, M.PARAM_LRELU)
		branch = self.c1(branch)
		branch = self.c2(branch)

		# slicing & shortcut
		branch_shape = branch.shape[-1]
		short_shape = short.shape[-1]
		start = (short_shape - branch_shape) // 2
		short = short[:, :, start:start+branch_shape]
		res = short + branch
		res = F.dropout(res, 0.4, self.training, False)
		return res

class Refine2dNet(M.Model):
	def initialize(self, num_pts, temp_length, pt_dim):
		self.num_pts = num_pts
		self.pt_dim = pt_dim
		self.temp_length = temp_length
		self.c1 = M.ConvLayer1D(5, 512, activation=M.PARAM_PRELU, batch_norm=True, usebias=False, pad='VALID')
		self.r1 = ResBlock1D(k=3, dilation=2)
		self.r2 = ResBlock1D(k=3, dilation=4)
		self.r3 = ResBlock1D(k=5, dilation=8)
		self.r4 = ResBlock1D(k=5, dilation=16)
		self.c5 = M.ConvLayer1D(9, 512, activation=M.PARAM_PRELU, batch_norm=True, usebias=False, pad='VALID')
		self.c4 = M.ConvLayer1D(1, num_pts*pt_dim)

	def forward(self, x):
		x = x.view(x.shape[0], x.shape[1], 17*2)
		x = x.permute(0,2,1)
		x = self.c1(x)
		x = self.r1(x)
		x = self.r2(x)
		x = self.r3(x)
		x = self.r4(x)
		x = self.c5(x)
		x = self.c4(x)
		x = x.permute(0,2,1)
		x = x.reshape(x.shape[0], x.shape[1], self.num_pts, self.pt_dim)
		return x 

	def evaluate(self, x):
		aa = []
		for i in range(x.shape[0]-self.temp_length+1):
			aa.append(x[i:i+self.temp_length])
		aa = torch.stack(aa, dim=0)
		y = self(aa)
		y = y[:,0]
		return y 

class Discriminator2D(M.Model):
	def initialize(self):
		self.c1 = M.ConvLayer1D(1, 1024, activation=M.PARAM_PRELU, batch_norm=True, usebias=False)
		self.c2 = M.ConvLayer1D(1, 256, activation=M.PARAM_PRELU, batch_norm=True, usebias=False)
		self.c3 = M.ConvLayer1D(1, 256, activation=M.PARAM_PRELU, batch_norm=True, usebias=False)
		self.c4 = M.ConvLayer1D(1, 1)

	def forward(self, x):
		return self.c4(self.c3(self.c2(self.c1(x))))

class modelBundle(M.Model):
	def initialize(self, num_pts, temp_length):
		self.refine2D = Refine2dNet(num_pts, temp_length, 2)
		self.Depth3D = Refine2dNet(num_pts, temp_length, 1)
	def forward(self, x):
		x2d = self.refine2D(x)
		x3d = self.Depth3D(x)
		# print(x2d.shape, x3d.shape)
		x = torch.cat([x2d, x3d], dim=-1)
		return x 
	def evaluate(self, x):
		x2d = self.refine2D.evaluate(x)
		x3d = self.Depth3D.evaluate(x)
		x = torch.cat([x2d, x3d], dim=-1)
		return x 

class NetworkRoot(M.Model):
	def initialize(self):
		self.c1 = M.ConvLayer1D(5, 512, pad='VALID', activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
		self.c2 = M.ConvLayer1D(5, 512, dilation_rate=2, pad='VALID', activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
		self.c3 = M.ConvLayer1D(3, 512, dilation_rate=4, pad='VALID', activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
		self.c4 = M.ConvLayer1D(3, 512, dilation_rate=4, pad='VALID', activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
		self.c5 = M.ConvLayer1D(5, 512, dilation_rate=8, pad='VALID', activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
		self.c6 = M.ConvLayer1D(5, 512, pad='VALID', activation=M.PARAM_LRELU, usebias=False, batch_norm=True)
		self.c7 = M.ConvLayer1D(1, 62, usebias=False)

	def forward(self, x):
		x = self.c1(x)
		x = self.c2(x)
		x = self.c3(x)
		x = F.dropout(x, 0.3, self.training, False)
		x = self.c4(x)
		x = F.dropout(x, 0.3, self.training, False)
		x = self.c5(x)
		x = F.dropout(x, 0.3, self.training, False)
		x = self.c6(x)
		x = self.c7(x)
		return x

if __name__=='__main__':
	net = modelBundle(17, 129)
	x = torch.zeros(150, 17, 2)
	y = net.evaluate(x)
	print(y.shape)
