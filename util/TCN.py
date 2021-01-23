from . import networktcn 

seq_len = 243
nettcn = networktcn.Refine2dNet(17, seq_len)
x_dumb = torch.zeros(2,243, 17*3)
nettcn(x_dumb)
M.Saver(nettcn).restore('./models/model_tcn/')
nettcn.eval()
nettcn.cuda()

def run_points(pts):
	pts = pts.astype(np.float32)[:,:2]
	pts = torch.from_numpy(pts).cuda()
	pred = F.pad(pts, (0,0,0,0,seq_len//2, seq_len//2), mode='replicate')
	pred = pred.squeeze()
	with torch.no_grad():
		pred = nettcn.evaluate(pts)
	pred = pred.cpu().numpy()
	return pred
