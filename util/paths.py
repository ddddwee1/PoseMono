import os 

def makedir(path):
	if not os.path.exists(path):
		os.makedirs(path)