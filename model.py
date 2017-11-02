import nn
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

class CapsuleNet(nn.Module):
	"""docstring for CapsuleNet"""
	def __init__(self, arg):
		super(CapsuleNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 256, kernel_size=9)
		self.PrimaryCaps = 
		