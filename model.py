import nn
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

class CapsuleNet(nn.Module):
	"""docstring for CapsuleNet"""
	def __init__(self,):
		super(CapsuleNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)
		self.primary_cap_conv = [nn.Conv2d(256, 8, kernel_size=9, stride=1) for i in range(32)]
		
	def forward(self, input):
		conv1_out = self.conv1(input)
		conv1_relu_out = F.elu(conv1_out)
		primary_cap_list = [self.primary_cap_conv[i](conv1_relu_out) \
                            for i in range(32)]