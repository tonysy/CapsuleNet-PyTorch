import numpy as np 
from config import config
import torch
import torch.nn as nn
from torch.nn.Module import Conv2d




class CapsuleLayer(nn.Module):
    """ Capsule Layer

    Args:
        cap_sum: corresponding to channel number in Capsule Network Paper(e.g, 32 in CapsuleNet)
        input_size: 
        filter_num: convolutional filter number (e.g, 8 in CapsuleNet)

    Shapes:

    Returns:
    
    """
    def __init__(self, cap_num, input_size, filter_num, kernel_size):
        super(CapsuleLayer, self).__init__()
        self.capsule_num = cap_num
        self.capsule_conv = [nn.Conv2d(input_size, filter_num, kernel_size) \
                                for i in range(self.capsule_num)]

    # def capsule_conv(self, cap_num, input_size, filter_num, kernel_size):
    #     self.cap_conv0 = nn.Conv2d(input_size,32, 9)
    #     self.cap_conv1 = nn.Conv2d(input_size,32, 9)
    #     self.cap_conv2 = nn.Conv2d(input_size,32, 9)
    #     self.cap_conv3 = nn.Conv2d(input_size,32, 9)
    #     self.cap_conv4 = nn.Conv2d(input_size,32, 9)
    #     self.cap_conv5 = nn.Conv2d(input_size,32, 9)
    #     self.cap_conv6 = nn.Conv2d(input_size,32, 9)
    #     self.cap_conv7 = nn.Conv2d(input_size,32, 9)
        
    def forward(self, X):
        cap_conv_list = [self.capsule_conv[i](X) \
                            for i in range(self.capsule_num)] 
        # cap_conv0_out = self.cap_conv0(X)
        # cap_conv1_out = self.cap_conv1(X)
        # cap_conv2_out = self.cap_conv2(X)
        # cap_conv3_out = self.cap_conv3(X)
        # cap_conv4_out = self.cap_conv4(X)
        # cap_conv5_out = self.cap_conv5(X)
        # cap_conv6_out = self.cap_conv6(X)
        # cap_conv7_out = self.cap_conv7(X)
        # cap_conv_list = [cap_conv0_out,cap_conv1_out,
        #                  cap_conv2_out,cap_conv3_out,
        #                  cap_conv4_out,cap_conv5_out,
        #                  cap_conv6_out,cap_conv7_out]

        cap_out = torch.cat(cap_conv_list, dim=1)

        return cap_out


def squash(vector):
    """Squashing Function

    Args:
        vector: A 4-D tensor with shape [batch_size, num_caps, vec_len, 1],
    Returns:
        A 4-D tensor with the same shape as vector but
        squashed in 3-rd and 4-th dimensions
    """
    vector_data = vector.numpy()
    vector_abs = np.sqrt(np.sum(np.square(vector_data), axis=2)) # get length of vector

    scalar_factor = np.square(vector_abs) / (1 + np.square(vector_abs))
    vec_squashed = scalar_factor[:,:,:,np.newaxis] * (vector / vector_abs[:,:,:,np.newaxis])

    return vec_squashed