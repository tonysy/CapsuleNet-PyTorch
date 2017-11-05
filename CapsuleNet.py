import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from CapsuleLayer import CapsuleConv, CapsuleLinear

class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256,     
                        kernel_size=9,stride=1)

        self.primary_capsules = CapsuleConv(dim_vector=8, in_channels=256, out_channels=32,
                                kernel_size=9,stride=2)

        self.digit_capsules = CapsuleLinear(dim_vector=16, dim_input_vector=8,  
                                            out_channels=10, num_routing_iter=3)

        self.decoder_module = nn.Sequential(
                                nn.Linear(16, 512),
                                nn.ReLU(inplace=True),
                                nn.Linear(512, 1024),
                                nn.ReLU(inplace=True),
                                nn.Linear(1024, 784),
                                nn.Sigmoid()
                            )

    def forward(self, X, y=None, with_label=True):
        import pdb
        # pdb.set_trace()
        # input batch_sizex1x28x28
        X = F.elu(self.conv1(X), inplace=True)
        X = self.primary_capsules(X)
        # batch_size x 10 x 16
        X = self.digit_capsules(X)
        X = X.view(X.size(0),X.size(2),X.size(4))
        X_l2norm = torch.sqrt((X ** 2).sum(dim=-1))
        prob = F.softmax(X_l2norm)

        if with_label:
            # size: batch_size
            max_len_indices = y
            
        else:
            # size: batch_size
            max_len_indices = prob.max(dim=1)
        import pdb
        # pdb.set_trace()
        batch_activated_capsules = X[range(X.size()[0]), max_len_indices.data] # batch_size x 16

        reconstructions = self.decoder_module(batch_activated_capsules)

        return prob, X_l2norm, reconstructions

class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        
        self.reconstruction_loss = nn.MSELoss(size_average=True)

    def forward(self, num_classes, images, labels, X_l2norm, reconstructions, 
                lambda_value=0.5, scale_factor=0.0005):
        self.num_classes = num_classes
        # import pdb; pdb.set_trace()
        # first_term_base = F.elu(0.9 - X_l2norm, inplace=True) ** 2
        # second_term_base = F.elu(X_l2norm - 0.1, inplace=True) ** 2
        zeros = Variable(torch.zeros(1)).cuda()
        first_term_base = torch.max(0.9 - X_l2norm,zeros) ** 2
        second_term_base = torch.max(X_l2norm - 0.1, zeros) ** 2
        labels = Variable(torch.FloatTensor(labels).cuda())
        margin_loss = labels * first_term_base + lambda_value * \
                        (1.0 - labels) * second_term_base
        # margin_loss = Variable(labels * first_term_base.data + lambda_value * \
                        # (1.0 - labels) * second_term_base.data)
        margin_loss = margin_loss.sum(dim=1).mean()

        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return margin_loss + scale_factor * reconstruction_loss
