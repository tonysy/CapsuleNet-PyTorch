import torch
import torch.nn as nn
import torch.nn.functional as F

from CapsuleLayer import CapsuleLayer

class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256,     
                        kernel_size=9,stride=1)

        self.primary_capsules = CapsuleLayer(num_capsules=8,num_routing=-1, in_channels=256,
                                    out_channels=32,kernel_size=9,stride=2)
        
        self.digit_capsules = CapsuleLayer(num_capsules=10, num_routing=32*6*6, 
                                in_channels=8, out_channels=16)

        self.decoder_module = nn.Sequential(
                                nn.Linear(16, 512),
                                nn.ReLU(inplace=True),
                                nn.Linear(512, 1024),
                                nn.ReLU(inplace=True),
                                nn.Sigmoid()
                            )
    
    def forward(self, X, y):
        X = F.elu(self.conv1(X), inplace=True)
        X = self.primary_capsules(X)
        X = self.digit_capsules(X).squeeze().transpose(0, 1)

        X_l2norm = torch.sqrt((X ** 2).sum(dim=-1))
        prob = F.softmax(X_l2norm)

        max_len_indices = y.max(dim=1)

        vectors = []
        for batch, index in enumerate(max_len_indices):
            vectors.append(X[batch, index.data[0], :])

        reconstructions = self.decoder_module(torch.stack(vectors, dim=0))

        return prob, reconstructions

class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=True)

    def forward(self, images, labels, prob_X, reconstructions, 
                lambda_value=0.5, scale_factor=0.0005):
        first_term_base = torch.clamp(0.9 - prob_X, min=0) ** 2
        second_term_base = torch.clamp(prob_X - 0.1, min=0) ** 2

        margin_loss = labels * first_term_base + lambda_value * \
                        (1 - labels) * second_term_base
        margin_loss = margin_loss.sum(dim=1).mean()

        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return margin_loss + scale_factor * reconstruction_loss

