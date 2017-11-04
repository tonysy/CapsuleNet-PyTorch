import torch
import torch.nn as nn
import torch.nn.functional as F

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
                                nn.Sigmoid()
                            )
    def onehot(self):
        batch_size = 5
        nb_digits = 10
        # Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
        y = torch.LongTensor(batch_size,1).random_() % nb_digits
        # One hot encoding buffer that you create out of the loop and just keep reusing
        y_onehot = torch.FloatTensor(batch_size, nb_digits)

        # In your for loop
        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)
        return y, y_onehot
        print(y)
        print(y_onehot)

    def forward(self, X, y):
        # input batch_sizex1x28x28
        X = F.elu(self.conv1(X), inplace=True)
        X = self.primary_capsules(X)
        X = self.digit_capsules(X).squeeze()
        X_l2norm = torch.sqrt((X ** 2).sum(dim=-1))
        prob = F.softmax(X_l2norm)

        max_len_indices = y.max(dim=0)

        vectors = []
        for batch, index in enumerate(max_len_indices):
            vectors.append(X[batch, index.data[0], :])

        reconstructions = self.decoder_module(torch.stack(vectors, dim=0))

        return prob, reconstructions

class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, prob_X, reconstructions, 
                lambda_value=0.5, scale_factor=0.0005):
        first_term_base = F.elu(0.9 - prob_X, inplace=True) ** 2
        second_term_base = F.elu(prob_X - 0.1, inplace=True) ** 2

        margin_loss = labels * first_term_base + lambda_value * \
                        (1.0 - labels) * second_term_base
        margin_loss = margin_loss.sum().mean()

        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + scale_factor * reconstruction_loss) / images.size(0)

