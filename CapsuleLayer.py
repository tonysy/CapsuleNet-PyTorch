import torch
import torch.nn as nn
from torch.autograd import Variable

class CapsuleLayer(nn.Module):
    """
    Args:
        num_routing_iter: number of iterations for the routing algorithm (default: 3)

        num_routing_nodes:
                           when num_routings = -1, it means it's the first capsule layer

    """
    def __init__(self, num_capsules, in_channels, out_channels, kernel_size, 
                    num_routing_nodes, stride=1, padding=0, num_routing_iter=3):
        super(CapsuleLayer, self).__init__()

        self.num_capsules = num_capsules
        self.num_routing = num_routing
        self.num_routing_iter = num_routing_iter

        if num_routing_nodes == -1:
            # for primary_capsule layer
            self.capsules_list = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 
                                kernel_size=kernel_size, stride=stride, padding=padding) 
                                for item in range(num_capsules)])
        else:
            self.rout_weights = nn.Parameter(torch.randn(num_capsules, num_routing_nodes, in_channels, out_channels))
    def squash(self, tensor, dim=-1):
        tensor_l2norm = (tensor**2).sum(dim=dim, keepdim=True)
        scale_factor = tensor_l2norm / (1 + tensor_l2norm)
        tensor_squashed = scale_factor * tensor / torch.sqrt(tensor_l2norm)
        return tensor_squashed
    def routing_softmax(self,input, dim=1):
        transposed_input = input.transpose(dim, len(input.size())-1)
        softmaxed_output = F.softmax(transposed_input.contiguouts().view(-1, transposed_input.size(-1)))

        return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size())-1)
    def forward(self, X):
        if self.num_routing_nodes == 1:
            # For the primary layer
            outputs = [capsule(X).view(X.size(0), -1, -1) for capsule in self.capsules_list]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)
        else:
            # For the DigitLayer in CapsuleNet-MINIST-V1
            # priors = X[None, :, :, None, :] @ self.rout_weights[:, None, :, :,:] # matrix multiplication
            priors = torch.mm(X[None, :, :, None, :], self.rout_weights[:, None, :, :,:])

            logits = Variable(torch.zeros(*priors.size())).cuda()
            for i in range(self.num_iterations):
                probs = routing_softmax(logits, dim=2)
                outputs = self.squash((probs*probs).sum(dim=2,keepdim=True))

                if i != self.num_routing_iter - 1:
                    delta_logits = (priors*outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits         

        return outputs

# def squash(vector):
#     """Squashing Function

#     Args:
#         vector: A 4-D tensor with shape [batch_size, num_caps, vec_len, 1],
#     Returns:
#         A 4-D tensor with the same shape as vector but
#         squashed in 3-rd and 4-th dimensions
#     """
#     vector_data = vector.numpy()
#     vector_abs = np.sqrt(np.sum(np.square(vector_data), axis=2)) # get length of vector

#     scalar_factor = np.square(vector_abs) / (1 + np.square(vector_abs))
#     vec_squashed = scalar_factor[:,:,:,np.newaxis] * (vector / vector_abs[:,:,:,np.newaxis])

#     return vec_squashed