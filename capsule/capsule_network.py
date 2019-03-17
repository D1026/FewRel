"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829
PyTorch implementation by Ivan_Duan @ AntFin.AI Depart.
"""
import sys

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import numpy as np


class CapsuleLayer(nn.Module):
    def __init__(self, input_dim, output_dim, input_atoms, output_atoms, paras=None, num_routing=3, leaky=False, kernel_size=None, stride=None,
                 ):
        super(CapsuleLayer, self).__init__()
        self.input_shape = (input_dim, input_atoms)  # omit batch dim
        self.output_shape = (output_dim, output_atoms)
        self.num_routing = num_routing
        self.leaky = leaky
        if paras is None:
            self.weights = nn.Parameter(torch.randn(input_dim, input_atoms, output_dim * output_atoms))
            self.biases = nn.Parameter(torch.randn(output_dim, output_atoms))
        else:
            self.weights = nn.Parameter(paras['weights'])
            self.biases = nn.Parameter(paras['biases'])

    def _squash(self, input_tensor):
        norm = torch.norm(input_tensor, dim=2, keepdim=True)
        norm_squared = norm * norm
        return (input_tensor / norm) * (norm_squared / (1 + norm_squared))

    def _leaky_route(self, x, output_dim):
        leak = torch.zeros(x.shape).to(x.device.type)
        leak = leak.sum(dim=2, keepdim=True)
        leak_x = torch.cat((leak, x), 2)
        leaky_routing = F.softmax(leak_x, dim=2)
        return leaky_routing[:, :, 1:]

    def _margin_loss(self, labels, raw_logits, margin=0.4, downweight=0.5):
        logits = raw_logits - 0.5
        positive_cost = labels * torch.lt(logits, margin).float() * torch.pow(logits - margin, 2)
        negative_cost = (1 - labels) * torch.gt(logits, -margin).float() * torch.pow(logits + margin, 2)
        margin_loss = 0.5 * positive_cost + downweight * 0.5 * negative_cost
        per_example_loss = torch.sum(margin_loss, dim=-1)
        loss = torch.mean(per_example_loss)
        return loss

    def forward(self, x, fast_weights=None):
        x = x.unsqueeze(-1).repeat(1, 1, 1, self.output_shape[0]*self.output_shape[1])  # [b, i, i_o, j*j_o]
        if fast_weights is None:
            votes = torch.sum(x * self.weights, dim=2)  # [b, i, j*j_o]
        else:
            votes = torch.sum(x * fast_weights['weights'], dim=2)  # [b, i, j*j_o]
        votes_reshaped = torch.reshape(votes,
                                       [-1, self.input_shape[0], self.output_shape[0], self.output_shape[1]])  # [b, i, j, j_o]
        # votes_t_shape = [3, 0, 1, 2]
        # votes_trans = votes_reshaped.permute(*votes_t_shape)

        # routing loop
        logits = torch.zeros(x.shape[0], self.input_shape[0], self.output_shape[0]).to(x.device.type)  # [b, i, j]
        for i in range(self.num_routing):
            if self.leaky:
                route = self._leaky_route(logits, self.output_shape[0])
            else:
                route = F.softmax(logits, dim=2)
            route = route.unsqueeze(-1)  # [b, i, j, 1]
            preactivate_unrolled = route * votes_reshaped   # [b, i, j, j_o]

            if fast_weights is None:
                s = preactivate_unrolled.sum(1, keepdim=True) + self.biases  # [b, 1, j, j_o]
            else:
                s = preactivate_unrolled.sum(1, keepdim=True) + fast_weights['biases']  # [b, 1, j, j_o]

            v = self._squash(s)

            distances = (votes_reshaped * v).sum(dim=3)  # [b, i, j]
            logits = logits + distances

        return v


class CapsuleClassification(nn.Module):
    def __init__(self):
        super(CapsuleClassification, self).__init__()

    # x, the capsule layer output
    def forward(self, x):
        x = x.squeeze()  # [b, j, j_o]
        logits = torch.norm(x, dim=-1)
        probs = F.softmax(logits, dim=-1)
        return logits, probs

# from collections import OrderedDict
# p = CapsuleLayer(5, 3, 5, 3).named_parameters()
# ps = OrderedDict((name, param) for (name, param) in p)
# print(ps['weights'])
# a = ps['weights'] * 8
# print(a)
#
# for i, n in enumerate(np.arange(5)):
#     print(i, ' --- ', n)

for i in range(5):
    pass
print(i)