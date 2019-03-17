"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829

"""
import sys

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import numpy as np


class CapsuleLayer(nn.Module):
    def __init__(self, input_dim, output_dim, input_atoms, output_atoms, leaky=False, kernel_size=None, stride=None,
                 ):
        super(CapsuleLayer, self).__init__()
        self.input_shape = (input_dim, input_atoms)  # omit batch dim
        self.output_shape = (output_dim, output_atoms)

        self.leaky = leaky
        self.weights = nn.Parameter(torch.randn(input_dim, input_atoms, output_dim * output_atoms))
        self.biases = nn.Parameter(torch.randn(output_dim, output_atoms))

        self.attW = nn.Parameter(torch.randn(input_dim, output_dim))
        self.attB = nn.Parameter(torch.randn(input_dim, output_dim))

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

    def forward(self, x, labelsVec):

        att = torch.matmul(x, labelsVec.transpose(0, 1))    # [b, i, j]
        att = F.normalize(att, dim=-1)
        # ---
        # print(' --- x ---')
        # print(x.detach().cpu().numpy().tolist()[0])
        # print(' --- Vec --- ')
        # print(labelsVec.detach().cpu().numpy().tolist()[0])
        # print(' --- before softmax --- ')
        # print(att.detach().cpu().numpy().tolist()[0])

        x = x.unsqueeze(-1).repeat(1, 1, 1, self.output_shape[0]*self.output_shape[1])  # [b, i, i_o, j*j_o]
        votes = torch.sum(x * self.weights, dim=2)  # [b, i, j*j_o]
        votes_reshaped = torch.reshape(votes,
                                       [-1, self.input_shape[0], self.output_shape[0], self.output_shape[1]])  # [b, i, j, j_o]

        # routing loop
        att = F.softmax(att, dim=-1)

        # --- att * w + b ---
        att = att * self.attW + 1.0
        # att = att * self.attW + self.attB
        # ---
        # print('--- Attention weights ---')
        # print(att.detach().cpu().numpy().tolist()[0])
        # print(' --- ')
        att = att.unsqueeze(-1)   # [b, i, j, 1]

        preactivate_unrolled = att * votes_reshaped   # [b, i, j, j_o]

        s = preactivate_unrolled.sum(1, keepdim=True) + self.biases  # [b, 1, j, j_o]
        v = self._squash(s)


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


