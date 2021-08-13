import torch
from torch import nn
import math

class RelationMemory(nn.Module):
    def __init__(self, dim_s, dim_t, inputSize, outputSize, K, T, momentum=0.5, device='cuda'):
        super(RelationMemory, self).__init__()
        # M_T, M_T_S
        self.m_t = Synchronize(dim_t, dim_t, inputSize)
        self.m_t_s = Synchronize(dim_s, dim_t, inputSize)
        # critic
        self.h_t = Embed(inputSize, inputSize)
        self.h_t_s = Embed(inputSize, inputSize)
        # memory
        self.inputSize = inputSize
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.to(device)
        self.register_buffer('params', torch.tensor([K, T, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_v2', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def _forward(self, v1, v2, forward_type='s'):
        if forward_type == 's':
            w_v = self.w_s_v
            relu = self.relu_s
            w_v1 = self.w_s_v1
            w_v2 = self.w_s_v2
        elif forward_type == 't':
            w_v = self.w_t_v
            relu = self.relu_t
            w_v1 = self.w_t_v1
            w_v2 = self.w_t_v2
        return w_v(relu(w_v1(v1)-w_v2(v2)))
    
    def forward(self, v1, v2, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        momentum = self.params[2].item()
        batchSize = v1.size(0)
        outputSize = self.nLem
        inputSize = self.inputSize

        # original score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * batchSize * (K + 1)).view(batchSize * batchSize, -1)
            idx.select(1, 0).copy_(y.tile(y.shape[0]).data)

        # sample
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize).transpose(0, 1)

        # critic
        outs_m_t = torch.stack([self.h_t(self.m_t(v2, w_v2)) for w_v2 in weight_v2])
        outs_m_t_s = torch.stack([self.h_t_s(self.m_t_s(v1, w_v2)) for w_v2 in weight_v2])


        # update memory
        with torch.no_grad():
            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)
        
        return torch.div(torch.exp(torch.div(torch.sum(torch.mul(outs_m_t, outs_m_t_s), dim=2, keepdim=True), T)), torch.exp(torch.div(1, T)))


class Synchronize(nn.Module):
    """
    sub-module MT and MTS
    """
    def __init__(self, dim_v1, dim_v2, dim_out):
        super(Synchronize, self).__init__()
        self.w_v1 = nn.Linear(dim_v1, dim_out)
        self.w_v2 = nn.Linear(dim_v2, dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.v = nn.Linear(dim_out, dim_out)
    
    def forward(self, v1, v2):
        return self.v(self.relu(self.w_v1(v1.expand_as(v2))-self.w_v2(v2)))



class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """
    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def to(self, device):
        self.prob = self.prob.to(device)
        self.alias = self.alias.to(device)

    def draw(self, N):
        """ Draw N samples from multinomial """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj

class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out