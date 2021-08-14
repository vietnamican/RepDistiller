import torch
from torch import nn
from torch.nn import functional as F
import math

class RelationMemory(nn.Module):
    def __init__(self, dim_s, dim_t, input_size, output_size, K, T, momentum=0.5, device='cuda'):
        super(RelationMemory, self).__init__()
        self.embed_s = nn.Linear(dim_s, input_size)
        self.embed_t = nn.Linear(dim_t, input_size)
        self.m_t_v = nn.Linear(input_size, input_size)
        self.m_t_q = nn.Linear(input_size, input_size)
        self.m_t_s_v = nn.Linear(input_size, input_size)
        self.m_t_s_q = nn.Linear(input_size, input_size)
        self.m_t = nn.Linear(input_size, input_size)
        self.m_t_s = nn.Linear(input_size, input_size)
        # # M_T, M_T_S
        # self.m_t = Synchronize(dim_t, dim_t, input_size)
        # self.m_t_s = Synchronize(dim_s, dim_t, input_size)
        # critic
        self.h_t = Embed(input_size, input_size)
        self.h_t_s = Embed(input_size, input_size)
        # memory
        self.input_size = input_size
        self.nLem = output_size
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.to(device)
        self.register_buffer('params', torch.tensor([K, T, momentum]))
        stdv = 1. / math.sqrt(input_size / 3)
        self.register_buffer('memory_s', torch.rand(output_size, input_size).mul_(2 * stdv).add_(-stdv))
    
    def forward(self, s, t, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        momentum = self.params[2].item()
        batch_size = s.size(0)
        output_size = self.nLem
        input_size = self.input_size
        idx = idx.view(batch_size,batch_size,-1)[:,:,:K]
        # idx = idx.view(batch_size*batch_size,-1).select(1, 0).copy_(y.tile(y.size(0)).data).transpose(0, 1)
        # original score computation
        if idx is None:
            idx = self.multinomial.draw(batch_size * batch_size * (K + 1)).view(batch_size * batch_size, -1)
            idx.select(1, 0).copy_(y.tile(y.shape[0]).data)

        # sample
        neg_s = torch.index_select(self.memory_s, 0, idx.reshape(-1)).detach()
        neg_s = neg_s.view(batch_size, batch_size, K, input_size)

        # embed
        debug = False
        s = self.embed_s(s) # 64,128
        t = self.embed_t(t) # 64,128
        m_t_v = self.m_t_v(t) # 64,128
        m_t_q = self.m_t_q(t) # 64,128
        if debug:
            print('m_t_v: ', m_t_v.shape)
            print('m_t_q: ', m_t_q.shape)
        m_t_s_v = self.m_t_s_v(t) # 64,128
        m_t_s_q_pos = self.m_t_s_q(s) # 64,128
        m_t_s_q_neg = self.m_t_s_q(neg_s) # 64,64,512,128
        if debug:
            print('m_t_s_v: ', m_t_s_v.shape)
            print('m_t_s_q_pos: ', m_t_s_q_pos.shape)
            print('m_t_s_q_neg: ', m_t_s_q_neg.shape)
        r_t = self.m_t(F.relu(m_t_v.unsqueeze(0) - m_t_q.unsqueeze(1))) # 64,64,128
        r_t_s_pos = self.m_t_s(F.relu(m_t_s_v.unsqueeze(0) - m_t_s_q_pos.unsqueeze(1))) # 64,64,128
        r_t_s_neg = self.m_t_s(F.relu(m_t_s_v.unsqueeze(0).unsqueeze(2) - m_t_s_q_neg)) # 64,64,512,128
        if debug:
            print('r_t: ', r_t.shape)
            print('r_t_s_pos: ', r_t_s_pos.shape)
            print('r_t_s_neg: ', r_t_s_neg.shape)
        h_t = self.h_t(r_t, 2) # 64,64,128
        h_t_s_pos = self.h_t_s(r_t_s_pos, 2) # 64,64,128
        h_t_s_neg = self.h_t_s(r_t_s_neg, 3) # 64,64,512,128
        h_t_s = torch.cat([h_t_s_pos.unsqueeze(2), h_t_s_neg], dim=2).view(batch_size, batch_size, K+1, -1)
        if debug:
            print('h_t: ', h_t.shape)
            print('h_t_s_pos: ', h_t_s_pos.shape)
            print('h_t_s_neg: ', h_t_s_neg.shape)
            print('h_t_s: ', h_t_s.shape)

        # update memory
        with torch.no_grad():
            ab_pos = torch.index_select(self.memory_s, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(s, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_s = ab_pos.div(ab_norm)
            self.memory_s.index_copy_(0, y, updated_s)
        return torch.div(torch.exp(torch.div(torch.sum(torch.mul(h_t.unsqueeze(2), h_t_s), dim=3, keepdim=True), T)), torch.exp(torch.div(1, T))).view(batch_size * batch_size, -1, 1) # 64,64,513,128
        # return torch.div(torch.exp(torch.div(torch.sum(torch.mul(h_t.unsqueeze(2), h_t_s), dim=2, keepdim=True), T)), torch.exp(torch.div(1, T))) # 64,64,513,128


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

    def forward(self, x, dim):
        x = self.linear(x)
        x = self.l2norm(x, dim)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x, dim=1):
        norm = x.pow(self.power).sum(dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

# class nn.Linear(nn.Module):
#     def __init__(self, inplanes, outplanes):
#         super(nn.Linear, self).__init__()
#         self.linear = nn.Linear(inplanes, outplanes)
    
#     def forward(self, x):
#         feature_dim = x.size(-1)
#         orig_size = x.shape[:-1]
#         print(orig_size)
#         batch_size = orig_size.numel()
#         if batch_size > 512:
#             result = []
#             for i in range(0, batch_size, 512):
#                 x_chunk = x[i:i+512]
#                 result.append(self.linear(x_chunk))
#             return torch.cat(result, dim=0)
#         else:
#             return self.linear(x.view(batch_size, -1)).view(orig_size, -1)

if __name__ == '__main__':
    model = nn.Linear(64, 64)
    x = torch.Tensor(16, 128, 128, 64)
    x = model(x)
    print(x.shape)