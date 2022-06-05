import math
from numbers import Number

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class VariationalBottleneck(torch.nn.Module):
    """
    Implementation inspired by https://github.com/1Konny/VIB-pytorch/blob/master/model.py
    """
    def __init__(self, in_shape, K=256, beta=1e-3, *args, **kwargs):
        super().__init__()
        self.in_shape = in_shape
        self.K = K
        self.beta = beta

        if len(in_shape) == 1: # previous layer is a Linear layer
            enc_in_dim = in_shape[0]
            self.reducer = None
        elif len(in_shape) == 3:
            enc_in_dim = np.prod(in_shape)
            self.reducer = None
        else: 
            enc_in_dim = np.prod(in_shape[1:])
            self.reducer = nn.Conv2d(in_shape[0], 1, 1)

        self.encoder = nn.Linear(enc_in_dim, 2 * self.K)
        self.decoder = nn.Linear(self.K, np.prod(in_shape))

        self.mu = Variable(torch.Tensor(K))
        self.std = Variable(torch.Tensor(K))
        self.out_feats = Variable(torch.Tensor(in_shape))

    def forward(self, x):
        batch_size = x.shape[0]
        x_out = x
        if x.dim() > 2 and x.shape[1] > 1 and self.reducer != None:
            x = self.reducer(x)

        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        statistics = self.encoder(x)
        
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:])
        self.mu = mu
        self.std = std
    
        encoding = self.reparameterize(mu, std) # pull sample from the distribution

        x = self.decoder(encoding.unsqueeze(0))
        x_out = x.view((batch_size, *self.in_shape))
        self.out_feats = x_out

        return x_out


    def reparameterize(self, mu, std):
        def check_number(vector):
            if isinstance(vector, Number):
                return torch.Tensor([vector])
            else:
                return vector

        mu = check_number(mu)
        std = check_number(std)
        eps = Variable(std.data.new(std.size()).normal_().to(mu.device))

        return mu + eps * std

    def loss(self):
        return self.beta * (-0.5*(1+2*self.std.log()-self.mu.pow(2)-self.std.pow(2)).sum(1).mean().div(math.log(2))) # KLD to standard normal distribution

