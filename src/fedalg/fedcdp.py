import copy
import numpy as np

import torch

from src.fedalg import FedAlg

class FedCdp(FedAlg):
    def __init__(self, criterion, model, config):
        super().__init__(criterion, model, half=config.half)
    
        self.thres = config.thres
        self.snr = config.dpsnr

    def client_grad(self, x, y):
        n, c, w, h = x.shape
        x0 = x[0].view(1, c, w, h)
        out = self.model(x0)
        loss = self.criterion(out, y[0].view(1, -1))
        dy_dx = torch.autograd.grad(loss, self.model.parameters())

        if self.half:
            grad = list((_.detach().half().clone() for _ in dy_dx))
        else:
            grad = list((_.detach().clone() for _ in dy_dx))

        # clip_grad(grad, self.thres)
        # perturb_grad(grad, self.snr, self.thres)

        batch_size = n
        grad_aggregator = GradBuffer(grad)

        for i in range(1, batch_size):
            x_ = x[i].view(1, c, w, h)
            out = self.model(x_)
            loss = self.criterion(out, y[i].view(1, -1))
            dy_dx = torch.autograd.grad(loss, self.model.parameters())
            g = list((_.detach().clone() for _ in dy_dx))
            
            clip_grad(g, self.thres)
            perturb_grad(g, self.snr, self.thres)
            grad_aggregator += GradBuffer(g)

        grad_aggregator *= 1/batch_size
        grad = grad_aggregator._grad

        return grad


def clip_grad(grad, thres):
    for i, g in enumerate(grad):
        if grad[i].norm() > thres:
            grad[i] = g/grad[i].norm() * thres
    
def perturb_grad(grad, snr, C):
    for i, g in enumerate(grad):
        norm_sq = torch.var(g)
        noise_var = (norm_sq/(10**(snr/10))).item()
        sigma = np.sqrt(noise_var)
        # grad[i] = g + torch.randn_like(g) * sigma * C
        grad[i] = g + torch.randn_like(g) * sigma


class GradBuffer(object):
    def __init__(self, grads, mode="copy"):
        self._grad = copy.deepcopy(grads)
        if mode == "zeros":
            for i, grad in enumerate(grads):
                self._grad[i] = torch.zeros_like(grad)
        
    def __add__(self, grad_buffer):
        grads = copy.deepcopy(self._grad)
        for i, grad in enumerate(grads):
            grads[i] = grad.data + grad_buffer._grad[i].data

        return GradBuffer(grads)

    def __sub__(self, grad_buffer):
        grads = copy.deepcopy(self._grad)
        for i, grad in enumerate(grads):
            grads[i] = grad.data - grad_buffer._grad[i].data

        return GradBuffer(grads)

    def __mul__(self, rhs):
        grads = copy.deepcopy(self._grad)
        for i, grad in enumerate(grads):
            grads[i] = grad.data * rhs

        return GradBuffer(grads)