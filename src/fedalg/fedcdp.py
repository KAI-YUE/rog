import copy
import numpy as np

import torch

from src.fedalg import FedAlg

class FedCdp(FedAlg):
    """
    coming with opacus
    """
    pass

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