from abc import ABC, abstractmethod

import torch

class FedAlg(ABC):
    def __init__(self, criterion, model, half=False):
        self.criterion = criterion
        self.model = model
        self.half = half

    @abstractmethod
    def client_grad(self, x, y):
        """
        Compute the gradient of the loss function on the client side.
        """
