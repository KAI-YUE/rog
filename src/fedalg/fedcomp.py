import copy

import torch

from src.fedalg import FedAlg

class FedQuant(FedAlg):
    def __init__(self, criterion, model, config):
        super().__init__(criterion, model, half=config.half)
    
    def client_grad(self, x, y):

        out = self.model(x)
        loss = self.criterion(out, y)
        dy_dx = torch.autograd.grad(loss, self.model.parameters())

        if self.half:
            grad = list((_.detach().half().clone() for _ in dy_dx))
        else:
            grad = list((_.detach().clone() for _ in dy_dx))

        return grad


class UniformQuantizer:
    def __init__(self, config):
        self.quantbound = config.quantization_level - 1
        self.debug_mode = config.debug_mode

    def quantize(self, arr):
        """
        quantize a given arr array with unifrom quantization.
        """
        max_val = torch.max(arr.abs())
        sign_arr = arr.sign()
        quantized_arr = (arr/max_val)*self.quantbound
        quantized_arr = torch.abs(quantized_arr)
        quantized_arr = torch.round(quantized_arr).to(torch.int)
        
        quantized_set = dict(max_val=max_val, signs=sign_arr, quantized_arr=quantized_arr)
        return quantized_set
    
    def dequantize(self, quantized_set):
        """
        dequantize a given array which is uniformed quantized.
        """
        coefficients = quantized_set["max_val"]/self.quantbound  * quantized_set["signs"] 
        dequant_arr =  coefficients * quantized_set["quantized_arr"]

        return dequant_arr