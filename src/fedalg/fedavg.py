import copy

import torch

from src.fedalg import FedAlg
from networks.metanet import MetaNN

class FedSgd(FedAlg):
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


class FedAvg(FedAlg):
    def __init__(self, criterion, model, config):
        super().__init__(criterion, model, half=config.half)
        self.fed_lr = config.fed_lr
        self.tau = config.tau

        self.init_state = copy.deepcopy(self.model.state_dict())
    
    def client_grad(self, x, y):
        net_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.fed_lr)
        for t in range(self.tau):
            out = self.model(x)
            risk = self.criterion(out, y)
            
            net_optimizer.zero_grad()
            risk.backward()
            net_optimizer.step()
            
        grad = []
        st = self.model.state_dict()
        for w_name, w_val in st.items():
            grad.append((self.init_state[w_name] - w_val)/self.fed_lr)

        return grad

