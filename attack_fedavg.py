import torch
from main import main

if __name__ == '__main__':
    torch.manual_seed(0)
    main("config_fedavg.yaml")