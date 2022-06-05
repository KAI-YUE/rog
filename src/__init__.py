from src.fedalg.fedavg import FedAvg, FedSgd
from src.fedalg.fedcdp import FedCdp

fedlearning_registry = {
    "fedsgd":   FedSgd,
    "fedcdp":   FedCdp, 
    "fedavg":   FedAvg
}