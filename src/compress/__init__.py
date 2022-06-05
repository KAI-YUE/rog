from .compressor import *

compress_registry = {
    "uniform": UniformQuantizer,
    "topk": Topk,

    "qsgd": QsgdQuantizer
}