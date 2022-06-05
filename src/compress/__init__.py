from .compressor import *

compress_registry = {
    "sign": SignSGDCompressor,
    "uniform": UniformQuantizer,
    "topk": Topk,

    "qsgd": QsgdQuantizer
}