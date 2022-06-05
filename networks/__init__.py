from .resnet import ResNet9, ResNet18
from .vgg import VGG7, Lenet5

nn_registry = {
    "resnet9": ResNet9,
    "resnet18": ResNet18,

    "vgg7": VGG7,
    "vgg7_vb": VGG7,

    "lenet": Lenet5
}