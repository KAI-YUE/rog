import copy
import numpy as np
from scipy.io import loadmat
from collections import OrderedDict

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

from utils.utils import Adam16
from utils import single2tensor4
from networks.usrnet import USRNet
from networks.discriminator import Discriminator
from networks.generator import Ccgenerator, Generator
from networks.metanet import MetaNN


class Attacker:
    def __init__(self, config, criterion):
        """
        Initialize the attacker. 
        """
        self.sf = 4
        self.alpha_tv = 3.e-7
        self.T_max = config.T_max
        self.half = config.half
        
        self.device = config.device
        self.sample_size = config.sample_size
        self.eta = config.rog_lr
        self.printevery = config.printevery
        self.criterion = criterion

        self.device = config.device
        self.fed_lr = config.fed_lr
        self.resize = T.Resize((config.sample_size[0], config.sample_size[1]))

    def init_attacker_models(self, config):
        """
        Load post-processing models. 
        """
        # conditional generator
        self.postmodel_dir = config.joint_postmodel
        self.generator = Ccgenerator()
        state_dict = torch.load(config.joint_postmodel)
        self.generator.load_state_dict(state_dict["gen_state_dict"])
        self.generator.eval()
        for p in self.generator.parameters():
            p.requires_grad = False

        self.generator.to(self.device)

        # denoising block
        self.denoiser = Generator()
        state_dict = torch.load(config.denoiser)
        self.denoiser.load_state_dict(state_dict["gen_state_dict"])
        self.denoiser.eval()
        for p in self.denoiser.parameters():
            p.requires_grad = False

        self.denoiser.to(self.device)

        # super resolution block
        self.usrnet = USRNet(n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
                                nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
        state_dict = torch.load(config.usrnet)
        self.usrnet.load_state_dict(state_dict)
        self.usrnet.eval()
        for p in self.usrnet.parameters():
            p.requires_grad = False

        self.usrnet.to(self.device)
        
        kernels = loadmat(config.kernel)['kernels']
        kernel = kernels[0, self.sf-2].astype(np.float64)
        self.kernel = single2tensor4(kernel[..., np.newaxis])

        self.sigma = torch.tensor(config.noise_level).float().view([1, 1, 1, 1])

    def free_models(self):
        """
        Remove all post-processing models in the memory.
        """
        self.generator = None
        self.denoiser = None
        self.usrnet = None
        self.kernel = None
        self.sigma = None

        torch.cuda.empty_cache()

    def init_classifier(self):
        """
        Initialize the imagenet classifier
        """
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        self.transform = T.Compose([normalize, T.Resize(256)])

        efficientnet_b7 = models.efficientnet_b7(pretrained=True)
        efficientnet_b7.eval()
        for p in efficientnet_b7.parameters():
            p.requires_grad_(False)

        efficientnet_b7.to(self.device)
        self.classifier = efficientnet_b7

    def init_discriminator(self):
        """
        Initialize the discriminator
        """
        self.discriminator = Discriminator()

        state = torch.load(self.postmodel_dir)
        self.discriminator.load_state_dict(state["dis_state_dict"])

        self.discriminator.to(self.device)

    def grad_inv(self, grad, x, onehot, model, logger=None):
        """
        Perform gradient inversion attack.
        """

        sample_size = self.sample_size
        latent_data = np.random.rand(x.shape[0], 3, int(sample_size[0]/self.sf), int(sample_size[1]/self.sf))

        if self.half:
            latent_data = latent_data.astype(np.float16)
        else:
            latent_data = latent_data.astype(np.float32)

        dummy_data = torch.from_numpy(latent_data).to(self.device).requires_grad_(True)
        
        if self.half:
            dummy_data_optimizer = Adam16([dummy_data], lr=self.eta)
        else:
            dummy_data_optimizer = optim.Adam([dummy_data], lr=self.eta)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(dummy_data_optimizer, T_max=self.T_max, eta_min=0.1*self.eta)

        normal_func = lambda x: x

        # initialize a list to store the loss

        for iters in range(self.T_max):
            def closure():
                dummy_data_optimizer.zero_grad()

                pred = model(F.interpolate(dummy_data, scale_factor=self.sf, mode='bicubic'))
                dummy_loss = self.criterion(pred, onehot) 
                dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, grad):
                    grad_diff += ((normal_func(gx) - gy) ** 2).sum()

                grad_diff.backward()
                
                return grad_diff
        
            dummy_data_optimizer.step(closure)
            
            if iters % self.printevery == 0: 
                current_loss = closure()
                logger.info("iter: {:d} loss: {:.4e}".format(iters, current_loss.item()))
                
            if iters > 0:
                scheduler.step()

        # return the float32 tensor
        if self.half:
            dummy_data = dummy_data.to(torch.float32)

        return dummy_data.data

    def multistep_attack(self, grad, x, y, model, tau=1, logger=None):
        """
        Perform a multi-step gradient inversion attack.
        """

        sample_size = self.sample_size
        latent_data = np.random.rand(x.shape[0], 3, int(sample_size[0]/self.sf), int(sample_size[1]/self.sf))

        if self.half:
            latent_data = latent_data.astype(np.float16)
        else:
            latent_data = latent_data.astype(np.float32)

        dummy_data = torch.from_numpy(latent_data).to(self.device).requires_grad_(True)
        
        if self.half:
            dummy_data_optimizer = Adam16([dummy_data], lr=self.eta)
        else:
            dummy_data_optimizer = optim.Adam([dummy_data], lr=self.eta)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(dummy_data_optimizer, T_max=self.T_max, eta_min=0.1*self.eta)

        for iters in range(self.T_max):
            def closure():
                dummy_data_optimizer.zero_grad()

                meta_net = MetaNN(model)
                net0 = copy.deepcopy(OrderedDict(model.named_parameters()))

                for i in range(tau):
                    pred = meta_net(F.interpolate(dummy_data, scale_factor=self.sf, mode='bicubic'))
                    dummy_loss = self.criterion(pred, y)
                    g = torch.autograd.grad(dummy_loss, meta_net.parameters.values(),
                                retain_graph=True, create_graph=True, only_inputs=True)

                    meta_net.parameters = OrderedDict((name, param - self.fed_lr * grad_part)
                                            for ((name, param), grad_part)
                                            in zip(meta_net.parameters.items(), g))

                meta_net.parameters = OrderedDict((name, (param_origin - param)/self.fed_lr)
                                    for ((name, param), (name_origin, param_origin))
                                    in zip(meta_net.parameters.items(), net0.items()))
                
                dummy_dy_dx = list(meta_net.parameters.values())
                
                grad_diff = 0
                counter = 0
                for gx, gy in zip(dummy_dy_dx, grad):
                    counter += 1
                    grad_diff += ((gx - gy) ** 2).sum()
                
                grad_diff.backward()

                return grad_diff
    
            dummy_data_optimizer.step(closure)

            if iters % self.printevery == 0:
                current_loss = closure()
                logger.info("iter: {:d} loss: {:.4e}".format(iters, current_loss.item()))
                
            if iters > 0:
                scheduler.step()

        if self.half:
            dummy_data = dummy_data.to(torch.float32)

        return dummy_data.data


    def joint_postprocess(self, x, y):
        with torch.no_grad():
            refine1 = self.generator(self.resize(x), y)
            refine2 = self.denoiser(x)

            kernel = self.kernel.repeat(x.shape[0], 1, 1, 1)
            sigma = self.sigma.repeat(x.shape[0], 1, 1, 1)
            [kernel, sigma] = [el.to(self.device) for el in [kernel, sigma]]            
            refine2 = self.usrnet(refine2, kernel, self.sf, sigma)

        return refine1, 0.5*refine2 + 0.5*refine1

    def refine(self, x, y):
        pass

def grad_inv(attacker, grad, x, onehot, model, config, logger):
    if config.fedalg != "fedavg":
        dummy_data = attacker.grad_inv(grad, x, onehot, model, logger)
    else:
        dummy_data = attacker.multistep_attack(grad, x, onehot, model, config.tau, logger)

    return dummy_data
