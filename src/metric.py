import cv2
import pandas as pd
import numpy as np

import lpips
import torch
import torchvision.transforms as T

from utils.utils import tensor2img
#from networks.tresnet import TResnetM
#from networks.tresnet.argparser import parse_args

class Metrics:
    def __init__(self, config):
        #args = parse_args()
        #self.init_tresnet(config, args)

        # self.top_k = args.top_k
        # self.th = args.th
        self.half = config.half

        self.device = config.device

    def init_tresnet(self, config, args):
        state = torch.load(config.tresnet)
        
        args.num_classes = state['num_classes']
        self.class_list = np.array(list(state['idx_to_class'].values()))

        df_description = pd.read_csv(args.class_description_path)
        dict_desc = dict(zip(df_description.values[:, 0], df_description.values[:, 1]))
        self.class_tags = [dict_desc[x] for x in self.class_list]

        model_params = {'args': args, 'num_classes': args.num_classes}
        self.tresnet = TResnetM(model_params)
        self.tresnet.to(config.device)

        self.tresnet.load_state_dict(state['model'])
        self.tresnet.eval()
        for p in self.tresnet.parameters():
            p.requires_grad_(False)

    def semantic_labels(self, im1, im2):
        # x = torch.cat((im1, im2), dim=
        res = []

        for x in (im1, im2):
            labels = []
            labels_tag = []
            
            output = torch.squeeze(torch.sigmoid(self.tresnet(x)))
            if len(output.shape) == 1:
                output = output.unsqueeze(0)

            for i in range(output.shape[0]):
                np_output = output[i].cpu().detach().numpy()
                
                idx_sort = np.argsort(-np_output)
                
                detected_classes = np.array(self.class_list)[idx_sort][: self.top_k]
                detected_tags = np.array(self.class_tags)[idx_sort][: self.top_k]
                scores = np_output[idx_sort][: self.top_k]
                # Threshold
                idx_th = scores > self.th
                labels.append(detected_classes[idx_th])
                labels_tag.append(detected_tags[idx_th])
            
            res.append(labels_tag)

        return res

    def evaluate(self, original_img, recon_img, logger):
        if self.half:
            original_img = original_img.to(torch.float32)

        #labels_tag = self.semantic_labels(original_img, recon_img)
        labels = []
        #for i in range(len(labels_tag)):
        #    labels.append(labels_tag[i][0])
        
        lpips_metric = lpips.LPIPS(net='alex').to(self.device)

        avg_psnr = 0
        avg_ssim = 0
        avg_lpips = 0
        avg_jaccard = 0
        # ori_labels_tag, rec_labels_tag = self.semantic_labels(original_img, recon_img)
        batch_size = original_img.shape[0]
        for i in range(batch_size):
            print("i, {:d}".format(i))
            im1 = tensor2img(original_img[i])
            im2 = tensor2img(recon_img[i])
            
            psnr_val = psnr(im1, im2)
            ssim_val = ssim(im1, im2)
            # jaccard_val = jaccard(ori_labels_tag[i], rec_labels_tag[i])
            jaccard_val = 0
            
            avg_psnr += psnr_val
            avg_ssim += ssim_val
            avg_jaccard += jaccard_val
            
            logger.info("{:d} PSNR: {:.3f} SSIM: {:.3f} Jaccard {:.3f}".format(i, psnr_val, ssim_val, jaccard_val))
            
        avg_psnr /= original_img.shape[0]
        avg_ssim /= original_img.shape[0]
        avg_jaccard /= original_img.shape[0]

        avg_lpips = lpips_metric(2*original_img-1, 2*recon_img-1).mean().item()

        return avg_psnr, avg_ssim, avg_jaccard, avg_lpips

def psnr(img1, img2):
    """
    It takes 2 numpy images and then returns the psnr value.
    """
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * np.log10(255.0**2/mse)



def ssim(im1,im2):
    """
    It takes 2 numpy images and then returns the ssim value.
    """
    if len(im1.shape) == 3:
        im1_ycbcr = cv2.cvtColor(im1, cv2.COLOR_RGB2YCrCb)
        im2_ycbcr = cv2.cvtColor(im2, cv2.COLOR_RGB2YCrCb)
        return 0.8*ssim(im1_ycbcr[..., 0], im2_ycbcr[..., 0]) + \
            0.1*ssim(im1_ycbcr[..., 1], im2_ycbcr[..., 1]) + \
            0.1*ssim(im1_ycbcr[..., 2], im2_ycbcr[..., 2])

    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 255
    C1 = (k1*L) ** 2
    C2 = (k2*L) ** 2
    C3 = C2/2
    l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
    ssim_val = l12 * c12 * s12
    
    return ssim_val

def jaccard(a, b):
    intersection = np.intersect1d(a, b)

    # print("-"*10)
    # print(a, b, intersection)
    
    union = a.shape[0] + b.shape[0] - intersection.shape[0]
    return float(intersection.shape[0]) / union


