import pickle

import torch

from networks import nn_registry
from src.metric import Metrics
from src.dataloader import fetch_trainloader
from src import fedlearning_registry
from src.attack import Attacker, grad_inv
from src.compress import compress_registry
from utils import *

def main(config_file):
    config = load_config(config_file)
    output_dir = init_outputfolder(config)
    logger = init_logger(config, output_dir)

    # Load dataset and fetch the data
    train_loader = fetch_trainloader(config, shuffle=True)

    for batch_idx, (x, y) in enumerate(train_loader):
        if batch_idx == 0:
            break

    criterion = cross_entropy_for_onehot
    model = nn_registry[config.model](config)

    onehot = label_to_onehot(y, num_classes=config.num_classes)
    x, y, onehot, model = preprocess(config, x, y, onehot, model)

    # federated learning algorithm on a single device
    fedalg = fedlearning_registry[config.fedalg](criterion, model, config) 
    grad = fedalg.client_grad(x, onehot)

    # gradient postprocessing
    if config.compress != "none":
        compressor = compress_registry[config.compress](config)
        for i, g in enumerate(grad):
            compressed_res = compressor.compress(g)
            grad[i] = compressor.decompress(compressed_res)

    # initialize an attacker and perform the attack 
    attacker = Attacker(config, criterion)
    attacker.init_attacker_models(config)
    recon_data = grad_inv(attacker, grad, x, onehot, model, config, logger)

    synth_data, recon_data = attacker.joint_postprocess(recon_data, y)        
    # recon_data = synth_data

    # Report the result first 
    logger.info("=== Evaluate the performance ====")
    metrics = Metrics(config)
    snr, ssim, jaccard, lpips = metrics.evaluate(x, recon_data, logger)
    
    logger.info("PSNR: {:.3f} SSIM: {:.3f} Jaccard {:.3f} Lpips {:.3f}".format(snr, ssim, jaccard, lpips))

    save_batch(output_dir, x, recon_data)

    record = {"snr":snr, "ssim":ssim, "jaccard":jaccard, "lpips":lpips}
    with open(os.path.join(output_dir, config.fedalg+".dat"), "wb") as fp:
        pickle.dump(record, fp)

if __name__ == '__main__':
    torch.manual_seed(0)
    main("config.yaml")

