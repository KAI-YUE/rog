import yaml
import json
import os
import logging
import numpy as np

def load_config(filename=None):
    """Load configurations of yaml file"""
    current_path = os.path.dirname(__file__)

    if filename is None:
        filename = "config.yaml"

    with open(os.path.join(current_path, filename), "r") as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    # Empty class for yaml loading
    class cfg: pass
    
    for key in config:
        setattr(cfg, key, config[key])
    
    if not hasattr(cfg, "model"):
        cfg.model = ""

    return cfg


def init_logger(config, output_dir):
    """Initialize a logger object. 
    """
    log_level = "INFO"
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    fh = logging.FileHandler(os.path.join(output_dir, "main.log"))
    fh.setLevel(log_level)
    sh = logging.StreamHandler()
    sh.setLevel(log_level)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("-"*80)

    attributes = filter(lambda a: not a.startswith('__'), dir(config))
    for attr in attributes:
        logger.info("{:<20}: {}".format(attr, getattr(config, attr)))

    return logger
