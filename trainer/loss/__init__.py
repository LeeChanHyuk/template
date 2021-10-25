import torch.nn as nn
import logging

LOGGER = logging.getLogger(__name__)

def create(conf, rank):

    if conf['type'] == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif conf['type'] ==  'bce': 
        criterion = nn.BCEWithLogitsLoss()
    elif conf['type'] ==  'sL1loss': 
        criterion = nn.SmoothL1Loss()
    elif conf['type'] ==  'MAE': 
        criterion = nn.L1Loss()
    elif conf['type'] ==  'MSE': 
        criterion = nn.MSELoss()
    else:
        raise AttributeError(f'not support loss config: {conf}')

    return criterion