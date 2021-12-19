import torch.nn as nn
import logging
import segmentation_models_pytorch as smp

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
    elif conf['type'] == 'jaccard':
        criterion = smp.losses.JaccardLoss(**conf['params'])
    elif conf['type'] == 'jaccard_no_logits':
        criterion = smp.losses.JaccardLoss(mode='binary', from_logits=False)
    elif conf['type'] == 'dice':
        criterion = smp.losses.DiceLoss(mode='binary', from_logits=False)
    else:
        raise AttributeError(f'not support loss config: {conf}')

    return criterion