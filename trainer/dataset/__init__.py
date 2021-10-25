import torch
import torchvision
import torchvision.transforms as transforms
import logging
import os
import pandas as pd
from .custom_dataset import custom_dataset
from .custom_25d_dataset import custom_25d_dataset
from .data_utils import preprocess

LOGGER = logging.getLogger(__name__)


def create(conf, world_size=1, local_rank=-1, mode='train'):

    conf = conf[mode]
    transformers = transforms.Compose([preprocess(t) for t in conf['preprocess']] )
    
    if conf['name'] == 'mnist':
        dataset = torchvision.datasets.FashionMNIST(
            root='../', 
            download=True, 
            transform=transformers, 
            train=(mode == 'train')
            )
    elif conf['name'] == 'custom_dataset':
        data_directory = conf["data_directory"]
        flag_index = conf["flag_index"]
        df = pd.read_csv(data_directory)
        df # filtering small image
        if mode != "test":
            # if we were to divide dataset into K-fold and use 0~K-2 fold for train dataset,
            # the flag_index must be list of indices [0, 1, ..., K-2]
            if isinstance(flag_index, list) or isinstance(flag_index, tuple):
                df = df[df["flag_index"].isin(flag_index)]
            else:
                df = df[df["flag_index"] == flag_index]

            targets = df[conf["target_column"]]
        else:
            targets = None

        input_paths = df[conf["input_column"]]
        dataset = custom_dataset(input_paths=input_paths,
                                 targets=targets,
                                 mode=mode,
                                 transform=transformers,
                                 conf=conf
                                 )

    elif conf['name'] == 'custom_25d_dataset': 
        label_dir = conf['label_dir']
        if conf['dicom_train']:
            label_dir_name, extension = label_dir.split('.')
            label_dir_name += '_dicom'
            label_dir = label_dir_name + '.' + extension
        df = pd.read_csv(label_dir)
        
        # df = df[~df[conf['patient_id']].isin([109,123,709])]
        filerlist = [100, 102, 105, 108, 109, 112, 113, 123, 124, 139, 148, 149, 151,
       443, 459, 571, 584, 589, 709, 818]
        #df = df[~df[conf['patient_id']].isin(filerlist)]
        if mode != "test":
            flag_index = conf["flag_index"]    
            # if we were to divide dataset into K-fold and use 0~K-2 fold for train dataset,
            # the flag_index must be list of indices [0, 1, ..., K-2]
            # if isinstance(flag_index, list) or isinstance(flag_index, tuple):
            #     print(df["flag_index"],'123123123')
            #     df = df[df["flag_index"].isin(flag_index)]
            #     print(df)
            # else:
            #df = df[df["flag_index"] == flag_index]
            df = df[df["flag_index"].isin(flag_index)]
            targets = df[conf["label_name"]].to_numpy()
        else:
            targets = None
        input_paths = df[conf["patient_id"]]
        
        input_paths = [f'{conf["data_path"]}/{str(i).zfill(5)}' for i in input_paths]

        # input_paths = list(map(lambda x:f'{conf["data_path"]}/{x}'),input_paths)
        dataset = custom_25d_dataset(input_paths=input_paths,
                                 targets=targets,
                                 mode=mode,
                                 transform=transformers,
                                 conf=conf,
                                 local_rank=local_rank
                                 )
        

    else:
        raise AttributeError(f'not support dataset config: {conf}')
    
    sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, 
            num_replicas=world_size, 
            rank=local_rank,
            shuffle=(mode == 'train')
        )
    
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        pin_memory=True,
        drop_last=conf['drop_last'],
        num_workers=0,
        sampler=sampler
    )
    print(conf['batch_size'])
    
    return dl, sampler, dataset.channel_length_call()