# init에서는 dataset을 create하는 함수를 만들어야지.
# init에서 create가 들어가는 이유는 만들고자하는 dataset이 하나가 아니라 여러 개 일 때, 여기서 총체적으로 관리할 수 있기 때문이다.
# 즉, dataset을 만들 때 필요한 정보 관리 등이나 인자들을 여기서 만들어서 전달하자.

import logging
from .ch_dataset import ch_dataset

import torch.utils.data
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split

def create(conf, local_rank, world_size, mode='train'):

    if conf[mode]['name'] == 'dataset':
        if mode == 'train' or mode == 'valid':
            dataset = ch_dataset(conf, mode)
            length = len(dataset)
            train_length = int(length * 0.8)
            valid_length = length - train_length
            train_dataset, valid_dataset = random_split(dataset, [train_length, valid_length])
            if mode == 'train':
                temp_dataset = train_dataset
            elif mode == 'valid':
                temp_dataset = valid_dataset
        else:
            dataset = ch_dataset(conf, mode)
            temp_dataset = dataset
                
    elif conf[mode]['name'] == 'mnist':
        if mode == 'train':
            temp_dataset = torchvision.datasets.MNIST(root='MNIST_data/',
            train=True,
            transform=transforms .ToTensor(),
            download=True)
        else:
            temp_dataset = torchvision.datasets.MNIST(root='MNIST_data/',
            train=False,
            transform=transforms.ToTensor(),
            download=True)
        
    sampler = torch.utils.data.distributed.DistributedSampler(
            temp_dataset, 
            num_replicas=world_size, 
            rank=local_rank,
            shuffle=(mode == 'train')
        )
    dataloader = torch.utils.data.DataLoader(
        temp_dataset,
        batch_size=conf[mode]['batch_size'],
        shuffle=False,
        pin_memory=True,
        drop_last=conf[mode]['drop_last'],
        num_workers=0,
        sampler=sampler
    )

    return dataloader, sampler
