import os
import sys
import logging
import datetime
import random
import numpy as np
import copy
import argparse
from contextlib import suppress

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import hydra
from omegaconf import DictConfig, OmegaConf

import trainer

from tqdm import tqdm 
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import precision_score, accuracy_score
import itertools
class Trainer():
    def __init__(self, conf, rank=0):
        self.conf = copy.deepcopy(conf)
        self.rank = rank
        self.is_master = True if rank == 0 else False
        self.set_env()
        
    def set_env(self):
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(self.rank)


        # mixed precision
        self.amp_autocast = suppress
        if self.conf.base.use_amp is True:
            self.amp_autocast = torch.cuda.amp.autocast
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            
            if self.is_master:
                print(f'[Hyper]: Use Mixed precision - float16')
        else:
            self.scaler = None
        
        # Hyperparameter
        if self.is_master:
            print(f'[Hyper]: learning_rate: {self.conf.hyperparameter.lr} -> {self.conf.hyperparameter.lr * self.conf.base.world_size}')
        # Scheduler
        if self.conf.scheduler.params.get('T_max', None) is None:
            self.conf.scheduler.params.T_max = self.conf.hyperparameter.epochs
        
        # warmup Scheduler
        # if self.conf.scheduler.warmup.get('status', False) is True:
        #     self.conf.scheduler.warmup.params.total_epoch = self.conf.hyperparameter.epochs
        self.start_epoch = 1
    def build_looger(self, is_use:bool):
        if is_use == True: 
            logger = trainer.log.create(self.conf)
            return logger
        else: 
            pass

    def build_model(self, num_classes=-1):
        model = trainer.architecture.create(self.conf.architecture)
        model = model.to(device=self.rank, non_blocking=True)
        model = DDP(model, device_ids=[self.rank], output_device=self.rank)

        return model

    def build_optimizer(self, model):
        optimizer = trainer.optimizer.create(self.conf.optimizer, model)
        return optimizer


    def build_scheduler(self, optimizer):
        scheduler = trainer.scheduler.create(self.conf.scheduler, optimizer)

        return scheduler
    # TODO: modulizaing
    def build_dataloader(self, ):

        """train_loader, train_sampler, channel_length = trainer.dataset.create(
            self.conf.dataset,
            world_size=self.conf.base.world_size,
            local_rank=self.rank,
            mode='train'
        )
        valid_loader, valid_sampler, channel_length = trainer.dataset.create(
            self.conf.dataset,
            world_size=self.conf.base.world_size,
            local_rank=self.rank,
            mode='valid'
        )

        test_loader, test_sampler, channel_length = trainer.dataset.create(
            self.conf.dataset,
            world_size=self.conf.base.world_size,
            local_rank=self.rank,
            mode='train'
        )"""
        train_loader, train_sampler = trainer.new_dataset.create(conf = self.conf.dataset,
        world_size=self.conf.base.world_size,
        local_rank=self.rank,
        mode = 'train')

        valid_loader, valid_sampler = trainer.new_dataset.create(conf = self.conf.dataset,
        world_size=self.conf.base.world_size,
        local_rank=self.rank,
        mode = 'train')

        test_loader, test_sampler = trainer.new_dataset.create(conf = self.conf.dataset,
        world_size=self.conf.base.world_size,
        local_rank=self.rank,
        mode = 'test')


        return train_loader, train_sampler, test_loader, test_sampler

    def build_loss(self):
        criterion = trainer.loss.create(self.conf.loss, self.rank)
        criterion.to(device=self.rank, non_blocking=True)

        return criterion

    def build_saver(self, model, optimizer, scaler):
        saver = trainer.saver.create(self.conf.saver, model, optimizer, scaler)

        return saver
    
    def load_model(self, model, path):
        data = torch.load(path)
        key = 'model' if 'model' in data else 'state_dict'

        if not isinstance(model, (DataParallel, DDP)):
            model.load_state_dict({k.replace('module.', ''): v for k, v in data[key].items()})
        else:
            model.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})
        return model

    def train_one_epoch(self, epoch, model, dl, criterion, optimizer,logger):
        # for step, (image, label) in tqdm(enumerate(dl), total=len(dl), desc="[Train] |{:3d}e".format(epoch), disable=not flags.is_master):
        train_hit = 0
        train_total = 0
        one_epoch_loss = 0
        # 0: train_loss, 1: train_hit, 2: train_total, 3: len(dl)
        counter = torch.zeros((4, ), device=self.rank)
        #torch.set_default_tensor_type(torch.cuda.LONG)
        model.train()
        pbar = tqdm(
            enumerate(dl), 
            bar_format='{desc:<15}{percentage:3.0f}%|{bar:18}{r_bar}', 
            total=len(dl), 
            desc=f"train:{epoch}/{self.conf.hyperparameter.epochs}", 
            disable=not self.is_master
            )
        current_step = epoch
        prediclist = []
        labellist = []
        for step, (image, label) in pbar:
            image= torch.stack(image)
            label= torch.stack(label)
            image = image.to(device=self.rank, non_blocking=True).float()
            label = label.to(device=self.rank, non_blocking=True).float()
            with self.amp_autocast():
                input = image
                y_pred = model(input).squeeze()
                #y_pred = torch.argmax(y_pred, dim=1)[None,:]
                #label = F.one_hot(label.to(torch.int64), num_classes=10)
                label = label.to(torch.int64)
                loss = criterion(y_pred, label).float()
            optimizer.zero_grad(set_to_none=True)
            
            if self.scaler is None:
                loss.backward()
                optimizer.step()
            else:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

            counter[0] += loss.item()
            # _, y_pred = y_pred.unsqueeze(0).max(1)
            y_pred = torch.argmax(y_pred, dim=1)
            counter[1] += y_pred.detach().eq(label).sum()
            counter[2] += image.shape[0]
            
            prediclist.append(y_pred.detach().cpu().numpy())
            labellist.append(label.cpu().numpy())
            if step % 100 == 0:
                pbar.set_postfix({'train_Acc':accuracy_score(label.cpu().numpy(),y_pred.detach().cpu().numpy()>0.6),'train_Loss':round(loss.item(),2)}) 

        counter[3] += len(dl)
        torch.distributed.reduce(counter, 0)
        if self.is_master:
            counter = counter.detach().cpu().numpy()
            labellist = np.array(list(itertools.chain(*labellist)))
            prediclist = np.array(list(itertools.chain(*prediclist)))
            fpr,tpr,thresholds  = metrics.roc_curve(labellist,prediclist,pos_label=1)

            prescore = precision_score(labellist,prediclist > 0.6)
            acccore = accuracy_score(labellist,prediclist > 0.6)
            # print(f'[Train_{epoch}] Acc: {train_hit / train_total} Loss: {one_epoch_loss / len(dl)}')
            metric = {'AUROC':metrics.auc(fpr, tpr),'Acc': acccore,'pre':prescore, 'Loss': counter[0] / counter[3],'optimizer':optimizer}
            logger.update_log(metric,current_step,'train') # update logger step
            logger.update_histogram(model,current_step,'train') # update weight histogram 
            logger.update_image(image,current_step,'train') # update transpose image
            logger.update_metric(labellist,prediclist,current_step,'train')
        # return loss, accuracy
        return counter[0] / counter[3], counter[1] / counter[2], dl


    @torch.no_grad()
    def eval(self, epoch, model, dl, criterion,logger):
        # 0: val_loss, 1: val_hit, 2: val_total, 3: len(dl)
        counter = torch.zeros((4, ), device=self.rank)
        model.eval()
        pbar = tqdm(
            enumerate(dl),
            bar_format='{desc:<15}{percentage:3.0f}%|{bar:18}{r_bar}', 
            total=len(dl),
            desc=f"val  :{epoch}/{self.conf.hyperparameter.epochs}", 
            disable=not self.is_master
            ) # set progress bar
        current_step = epoch
        prediclist = []
        labellist = []

        for step, (image, label) in pbar:
            # current_step = epoch*len(dl)+step
            
            image = image.to(device=self.rank, non_blocking=True).float()
            label = label.to(device=self.rank, non_blocking=True).float()
            #seg_array = seg_array.to(device=self.rank, non_blocking=True).float()
            with self.amp_autocast():
                #input = torch.cat((image, seg_array),1)
                input = image
                y_pred = model(input).squeeze()
                loss = criterion(y_pred, label).float()
            counter[0] += loss.item()
            # _, y_pred = y_pred.unsqueeze(0).max(1)
            y_pred_copy = torch.tensor(y_pred)
            if self.conf.loss.type == 'bce':
                y_pred_copy = torch.round(torch.sigmoid(y_pred_copy))
            else:
                _, y_pred_copy = y_pred_copy.max(1)
                # one_hot encoding
                if len(list(label.shape)) > 2:
                    _, label = label.max(1)
            counter[1] += y_pred_copy.detach().eq(label).sum()
            counter[2] += image.shape[0]
            prediclist.append(y_pred.detach().cpu().numpy())
            labellist.append(label.cpu().numpy())

            if step % 100 == 0:
                pbar.set_postfix({'valid_Acc':accuracy_score(label.cpu().numpy(),y_pred.detach().cpu().numpy()>0.6),'valid_Loss': round(loss.item(), 2)}) 
        counter[3] += len(dl)
        torch.distributed.reduce(counter, 0)
        if self.is_master:
            counter = counter.detach().cpu().numpy()
            labellist = np.array(list(itertools.chain(*labellist)))
            prediclist = np.array(list(itertools.chain(*prediclist)))
            fpr,tpr,thresholds  = metrics.roc_curve(labellist,prediclist,pos_label=1)
            prescore = precision_score(labellist,prediclist > 0.6)
            acccore = accuracy_score(labellist,prediclist>0.6)

            # print(f'[Val_{epoch}] Acc: {counter[1] / counter[2]} Loss: {counter[0] / counter[3]}')
            # metric = {'Acc':counter[1] / counter[2], 'Loss': counter[0] / counter[3]}
            metric = {'AUROC':metrics.auc(fpr, tpr),'Acc': acccore,'pre':prescore, 'Loss': counter[0] / counter[3]}
            logger.update_log(metric,current_step,'valid') # update logger step
            logger.update_histogram(model,current_step,'valid') # add image 
            # logger.update_image(image,current_step,'valid') # update transpose image
            # y_pred_ = np.array(predic_metric)[:,0].flatten()
            # label_ = np.array(predic_metric)[:,1].flatten()
            logger.update_metric(labellist,prediclist,current_step,'valid') # update transpose image
            
        return counter[0] / counter[3], counter[1] / counter[2]

    def train_eval(self):
        model = self.build_model()
        criterion = self.build_loss()
        optimizer = self.build_optimizer(model)

        scheduler = self.build_scheduler(optimizer)
        train_dl, train_sampler,test_dl, test_sampler= self.build_dataloader()

        logger = self.build_looger(is_use=self.is_master)
        saver = self.build_saver(model, optimizer, self.scaler)
        # Wrap the model
        
        # initialize
        for name, x in model.state_dict().items():
            dist.broadcast(x, 0)
        torch.cuda.synchronize()

        # add graph to tensorboard
        if logger is not None:
            logger.update_graph(model, torch.rand((1,1,28,28)).float())

        # load checkpoint
        if self.conf.base.resume == True:
            self.start_epoch = saver.load_for_training(model,optimizer,self.rank,scaler=None)
        
        for epoch in range(self.start_epoch, self.conf.hyperparameter.epochs + 1):
            train_sampler.set_epoch(epoch)
            # train
            train_loss, train_acc, train_dl = self.train_one_epoch(epoch, model, train_dl, criterion, optimizer, logger)
            scheduler.step()

            # eval
            valid_loss, valid_acc = self.eval(epoch, model, valid_dl, criterion, logger)
            
            torch.cuda.synchronize()

            # save_model
            saver.save_checkpoint(epoch=epoch, model=model, loss=train_loss, rank=self.rank, metric=valid_acc)

            if self.is_master:
                print(f'Epoch {epoch}/{self.conf.hyperparameter.epochs} - train_Acc: {train_acc:.3f}, train_Loss: {train_loss:.3f}, valid_Acc: {valid_acc:.3f}, valid_Loss: {valid_loss:.3f}')

    def run(self):
        if self.conf.base.mode == 'train':
            pass
        elif self.conf.base.mode == 'train_eval':
            self.train_eval()
        elif self.conf.base.mode == 'finetuning':
            pass


def set_seed(conf):
    if conf.base.seed is not None:
        conf.base.seed = int(conf.base.seed, 0)
        print(f'[Seed] :{conf.base.seed}')
        os.environ['PYTHONHASHSEED'] = str(conf.base.seed)
        random.seed(conf.base.seed)
        np.random.seed(conf.base.seed)
        torch.manual_seed(conf.base.seed)
        torch.cuda.manual_seed(conf.base.seed)
        torch.cuda.manual_seed_all(conf.base.seed)  # if use multi-G
        torch.backends.cudnn.deterministic = True


def runner(rank, conf):
    # Set Seed
    set_seed(conf)

    os.environ['MASTER_ADDR'] = conf.MASTER_ADDR
    os.environ['MASTER_PORT'] = conf.MASTER_PORT

    print(f'Starting train method on rank: {rank}')
    dist.init_process_group(
        backend='nccl', world_size=conf.base.world_size, init_method='env://',
        rank=rank
    )
    trainer = Trainer(conf, rank)
    trainer.run()



@hydra.main(config_path='conf', config_name='mine')
def main(conf: DictConfig) -> None:
    print(f'Configuration\n{OmegaConf.to_yaml(conf)}')
    
    mp.spawn(runner, nprocs=conf.base.world_size, args=(conf, ))
    

if __name__ == '__main__':
    main()

