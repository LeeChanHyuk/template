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
from torch.utils import data
from torch.utils.tensorboard.writer import SummaryWriter

import torchvision
import torchvision.transforms as transforms

import hydra
from omegaconf import DictConfig, OmegaConf

import trainer

from tqdm import tnrange, tqdm 
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import precision_score, accuracy_score
import itertools
class Trainer():
    def __init__(self, conf, rank=0):
        self.conf = copy.deepcopy(conf)
        self.rank = rank
        self.is_master = True if rank == 0 else False
        self.writer = SummaryWriter()
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
        model = DDP(model, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True)

        return model

    def build_optimizer(self, model):
        optimizer = trainer.optimizer.create(self.conf.optimizer, model)
        return optimizer


    def build_scheduler(self, optimizer):
        scheduler = trainer.scheduler.create(self.conf.scheduler, optimizer)

        return scheduler
    # TODO: modulizaing
    def build_dataloader(self, ):

        train_loader, train_sampler = trainer.new_dataset.create(conf = self.conf.dataset,
        world_size=self.conf.base.world_size,
        local_rank=self.rank,
        mode = 'train')

        valid_loader, valid_sampler = trainer.new_dataset.create(conf = self.conf.dataset,
        world_size=self.conf.base.world_size,
        local_rank=self.rank,
        mode = 'valid')

        test_loader, test_sampler = trainer.new_dataset.create(conf = self.conf.dataset,
        world_size=self.conf.base.world_size,
        local_rank=self.rank,
        mode = 'test')

        return train_loader, train_sampler, valid_loader, valid_sampler, test_loader, test_sampler

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
    
    def evaluation_for_semantic_segmentation(self, y_preds, labels, thresh=0.5):
        y_preds = y_preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        accuracies = []
        precisions = []
        recalls = []
        for y_pred, label in zip(y_preds, labels):
            # thresholding
            y_pred[y_pred > thresh] = 1
            y_pred[y_pred <= thresh] = 0
            # label distinction
            label_1 = (label == 1)
            label_0 = (label == 0)
            # TP, TN, FP, FN
            TP = np.sum((y_pred == 1) * (label == 1)) 
            TN = np.sum((y_pred == 0) * (label == 0))
            FP = np.sum((y_pred == 1) * (label == 0))
            FN = np.sum((y_pred == 0) * (label == 1))
            # Calculate
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            precision = TP / (TP + FP)
            recall = (TP) / (TP + FN)
            # save
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
        return accuracies, precisions, recalls
    
    def evaluation_for_multi_class_semantic_segmentation(self, y_preds, labels):
        y_preds = y_preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        accuracies = []
        precisions = []
        recalls = []
        for y_pred, label in zip(y_preds, labels):
            # thresholding
            y_pred = np.rint(y_pred)
            # label distinction
            label = np.rint(label)
            # TP, TN, FP, FN
            TP = np.sum((y_pred == 1) * (label == 1)) 
            TN = np.sum((y_pred == 0) * (label == 0))
            FP = np.sum((y_pred == 1) * (label == 0))
            FN = np.sum((y_pred == 0) * (label == 1))
            # Calculate
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            precision = TP / (TP + FP)
            recall = (TP) / (TP + FN)
            # save
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
        return accuracies, precisions, recalls

    def train_one_epoch(self, epoch, model, dl, criterion, optimizer,logger):
        # for step, (image, label) in tqdm(enumerate(dl), total=len(dl), desc="[Train] |{:3d}e".format(epoch), disable=not flags.is_master):
        train_hit = 0
        train_total = 0
        one_epoch_loss = 0
        # eval_result = [accuracy, precision, recall, loss, image_num]
        t_acc = np.zeros(1)
        t_recall = np.zeros(1)
        t_precision = np.zeros(1)
        t_loss = np.zeros(1)
        t_imgnum = np.zeros(1)

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
        for step, (image, label) in pbar:
            #image= torch.stack(image)
            #label= torch.stack(label)
            image = image.to(device=self.rank, non_blocking=True).float()
            label = label.to(device=self.rank, non_blocking=True).float()
            with self.amp_autocast():
                input = image
                y_pred = model(input).squeeze()
                logit = torch.argmax(y_pred, dim=1)
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
            accuracies, precisions, recalls = self.evaluation_for_semantic_segmentation(y_pred, label)
            temp_acc, temp_recall, temp_precision, temp_imgnum = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
            for i in range(image.shape[0]):
                temp_acc += accuracies[i]
                temp_recall += recalls[i]
                temp_precision += precisions[i]
                t_acc += accuracies[i]
                t_recall += recalls[i]
                t_precision += precisions[i]
            t_imgnum += image.shape[0]
            t_loss += loss.item()
            temp_imgnum += image.shape[0]
            t_loss += loss.item()
            if step % 100 == 0:
                pbar.set_postfix({'train_Acc':temp_acc / temp_imgnum,'train_Loss':round(loss.item(),2)}) 
        
        #torch.distributed.reduce(counter, 0)
        if self.is_master:
            #counter = counter.detach().cpu().numpy()
            metric = {'Acc': t_acc / t_imgnum,'pre': t_precision / t_imgnum, 'recall':t_recall/t_imgnum, 'Loss': t_loss / t_imgnum,'optimizer':optimizer}
            self.writer.add_scalar("Loss/train", t_loss / t_imgnum, epoch)
            self.writer.add_scalar("ACC/train", t_acc / t_imgnum, epoch)
            self.writer.add_scalar("Recall/train", t_recall / t_imgnum, epoch)
            self.writer.add_scalar("Precision/train", t_precision / t_imgnum, epoch)
            #logger.update_log(metric,current_step,'train') # update logger step
            #logger.update_histogram(model,current_step,'train') # update weight histogram 
            #logger.update_image(image,current_step,'train') # update transpose image
            #logger.update_metric()
            self.writer.flush()
        # return loss, accuracy
        return t_loss / t_imgnum, t_acc / t_imgnum, dl


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
        t_acc = np.zeros(1)
        t_recall = np.zeros(1)
        t_precision = np.zeros(1)
        t_loss = np.zeros(1)
        t_imgnum = np.zeros(1)

        for step, (image, label) in pbar:
            # current_step = epoch*len(dl)+step
            #image= torch.stack(image)
            #label= torch.stack(label)
            image = image.to(device=self.rank, non_blocking=True).float()
            label = label.to(device=self.rank, non_blocking=True).float()
            #seg_array = seg_array.to(device=self.rank, non_blocking=True).float()
            with self.amp_autocast():
                input = image
                y_pred = model(input).squeeze()
                logit = torch.argmax(y_pred, dim=1)
                label = label.to(torch.int64)
                loss = criterion(y_pred, label).float()
            accuracies, precisions, recalls = self.evaluation_for_semantic_segmentation(y_pred, label)
            temp_acc, temp_recall, temp_precision, temp_imgnum = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
            for i in range(image.shape[0]):
                temp_acc += accuracies[i]
                temp_recall += recalls[i]
                temp_precision += precisions[i]
                t_acc += accuracies[i]
                t_recall += recalls[i]
                t_precision += precisions[i]
            t_imgnum += image.shape[0]
            t_loss += loss.item()
            temp_imgnum += image.shape[0]
            t_loss += loss.item()
            if step % 100 == 0:
                pbar.set_postfix({'Valid_Acc':temp_acc / temp_imgnum,'Valid_Loss':round(loss.item(),2)}) 
        counter[3] += len(dl)
        torch.distributed.reduce(counter, 0)
        if self.is_master:
            self.writer.add_scalar("Loss/valid", t_loss / t_imgnum, epoch)
            self.writer.add_scalar("ACC/valid", t_acc / t_imgnum, epoch)
            self.writer.add_scalar("Recall/valid", t_recall / t_imgnum, epoch)
            self.writer.add_scalar("Precision/valid", t_precision / t_imgnum, epoch)
            #logger.update_log(metric,current_step,'train') # update logger step
            #logger.update_histogram(model,current_step,'train') # update weight histogram 
            #logger.update_image(image,current_step,'train') # update transpose image
            #logger.update_metric()
            self.writer.flush()
            
        return t_loss / t_imgnum, t_acc / t_imgnum

    def train_eval(self):
        model = self.build_model()
        criterion = self.build_loss()
        optimizer = self.build_optimizer(model)

        scheduler = self.build_scheduler(optimizer)
        train_dl, train_sampler,valid_dl, valid_sampler, test_dl, test_sampler= self.build_dataloader()

        logger = self.build_looger(is_use=self.is_master)
        saver = self.build_saver(model, optimizer, self.scaler)
        # Wrap the model
        
        # initialize
        for name, x in model.state_dict().items():
            dist.broadcast(x, 0)
        torch.cuda.synchronize()

        # add graph to tensorboard
        #if logger is not None:
            #logger.update_graph(model, torch.rand((1,1,28,28)).float())

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
                print(f'Epoch {epoch}/{self.conf.hyperparameter.epochs} - train_Acc: {train_acc[0]:.3f}, train_Loss: {train_loss[0]:.3f}, valid_Acc: {valid_acc[0]:.3f}, valid_Loss: {valid_loss[0]:.3f}')

    def test_sample_visualization(self, y_pred, label, num, thresh=0.5):
        y_pred = y_pred.detach().cpu().numpy()[0]
        label = label.detach().cpu().numpy()[0]
        y_pred[y_pred > thresh] = 1
        y_pred[y_pred <= thresh] = 0
        #img_grid = torchvision.utils.make_grid(y_pred)
        #label_grid = torchvision.utils.make_grid(label)
        self.writer.add_image(str(num) + '/prediction', y_pred, global_step=25, dataformats='HW')
        self.writer.add_image(str(num) + '/label', label, global_step=25, dataformats='HW')
        self.writer.flush()


    def test_multiple_sample_visualization(self, y_pred, label, num, thresh=0.5):
        y_pred = torch.argmax(y_pred, dim=0)
        y_pred = y_pred.detach().cpu().numpy()[0]
        label = label.detach().cpu().numpy()[0]
        y_pred = np.rint(y_pred)
        colors = []
        for i in range(255):
            if i == 0:
                colors.append((0,0,0))
                continue
            random.seed(i)
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            colors.append((r, g, b))
        y_pred_new = np.zeros((y_pred.shape[0], y_pred.shape[1], 3))
        for i in range(y_pred.shape[0]):
            for j in range(y_pred.shape[1]):
                value = int(y_pred[i, j])
                y_pred_new[i,j,0] = colors[value][0]
                y_pred_new[i,j,1] = colors[value][1]
                y_pred_new[i,j,2] = colors[value][2]
        #img_grid = torchvision.utils.make_grid(y_pred)
        #label_grid = torchvision.utils.make_grid(label)
        self.writer.add_image(str(num) + '/prediction', y_pred_new, global_step=25, dataformats='HWC')
        self.writer.add_image(str(num) + '/label', label, global_step=25, dataformats='HW')
        self.writer.flush()


    def test(self):
        # settings
        model = self.build_model()
        optimizer = self.build_optimizer(model)
        saver = self.build_saver(model, optimizer, self.scaler)
        checkpoint_path = '/home/ddl/git/template/outputs/2021-12-13/multi_class_semantic_segmetation_2/checkpoint/last_checkpoint_epoch_124.pth.tar'
        saver.load_for_inference(model, self.rank, checkpoint_path)
        train_dl, train_sampler,valid_dl, valid_sampler, test_dl, test_sampler= self.build_dataloader()
        # inference
        pbar = tqdm(
            enumerate(test_dl),
            bar_format='{desc:<15}{percentage:3.0f}%|{bar:18}{r_bar}', 
            total=len(test_dl),
            disable=not self.is_master
            ) # set progress bar
        t_acc = np.zeros(1)
        t_recall = np.zeros(1)
        t_precision = np.zeros(1)
        t_loss = np.zeros(1)
        t_imgnum = np.zeros(1)
        epoch = 1

        for step, (image, label) in pbar:
            image = image.to(device=self.rank, non_blocking=True).float()
            label = label.to(device=self.rank, non_blocking=True).float()
            with self.amp_autocast():
                input = image
                y_pred = model(input).squeeze()
                label = label.to(torch.int64)
            self.test_multiple_sample_visualization(y_pred, label, step)
            accuracies, precisions, recalls = self.evaluation_for_semantic_segmentation(y_pred, label)
            temp_acc, temp_recall, temp_precision, temp_imgnum = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
            for i in range(image.shape[0]):
                temp_acc += accuracies[i]
                temp_recall += recalls[i]
                temp_precision += precisions[i]
                t_acc += accuracies[i]
                t_recall += recalls[i]
                t_precision += precisions[i]
            t_imgnum += image.shape[0]
            temp_imgnum += image.shape[0]
        if self.is_master:
            self.writer.add_scalar("ACC/test", t_acc / t_imgnum, epoch)
            self.writer.add_scalar("Recall/test", t_recall / t_imgnum, epoch)
            self.writer.add_scalar("Precision/test", t_precision / t_imgnum, epoch)
            self.writer.flush()
            
        return t_acc / t_imgnum, t_recall / t_imgnum, t_precision / t_imgnum




    def run(self):
        if self.conf.base.mode == 'train':
            pass
        elif self.conf.base.mode == 'train_eval':
            self.train_eval()
        elif self.conf.base.mode == 'finetuning':
            pass
        elif self.conf.base.mode == 'test':
            test_acc, test_recall, test_precision = self.test()
            print('test_acc:',test_acc, 'test_recall',test_recall, 'test_precision',test_precision)


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


