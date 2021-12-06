# init에서는 dataset을 create하는 함수를 만들어야지.
import logging
import torch
import torchvision
import torch.utils.data
import hydra
import glob
import numpy as np
import os
import pandas as pd
import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from PIL import Image
import collections

class ch_dataset(torch.utils.data.Dataset):
    def __init__(self, conf, mode, width=224, height=224) -> None:
        super().__init__() # 아무것도 안들어있으면 자기 자신을 호출 (왜?)
        self.conf = conf[mode]
        self.mode = mode
        self.data_path = self.conf['dataset_path']
        self.basic_transforms = ToTensorV2()
        self.image_info = collections.defaultdict(dict)
        if self.mode == 'train':
            self.label_path = self.conf['label_path']
            self.additional_transforms_2d = A.Compose([
                A.VerticalFlip(),
                A.HorizontalFlip(), # normalize
            ])
            self.label = self.get_label()
        self.length = len(os.listdir(self.data_path)) # + data_augmentation (not applied)
        self.width = width
        self.height = height
        self.should_resize = False

    def get_label(self):
        csv_file = pd.read_csv(self.conf['label_path'])
        temp_df = csv_file.groupby('id')['annotation'].agg(lambda x: list(x)).reset_index()
        for index, row in temp_df.iterrows():
            self.image_info[index] = {
                    'image_id': row['id'],
                    'image_path': os.path.join(self.data_path, row['id'] + '.png'),
                    'annotations': row["annotation"]
                    }
        return csv_file

    def rle_decode(self, mask_rle, shape, color=1):
        '''
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return 
        Returns numpy array, 1 - mask, 0 - background
        '''
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.float32)
        for lo, hi in zip(starts, ends):
            img[lo : hi] = color
        return img.reshape(shape)


    def __len__(self):
        return self.length

    def get_box(self, a_mask):
        ''' Get the bounding box of a given mask '''
        pos = np.where(a_mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax, ymax]

    # data augmentation is conducted in here because of probability of augmentation method
    def __getitem__(self, index):
        if self.mode == 'train':
            # load data  
            img_path = self.image_info[index]["image_path"]
            data = np.array(Image.open(img_path).convert("RGB"))

            # data augmentation # must solve normalize issue!
            transformed_data = self.additional_transforms_2d(image=data)

            # concat in channel dimension
            if len(data.shape) == 2:
                data_concat = np.concatenate([data[None,:], transformed_data['image'][None, :]], axis=0)
            else:
                data_concat = np.concatenate([data, transformed_data['image']], axis=0)

            # masks
            info = self.image_info[index]

            # object number
            n_objects = len(info['annotations'])


            n_objects = len(info['annotations'])
            masks = np.zeros((len(info['annotations']), data.shape[1], data.shape[0]), dtype=np.uint8)
            boxes = []
        
            for i, annotation in enumerate(info['annotations']):
                a_mask = self.rle_decode(annotation, (data.shape[1], data.shape[0]))
                a_mask = Image.fromarray(a_mask)
            
            if self.should_resize:
                a_mask = a_mask.resize((self.width, self.height), resample=Image.BILINEAR)
            
            a_mask = np.array(a_mask) > 0
            masks[i, :, :] = a_mask
            
            boxes.append(self.get_box(a_mask))

            # dummy labels
            labels = [1 for _ in range(n_objects)]
            
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

            image_id = torch.tensor([index])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((n_objects,), dtype=torch.int64)

            # This is the required target for the Mask R-CNN
            label = {
                'boxes': boxes,
                'labels': labels,
                'masks': masks,
                'image_id': image_id,
                'area': area,
                'iscrowd': iscrowd
            }
                
            # basic transformation
            data = self.basic_transforms(image = data_concat)['image']

            return data, label
            
        else: # for test dataset
            files_names = os.listdir(self.data_path)
            # load data
            data = Image.open(os.path.join(self.data_path, files_names[index])).convert("RGB")
            # data augmentation
            transformed_data = self.additional_transforms_2d(image = data)['image']

            # concat in channel dimension
            if len(data.shape) == 2:
                data = np.concatenate([data[None,:], transformed_data['image'][None, :]], axis=0)
            else:
                data = np.concatenate([data, transformed_data['image']], axis=0)

            # concat
            data = self.basic_transforms(image = data)['image']
            return data
