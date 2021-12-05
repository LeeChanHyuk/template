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

class ch_dataset(torch.utils.data.Dataset):
    def __init__(self, conf, mode) -> None:
        super().__init__() # 아무것도 안들어있으면 자기 자신을 호출 (왜?)
        self.conf = conf[mode]
        self.mode = mode
        self.data_path = self.conf['dataset_path']
        self.basic_transforms = ToTensorV2()
        if self.mode == 'train':
            self.label_path = self.conf['label_path']
            self.additional_transforms_2d = A.Compose([
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.Normalize()
            ])
            self.label = self.get_label()
        self.data = self.get_data()
        self.length = self.data.shape[0] # + data_augmentation (not applied)

    def get_data(self):
        # get patient_folders name
        files = sorted(glob.glob(os.path.join(self.data_path, '*')))

        # initialize whole dataset list
        datas = []
        
        # for each patient folder
        for file in tqdm.tqdm(files):
            datas.append(cv2.imread(os.path.join(self.data_path, file)))
        return datas

    def get_label(self):
        csv_file = pd.read_csv(self.conf['label_path'])
        return csv_file

    def rle_decode(mask_rle, shape, color=1):
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

    # data augmentation is conducted in here because of probability of augmentation method
    def __getitem__(self, index):
        if self.mode == 'train':
            # load data  
            data = self.data[index]
            # data augmentation
            transformed_patient_data = np.squeeze(self.additional_transforms_2d(np.expand_dims(data, 0)))
            # concat in channel dimension
            data = np.concatenate([data, transformed_patient_data], axis=len(data.shape)-3)

            # basic transformation
            data = self.basic_transforms(image = data)['image']

            # load label_csv file
            label = self.label[index]

            # object number
            n_objects = len(label['annotations'])

            # masks
            height = 224
            width = 224
            masks = np.zeros((len(label['annotations']), self.height, self.width), dtype=np.uint8)
            boxes = []
        
            for i, annotation in enumerate(label['annotations']):
                a_mask = self.rle_decode(annotation, (self.height, self.width))
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

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((n_objects,), dtype=torch.int64)

        # This is the required target for the Mask R-CNN
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }
            
            return data, label
        
        else: # for test dataset
            # load data
            patient_data = self.data[index][0]
            # data augmentation
            transformed_patient_data = np.squeeze(self.transforms(np.expand_dims(patient_data), 0))
            # concat
            data = np.concatenate([patient_data, transformed_patient_data], axis=len(patient_data.shape)-3)
            data = self.basic_transforms(image = data)['image']
            return data


        


    