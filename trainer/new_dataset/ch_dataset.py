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
        self.test_transforms = A.Compose([A.Resize(height,width),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['bbox_classes']))
        self.image_info = collections.defaultdict(dict)
        if self.mode == 'train':
            self.label_path = self.conf['label_path']
            self.train_transformation = A.Compose([
                A.VerticalFlip(),
                A.HorizontalFlip(), # normalize
                A.Resize(height,width),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco', label_fields=['bbox_classes']))
            self.label = self.get_label()
        elif self.mode == 'valid':
            self.label_path = self.conf['label_path']
            self.train_transformation = A.Compose([
                A.Resize(height,width),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco', label_fields=['bbox_classes']))
            self.label = self.get_label()
        self.length = len(os.listdir(self.data_path)) # + data_augmentation (not applied)
        self.width = width
        self.height = height

    def label_transformation(self, image, label, horizontal_flip = False, vertical_flip = False, resize = False):
        if horizontal_flip:
            height, width = image.shape[-2:]
            label["masks"] = label["masks"].flip(-1)
        if vertical_flip:
            height, width = image.shape[-2:]
            image = image.flip(-2)
            label["masks"] = label["masks"].flip(-2)
        
            
        return label


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
        width = a_mask.shape[1]
        height = a_mask.shape[0]
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        if xmin> width:
            xmin = width
            print('xmin error')
        if xmax>width:
            xmax = width
            print('xmax error')
        if ymin>height:
            ymin = height
            print('ymin error')
        if ymax>height:
            ymax = height
            print('ymax error')
        mask_width = xmax - xmin
        mask_height = ymax- ymin
        return [xmin, ymin, mask_width, mask_height]

    # data augmentation is conducted in here because of probability of augmentation method
    def __getitem__(self, index):
        if self.mode == 'train':
            # load data  
            img_path = self.image_info[index]["image_path"]
            data = np.array(Image.open(img_path).convert("RGB"))

            # concat in channel dimension
            #if len(data.shape) == 2:
            #    data_concat = np.concatenate([data[:, None], transformed_data['image'][:, None]], axis=0)
            #else:
            #    data_concat = np.concatenate([data, transformed_data['image']], axis=2)

            # masks
            info = self.image_info[index]

            # object number
            n_objects = len(info['annotations'])


            n_objects = len(info['annotations'])
            masks = np.zeros((data.shape[0], data.shape[1], len(info['annotations'])), dtype=np.uint8)
            boxes = []
        
            whole_mask = np.zeros((data.shape[0], data.shape[1])) # for semantic segmentation
            for i, annotation in enumerate(info['annotations']):
                a_mask = self.rle_decode(annotation, (data.shape[0], data.shape[1])) # 이거 두 개가 바뀌어야 하는건가?
                a_mask = Image.fromarray(a_mask)
                whole_mask += a_mask
                a_mask = np.array(a_mask) > 0
                masks[:, :, i] = a_mask
                
                boxes.append(self.get_box(a_mask))
            whole_mask[whole_mask>0] = 1
            # dummy labels
            labels = [1 for _ in range(n_objects)]
            
            #boxes = torch.as_tensor(boxes, dtype=torch.float32)
            #labels = torch.as_tensor(labels, dtype=torch.int64)
            #masks = torch.as_tensor(masks, dtype=torch.uint8)

            image_id = torch.tensor([index])
            boxes = np.array(boxes)
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
            transformed_data= self.train_transformation(image = data, bboxes = label['boxes'], mask = whole_mask, bbox_classes=label['labels'])
            data, label['masks'], label['boxes'], label['labels'] =  transformed_data['image'], transformed_data['mask'], transformed_data['bboxes'], transformed_data['bbox_classes']
            mask_label = label['masks']
            return data, mask_label
            
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
            data = self.test_transforms(image = data)['image']
            return data
