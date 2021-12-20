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
    def __init__(self, conf, mode, width=512, height=512) -> None:
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
            self.train_transformation = A.Compose([
        A.Resize(height, width),
#         A.Normalize(
#                 mean=[0.485, 0.456, 0.406], 
#                 std=[0.229, 0.224, 0.225], 
#                 max_pixel_value=255.0, 
#                 p=1.0,
#             ),
        A.CLAHE(p=0.35),
        A.ColorJitter(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=90, p=0.5),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
#             A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
        ], p=0.25),
        A.CoarseDropout(max_holes=8, max_height=height//20, max_width=width//20,
                         min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ToTensorV2()], p=1.0)
            self.label, self.temp_df = self.get_label()
        else:
            self.label_path = self.conf['label_path']
            self.transformation = A.Compose([
                A.Resize(height,width),
                ToTensorV2()
            ])
            self.label, self.temp_df = self.get_label()
        self.length = len(self.temp_df) # + data_augmentation (not applied)
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
        return csv_file, temp_df

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
                #a_mask_int = a_mask.astype('uint8')
                #viz = np.zeros((a_mask.shape[0], a_mask.shape[1], 3), dtype='uint8')
                #contours, hierarchy = cv2.findContours(a_mask_int, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                #for cnt in contours:
                #    cv2.drawContours(viz, [cnt], 0, (255, 0, 0), 3)  # blueo
                #viz = cv2.cvtColor(viz, cv2.COLOR_RGB2GRAY)
                #a_mask = Image.fromarray(viz)
                whole_mask += a_mask
                a_mask = np.array(a_mask) > 0
                masks[:, :, i] = a_mask
                
                boxes.append(self.get_box(a_mask))
            whole_mask[whole_mask>1] = 1
            whole_mask[whole_mask==1] = 0
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
            #transformed_data= self.train_transformation(image = data, bboxes = label['boxes'], mask = whole_mask, bbox_classes=label['labels'])
            #data, label['masks'], label['boxes'], label['labels'] =  transformed_data['image'], transformed_data['mask'], transformed_data['bboxes'], transformed_data['bbox_classes']
            transformed_data= self.train_transformation(image = data, mask = whole_mask)
            data, label['masks'] =  transformed_data['image'], transformed_data['mask']
            mask_label = label['masks']
            return data, mask_label
            
        else: # for test dataset
            img_path = self.image_info[index]["image_path"]
            data = np.array(Image.open(img_path).convert("RGB"))

            # masks
            info = self.image_info[index]

            # object number
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
            #transformed_data= self.train_transformation(image = data, bboxes = label['boxes'], mask = whole_mask, bbox_classes=label['labels'])
            #data, label['masks'], label['boxes'], label['labels'] =  transformed_data['image'], transformed_data['mask'], transformed_data['bboxes'], transformed_data['bbox_classes']
            transformed_data= self.transformation(image = data, mask = whole_mask)
            data, label['masks'] =  transformed_data['image'], transformed_data['mask']
            mask_label = label['masks']
            return data, mask_label
