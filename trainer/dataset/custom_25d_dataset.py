from typing_extensions import final
from albumentations.core.serialization import save
import torch.utils.data as torch_data
import sklearn.model_selection as sk_model_selection
import pandas as pd
from torchvision import transforms
from glob import glob
import numpy as np
from .data_utils import mri_png2array, random_stack, sequential_stack
from .data_utils import dicom_file_to_img
from tqdm import tqdm 
import os 
import random
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import pickle
import cv2

class custom_25d_dataset(torch_data.Dataset):
    def __init__(self,
                 input_paths,
                 targets=None,
                 mode="train",
                 transform=transforms.ToTensor(),
                 conf=None,
                 local_rank=None):
        '''
        targets: list of values for classification
        or list of paths to segmentation mask for segmentation task.
        augment: list of keywords for augmentations.
        '''
        
        self.input_paths = input_paths  # paths to patients
        self.targets = targets
        self.mode = mode
        self.transform = transform
        self.conf = conf
        self.segmentation_path = '/home/ddl/다운로드/data_original/task1/png_data'
        self.transform = A.Compose([
                        A.Resize(192,192)
                        ])
        self.totensor = ToTensorV2() 
        self.local_rank = local_rank
        self.data=self.load_voxel()
        self.channel_length = 0
        self.sampled_image_paths = self.preload_img_paths()

    def __len__(self):
        return len(self.sampled_image_paths)

    def channel_length_call(self):
        return self.channel_length

    def preload_img_paths(self):
        mri_types = self.conf["mri_types"]
        patientes = []
        for imgpath in tqdm(self.input_paths,desc='loading path'): # patient folder
            patient_type = {}
            for j in mri_types:
                patient_type[j] = glob(f'{imgpath}/{j}/*.png') # patient['FLAIR']에는 patient_folder/image_dict['FLAIR']가 들어가게 됨.
            patientes.append(patient_type)
        return patientes # [patient_folder_index / dict1(about flair), dict2(about t2`w)]
        

    def load_voxel(self):
        patients=[]
        for path in self.input_paths:
            voxels = []
            for mri_type in self.conf['mri_types']:
                voxel = np.load(os.path.join(path, mri_type + '.npy'))
                voxels.append(voxel[51:84])
            new_voxels = np.concatenate((voxels), axis=0)
            patients.append(new_voxels)
        return patients
        

    def load_input(self, index):
        sampled_imgs = self.sampled_image_paths[index]
        for mri_i, each_MRI in enumerate(sampled_imgs):
            if len(sampled_imgs[each_MRI]) < 11:
                print(sampled_imgs[each_MRI],each_MRI,len(sampled_imgs[each_MRI]),len(sampled_imgs[each_MRI])-self.conf['N_sample'])
            start = random.randint(0,len(sampled_imgs[each_MRI])-self.conf['N_sample'])
            sampled_imgs[each_MRI] = sampled_imgs[each_MRI][start : start + self.conf['N_sample']]
        
        inputs = mri_png2array(sampled_imgs,
                               output_type=self.conf["output_type"])
        
        return inputs

    def min_max_normalization_per_img(self, img):
        max_val = np.max(img)
        min_val = np.min(img)
        if max_val == 0:
            return np.zeros((img.shape[0], img.shape[1]))
        img = (img - min_val) / max_val
        return img
    
    def min_max_normalization_per_imgs(self,imgs):
        max_val = np.max(imgs)
        min_val = np.min(imgs)
        if max_val == 0:
            return np.zeros((imgs.shape))
        imgs = (imgs - min_val) / max_val
        np.where(imgs<0, 0, imgs)
        return imgs

    def load_input_from_dicom_files(self, index):
        patient_index = self.input_paths[index].split('/')[-1]
        mri_types = self.conf['mri_types']
        mri_array = []
        patient_path = os.path.join(self.conf['dicom_path'], str(patient_index).zfill(5))
        for mri_type in mri_types:
            mri_array.append(dicom_file_to_img(self.conf, patient_path, mri_type))
        end = 114
        front = 50
        self.channel_length = end - front + 1
        temp_array = np.concatenate((mri_array[0][:,:,50:114], mri_array[1][:,:,50:114]), axis=2)
        temp_array = np.transpose(temp_array, (2,0,1))
        final_array = self.min_max_normalization_per_imgs(temp_array)
        final_array = np.array(final_array)
        return final_array

    def load_target(self, index):  # For classification task
        return self.targets[index]

    def load_segmentation_image(self, patient_index):
        folder_path = os.path.join(self.segmentation_path, 'BraTS2021_'+ patient_index, 'seg')
        imgs = []
        for img in os.listdir(folder_path):
            image = cv2.imread(os.path.join(folder_path, img), 0)
            image = cv2.resize(image, (192, 192))
            imgs.append(image / 255)
        imgs = np.array(imgs)[66:86]
        imgs = np.concatenate((imgs, imgs), axis=0)
        return np.array(imgs)

    def __getitem__(self, index):
        #if self.data.get(index):
        #inputs = self.data[index]
        #patient_index = self.input_paths[index].split('/')[-1] # 숫자만 반환함.
        #seg_array = self.load_segmentation_image(patient_index)
        #inputs = self.load_input(index)
        inputs = self.data[index]
        if self.conf['output_type'] == '25D' or self.conf['output_type'] == '3D':
            inputs = np.array([self.transform(image=np.array(i))['image'] for i in inputs])
            inputs = self.totensor(image=inputs/255.)['image'].permute(1,0,2)
            #seg_array = self.totensor(image = seg_array)['image'].permute(1,0,2)

        if self.mode != "test":
            targets = self.load_target(index)
            return inputs, targets
        else:
            return inputs