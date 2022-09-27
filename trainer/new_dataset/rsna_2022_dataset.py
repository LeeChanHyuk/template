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
import torchio as tio

class RSNA_2022_dataset(torch.utils.data.Dataset):
    def __init__(self, conf, mode) -> None:
        super().__init__() # 아무것도 안들어있으면 자기 자신을 호출 (왜?)
        self.conf = conf[mode]
        self.mode = mode
        self.data_path = self.conf['dataset_path']
        self.basic_transforms = ToTensorV2()
        if self.mode == 'train':
            self.label_path = self.conf['label_path']
            self.additional_transforms_2d = A.Compose([])
            self.additional_transforms_3d = tio.transforms.Compose([
                tio.RandomAffine()
            ])
            self.label = self.get_label()
        self.data = self.get_data()
        self.length = self.data.shape[0] # + data_augmentation (not applied)

    def get_data(self):
        # get patient_folders name
        patient_id_list = os.listdir(self.data_path)
        patient_datas = []
        for i in range(len(patient_id_list)):
            np_file = np.load(os.path.join(self.data_path, patient_id_list[i]))
            patient_id_list.append(np_file)
        for name in self.patient_names:
            self.patient_numpy_datas[name] = [[], [], [], [], [], [], []]
            patient_csv = pixel_count_csv[pixel_count_csv['StudyInstanceUID'] == name]
            patient_data = np.load(os.path.join(self.train_numpy_path, name))
            for line in range(len(patient_csv)):
                line_data = patient_csv.iloc[line]
                for index, vertebrae_name in enumerate(self.targets):
                    if line_data[vertebrae_name] * line_data['pixels'] > 500:
                        slice_num = line_data['Slice']
                        self.patient_numpy_datas[name][index].append(patient_data[slice_num])
            for i in range(7):
                self.patient_numpy_datas[name][i] = np.array(self.patient_numpy_datas[name][i])
            label_line = train_csv[train_csv['StudyInstanceUID'] == name]
            self.patient_label_datas[name] = [label_line[self.targets]]
        return patient_datas, patient_id_list

    def get_label(self):
        csv_file = pd.read_csv(self.conf['label_path'])
        return csv_file


    def __len__(self):
        return self.length

    # data augmentation is conducted in here because of probability of augmentation method
    def __getitem__(self, index):
        if self.mode == 'train':
            # load data  
            patient_data = self.data[index][0] # flair
            # data augmentation
            transformed_patient_data = np.squeeze(self.additional_transforms_3d(np.expand_dims(patient_data, 0)))
            # concat in channel dimension
            data = np.concatenate([patient_data, transformed_patient_data], axis=len(patient_data.shape)-3)

            # basic transformation
            data = self.basic_transforms(image = data)['image']

            # load label_csv file
            label = self.label['MGMT_value'][index]
            
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


        


    