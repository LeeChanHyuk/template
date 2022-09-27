import os
import glob
import gc

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from tqdm import tqdm

import pydicom
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import cv2


import nibabel as nib


#set desired image size and depth (number of patient's images to load)
class Config:
    img_size = 512
    depth = 128

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMG_PATH_TRAIN = os.path.join(BASE_PATH, 'dataset/rsna-2022-cervical-spine-fracture-detection/train_images')
IMG_PATH_TEST = os.path.join(BASE_PATH, 'dataset/rsna-2022-cervical-spine-fracture-detection/test_images')
TRAIN_CSV_PATH = os.path.join(BASE_PATH, 'dataset/rsna-2022-cervical-spine-fracture-detection/train.csv')
TEST_CSV_PATH = os.path.join(BASE_PATH, 'dataset/rsna-2022-cervical-spine-fracture-detection/test.csv')

train_images = os.listdir(IMG_PATH_TRAIN)
test_images = os.listdir(IMG_PATH_TEST)

train=pd.read_csv(TRAIN_CSV_PATH)
test=pd.read_csv(TEST_CSV_PATH)

def load_dicom(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    data = cv2.resize(data, (Config.img_size,Config.img_size), interpolation = cv2.INTER_AREA)
    return data
     

def load_dicom_line_par(path, indices:list = None):
    t_paths = sorted(glob.glob(os.path.join(path, "*")),
       key=lambda x: int(x.split('/')[-1].split(".")[0]))
    
    if indices is not None:
        t_paths = [t_paths[i] for i in indices]
        
    images = Parallel(n_jobs=-1)(delayed(load_dicom)(filename) for filename in t_paths)
    
    return np.array(images)

train_output_path = os.path.join('/data', 'RSNA', 'dataset', 'numpy', 'train_arrays/')
test_output_path = os.path.join('/data', 'RSNA', 'dataset', 'numpy', 'test_arrays/')

if not os.path.exists(train_output_path): os.mkdir(train_output_path)
if not os.path.exists(test_output_path): os.mkdir(test_output_path)
    
train_patients = train.StudyInstanceUID.to_list()

def save_3d_voxels(dicom_path, output_path):
    
    n_scans=len(os.listdir(dicom_path))
    
    #instead of zooming whole dicom series, load only part of the images
    ind = np.quantile(list(range(n_scans)), np.linspace(0., 1., Config.depth)).round().astype(int)
    ind = np.arange(n_scans)
    image = load_dicom_line_par(dicom_path, indices = ind)
    
    if image.ndim <4:
        image = np.expand_dims(image, -1)
    
    np.save(f"{output_path}{dicom_path.split('/')[-1]}.npy", image)
    
    del image
    return None
 

for i in tqdm(range(len(train_patients))):
    case = IMG_PATH_TRAIN + '/' + train_patients[i]
    save_3d_voxels(case, train_output_path)
    
gc.collect()
train_arrays = os.listdir(train_output_path)
array_path = train_output_path + train_arrays[np.random.randint(len(train_arrays))]
array = np.load(array_path)
case_id = array_path.split('/')[-1][:-4]