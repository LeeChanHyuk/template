import pickle
import os
import numpy as np
import cv2

local_path = '/home/ddl/다운로드/data_original/task1/png_data'
folder_path = '/home/ddl/다운로드/data_original/task1/png_data/BraTS2021_00000/seg'
voxel_path = '/home/ddl/git/kaggle-template/data/voxel_256x256/train'

for path in os.listdir(voxel_path):
    if len(os.listdir(os.path.join(voxel_path, path))) < 4:
        print(path)