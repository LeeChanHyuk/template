import os
import pandas as pd
import numpy as np
from tqdm import tqdm

csv_path = '/home/ddl/git/template/dataset/cell_dataset/train.csv'
csv = pd.read_csv(csv_path)
train_image_path = '/home/ddl/Downloads/label_info/binary_mask/fold0/train'
train_image_name_list = list(os.listdir(train_image_path))
test_image_path = '/home/ddl/Downloads/label_info/binary_mask/fold0/valid'
test_image_name_list = list(os.listdir(test_image_path))

annotations = csv['annotation']
ids = csv['id']
id_list = []
train_df = csv.loc[(csv['id']+'.png').isin(train_image_name_list)]
test_df = csv.loc[(csv['id']+'.png').isin(test_image_name_list)]
train_df.to_csv('train.csv')
test_df.to_csv('test.csv')