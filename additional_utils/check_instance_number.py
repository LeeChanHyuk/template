import os
import pandas as pd
import numpy as np

csv_path = 'C:/Users/user/Desktop/git/template/train_label.csv'
csv = pd.read_csv(csv_path)

annotations = csv['annotation']
ids = csv['id']
id_list = []
for id in ids:
	id_list.append(id)
id_list = list(set(id_list))
id_list_dict = {}
for id in id_list:
	id_list_dict[id] = 0
for id in ids:
	id_list_dict[id] += 1
total_num=0
for instance_number in id_list_dict.values():
	total_num += instance_number
print(total_num / len(id_list_dict))