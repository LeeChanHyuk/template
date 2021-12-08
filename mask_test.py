import os
import numpy as np
import cv2

file_path = '/home/ddl/다운로드/binary'
for file in os.listdir(file_path):
    loaded_file = np.load(os.path.join(file_path, file))
    cv2.imshow("file", loaded_file)
    cv2.waitKey(0)
    print(loaded_file)
