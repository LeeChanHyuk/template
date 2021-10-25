# General utility files for preprocessing data
from torchvision import transforms
from PIL import Image
import numpy as np
from glob import glob
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import pydicom
import nibabel as nib
import SimpleITK as sitk
import os
from nilearn.image import resample_img


'''
Problem:
patient = each sample consisting a mini-batch
2D image = images averaged over z-axis (each channel = MRI type)
3D image = images stacked over z-axis (each channel = MRI type)
'''

def preprocess(config):
    if config['type'] == 'pad':
        return transforms.Pad(**config['params'])
    elif config['type'] == 'resize':
        return transforms.Resize(**config['params'])
    elif config['type'] == 'randomcrop':
        return transforms.RandomCrop(**config['params'])
    elif config['type'] == 'horizontal':
        return transforms.RandomHorizontalFlip()
    elif config['type'] == 'tensor':
        return transforms.ToTensor()
    elif config['type'] == 'normalize':
        return transforms.Normalize(**config['params'])


def load_png(image_path, channel_type="L"):  # "RGB" or "L"
    # 2D
    return np.array(Image.open(image_path).convert(channel_type))

def resample(image, ref_image):

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    
    resampler.SetTransform(sitk.AffineTransform(image.GetDimension()))

    resampler.SetOutputSpacing(ref_image.GetSpacing())

    resampler.SetSize(ref_image.GetSize())

    resampler.SetOutputDirection(ref_image.GetDirection())

    resampler.SetOutputOrigin(ref_image.GetOrigin())

    resampler.SetDefaultPixelValue(image.GetPixelIDValue())

    resamped_image = resampler.Execute(image)
    
    return resamped_image

def registration(conf, dicom_path):
    # load dicom file and convert dicom file to nibabel file
    reader = sitk.ImageSeriesReader()
    reader.LoadPrivateTagsOn()
    filenamesDICOM = reader.GetGDCMSeriesFileNames(dicom_path)
    reader.SetFileNames(filenamesDICOM)
    converted_dicom = reader.Execute()
    if os.path.exists('dicom.nii'):
        os.unlink('dicom.nii')
    sitk.WriteImage(converted_dicom,'dicom.nii')
    converted_dicom_nibabel = nib.load('dicom.nii')

    # load reference file (From task 1)
    patient_number, mri_type = dicom_path.split('/')[-2], dicom_path.split('/')[-1]
    if mri_type == 'T2w':
        mri_type = 'T2'
    nibabel_path = os.path.join(conf['nibabel_path'], 'BraTS2021_' + patient_number)
    nibabel_file = nib.load(os.path.join(nibabel_path, 'BraTS2021_' + patient_number + '_' + mri_type.lower() + '.nii.gz'))
    dicom_resampled = resample_img(converted_dicom_nibabel, target_affine=nibabel_file.affine, target_shape=nibabel_file.shape)
    final_converted_image = dicom_resampled.get_fdata()
    #final_converted_image = sitk.GetArrayFromImage(dicom_resampled)

    return final_converted_image

def dicom_file_to_img(conf, patient_path, mri_type):
    imgs = registration(conf, os.path.join(patient_path, mri_type))
    return imgs


# RSNA-MICCAI Brain Tumor Radiogenomic Classification specific utils
def mri_png2array(image_path_dict,
                  output_type="2D"):
                  
    # image_path_dict = MRI_type : each paths from one patient

    stacked_img = []
    for mri_i, each_MRI in enumerate(image_path_dict):
        for each_img_path in image_path_dict[each_MRI]:
            img = np.array(Image.open(each_img_path).convert("L"))
            stacked_img.append(img)
    
    if output_type == "2D":
        stacked_img = np.asarray(stacked_img)
        stacked_img = np.average(stacked_img, axis=1)
    elif output_type == "25D":
        
        stacked_img  = stacked_img
        
    return stacked_img


# Sampling scheme for MRI scans (i.e. how to sample along z-axis)
def random_stack(img_path_list,
                 N_samples=10):
    return np.random.choice(img_path_list, N_samples)


def sequential_stack(img_path_list,
                     N_samples=10):
    N = len(img_path_list)
    if N_samples > N:
        # Random sample additional images to match the number of samples
        add_samples = N_samples - N
        sampled = np.random.choice(img_path_list, add_samples).tolist()
        img_path_list = sorted(img_path_list + sampled)
        return img_path_list
    else:
        d = N // N_samples
        indices = range(0, N, d)[:N_samples]
        img_path_list = np.array(img_path_list)[indices]

        return img_path_list.tolist()


def process_label_csv(source_csv="./train_labels.csv",
                      target_csv="./experiments/exp1/train.csv",
                      K_fold=5,
                      seed=1234
                      ):
    df = pd.read_csv(source_csv)
    X = df["BraTS21ID"]
    y = df["MGMT_value"]
    skf = StratifiedKFold(n_splits=K_fold,
                          shuffle=True,
                          random_state=seed)
    flag_indices = np.zeros(len(y))

    for fold_i, (train_index, val_index) in enumerate(skf.split(X, y)):
        flag_indices[val_index] = fold_i

    df["BraTS21ID"] = df["BraTS21ID"].astype(str).str.zfill(5)
    df["flag_index"] = flag_indices.astype(int)
    df.to_csv(target_csv)

    return
