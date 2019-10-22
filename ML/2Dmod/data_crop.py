#%%
import os, glob, shutil
import cv2
import numpy as np
import uuid
import tensorflow as tf
from skimage.io import imread, imsave, imshow
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from core.imageprep import random_crop, crop_generator, random_crop_batch

# %load_ext autoreload
# %autoreload 2

#%%
seed = 100
mainpath = '/Volumes/LaCie_DataStorage/PerlmutterData/training'
labeltype = 'cell_membrane'
foldernames = ['train', 'valid', 'test']
foldernames_class = ['images', 'labels']
label_names = ['cell_membrane']

#%% [markdown]
# # Prepare training image

#%%
# Prepare the training dataset
# Specify the data folder
data_path = os.path.join(mainpath, labeltype)

# create list for filenames
imglist = glob.glob(os.path.join(data_path, 'data', 'images', '*', '*.tif'), recursive=True)
labellist = glob.glob(os.path.join(data_path, 'data', 'labels', '*', '*.tif'), recursive=True)
print('First 5 filenames')
print(imglist[:5])
print('First 5 filenames')
print(labellist[:5])

#%% [markdown]
# # Batch output

#%%
# Create output folder
if not 'prepdata' in os.listdir(data_path):
    os.mkdir(os.path.join(data_path, 'prepdata'))
    for foldername in foldernames:    
        os.mkdir(os.path.join(data_path, 'prepdata', foldername))
        for foldername_class in foldernames_class:
            os.mkdir(os.path.join(data_path, 'prepdata', foldername, foldername_class))
            for label_name in label_names: 
                os.mkdir(os.path.join(data_path, 'prepdata', 
                                        foldername, foldername_class, label_name))                

#%%
# Batch Random Crop
ipfolder = os.path.join(data_path, 'data')
# create train dataset 
opfolder = os.path.join(data_path, 'prepdata', 'train')
random_crop_batch(ipfolder, opfolder, label_names[0], [256, 256], 20, seed)
# create valid dataset
opfolder = os.path.join(data_path, 'prepdata', 'valid')
random_crop_batch(ipfolder, opfolder, label_names[0], [256, 256], 1, seed+1)

