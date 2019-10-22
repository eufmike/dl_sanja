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

%load_ext autoreload
%autoreload 2


#%% [markdown]
# # Prepare training image

#%%
# Prepare the training dataset
# Specify the data folder
mainpath = '/Volumes/LaCie_DataStorage/PerlmutterData/training'
labeltype = 'cell_membrane'
data_path = os.path.join(mainpath, labeltype)

# create list for filenames
imglist = glob.glob(os.path.join(data_path, 'data', 'images', '*', '*.tif'), recursive=True)
labellist = glob.glob(os.path.join(data_path, 'data', 'labels', '*', '*.tif'), recursive=True)
print('First 5 filenames')
print(imglist[:5])
print('First 5 filenames')
print(labellist[:5])

#%%
# load the first image and label
# image
img_1 = imread(imglist[0])
# label image
label_1 = Image.open(labellist[0])
label_1_arrary = np.array(label_1)

# print the dimension
img_x, img_y = img_1.shape
print('height(x) frame size: {}'.format(img_x))
print('width(y) frame size: {}'.format(img_y))

# crop the data randomly
imgs_crop = random_crop([img_1, label_1_arrary], [256, 256])

img_1_crop = imgs_crop[0]
label_1_crop = imgs_crop[1]

fig, (ax1, ax2) = plt.subplots(nrows=2)
ax1.imshow(imgs_crop[0], cmap='gray')
ax2.imshow(imgs_crop[1], vmin=0, vmax=1)
plt.show()

#%% [markdown]
# # Batch output

#%%
# Create output folder

if not 'prepdata' in os.listdir(data_path):
    foldernames = ['train', 'valid', 'test']
    os.mkdir(os.path.join(data_path, 'prepdata'))
    for foldername in foldernames:    
        os.mkdir(os.path.join(data_path, 'prepdata', foldername))
        foldernames_class = ['images', 'labels']
        for foldername_class in foldernames_class:
            os.mkdir(os.path.join(data_path, 'prepdata', foldername, foldername_class))
        

#%%
# Batch Random Crop
ipfolder = os.path.join(data_path, 'data')
# create train dataset 
opfolder = os.path.join(data_path, 'prepdata', 'train')
random_crop_batch(ipfolder, opfolder, [256, 256], 20, 100)
# create test dataset
opfolder = os.path.join(data_path, 'prepdata', 'valid')
random_crop_batch(ipfolder, opfolder, [256, 256], 1, 101)

#%% [markdown]
# # Data Augmentation

#%%
# test the data augmentation
img = img_1_crop
print(img.shape)
samples = np.expand_dims(img, 2)
print(samples.shape)
samples_new = np.reshape(samples, [1, samples.shape[0], samples.shape[1], samples.shape[2]])
print(samples_new.shape)
#%%
datagen = ImageDataGenerator(width_shift_range=0.08)
#%%
print(samples_new.shape)
it = datagen.flow(samples_new, batch_size=1)
print(it)

#%%
# generate samples and plot
rows = 3
cols = 3
for i in range(9):
    batch = it.next()
    image = batch[0].astype('uint8').reshape(256, 256)
    plt.subplot(rows, cols, i+1)
    plt.axis('off')
    plt.imshow(image, cmap='gray')
plt.show()

#%%
