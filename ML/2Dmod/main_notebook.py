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

imgs_crop = random_crop([img_1, label_1_arrary], [256, 256])
fig, (ax1, ax2) = plt.subplots(nrows=2)
ax1.imshow(imgs_crop[0], cmap='gray')
ax2.imshow(imgs_crop[1], vmin=0, vmax=1)
plt.show()

#%% [markdown]
# # Batch output

#%%
# Create output folder
if not 'prepdata' in os.listdir(data_path):
    os.mkdir(os.path.join(data_path, 'prepdata'))
    os.mkdir(os.path.join(data_path, 'prepdata', 'images'))
    os.mkdir(os.path.join(data_path, 'prepdata', 'labels'))

# Batch Random Crop
ipfolder = os.path.join(data_path, 'data')
opfolder = os.path.join(data_path, 'prepdata')
random_crop_batch(ipfolder, opfolder, [256, 256], 10)

