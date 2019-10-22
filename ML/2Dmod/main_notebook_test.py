#%%
# import os, glob, shutil
import os, sys
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
from imutils import paths

%load_ext autoreload
%autoreload 2

#%%
path = "/Volumes/LaCie_DataStorage/PerlmutterData/training/cell_membrane/prepdata"
imgpath = list(paths.list_images(path))
print(imgpath[0])

#%%
data_gen_args = dict(
                featurewise_center=True,
                featurewise_std_normalization=True,
                horizontal_flip=True,
                vertical_flip=True,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.07,
                zoom_range=0.2,
                fill_mode='constant',
                cval=0.,)
seed = 100

#%%
image_datagen = ImageDataGenerator(**data_gen_args)
label_datagen = ImageDataGenerator(**data_gen_args)

#%%
image_generator = image_datagen.flow_from_directory(
    os.path.join(path, 'train/images/'),
    class_mode=None,
    seed=seed)

label_generator = label_datagen.flow_from_directory(
    os.path.join(path, 'train/labels'),
    class_mode=None,
    seed=seed)

#%% 
train_generator = zip(image_generator, label_generator)

#%%

#%%
image_datagen.fit(images, augment=True, seed=seed) 
mask_datagen.fit(masks, augment=True, seed=seed)


#%%
