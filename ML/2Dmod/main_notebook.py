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
from core.imageprep import random_crop, crop_generator

%load_ext autoreload
%autoreload 2

#%%
mainpath = '/Volumes/LaCie_DataStorage/PerlmutterData/training'
labeltype = 'cell_membrane'
data_path = os.path.join(mainpath, labeltype, 'data')

#%%
'''
id_01 = uuid.uuid4()
print(id_01.hex)
'''

#%%
imglist = glob.glob(os.path.join(data_path, 'images', '*', '*.tif'), recursive=True)
labellist = glob.glob(os.path.join(data_path, 'labels', '*', '*.tif'), recursive=True)
print(imglist)
print(labellist)

#%%
img_1 = imread(imglist[0])
imshow(img_1)

#%%
label_1 = Image.open(labellist[0])
label_1_arrary = np.array(label_1)
imshow(label_1_arrary)

#%%
img_dx, img_dy = img_1.shape
    
#%%
imgs_crop = random_crop([img_1, label_1_arrary], [256, 256])
'''
fig, (ax1, ax2) = plt.subplots(nrows=2)
ax1.imshow(imgs_crop[0], cmap='gray')
ax2.imshow(imgs_crop[1], vmin=0, vmax=1)
plt.show()
'''

#%%
img_crop = Image.fromarray(imgs_crop[0])
label_crop = Image.fromarray(imgs_crop[1])

#%%
if not 'prepdata' in os.listdir(data_path):
    os.mkdir(os.path.join(mainpath, labeltype, 'prepdata'))
    os.mkdir(os.path.join(mainpath, labeltype, 'prepdata', 'images'))
    os.mkdir(os.path.join(mainpath, labeltype, 'prepdata', 'labels'))
#%%
temp_id = uuid.uuid4()
print(temp_id.hex)
#%%
img_crop.save(os.path.join(mainpath, labeltype, 'prepdata', 'images', temp_id.hex + '.tif'))
label_crop.save(os.path.join(mainpath, labeltype, 'prepdata', 'labels', temp_id.hex + '.tif'))

#%%
for idx, imgpath in enumerate(imglist):
    
    img_tmp = imread(imglist[idx])
    label_tmp = Image.open(labellist[idx])
    label_tmp_array = np.array(label_tmp)
    
    for i in range(10):
        imgs_crop = random_crop([img_1, label_1_arrary], [256, 256])
    
        img_crop = Image.fromarray(imgs_crop[0])
        label_crop = Image.fromarray(imgs_crop[1])
    
        temp_id = uuid.uuid4()
    
        img_crop.save(os.path.join(mainpath, labeltype, 'prepdata', 'images', temp_id.hex + '.tif'))
        label_crop.save(os.path.join(mainpath, labeltype, 'prepdata', 'labels', temp_id.hex + '.tif'))

#%%
