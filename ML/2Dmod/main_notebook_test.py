
#%%
# import os, glob, shutil
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
