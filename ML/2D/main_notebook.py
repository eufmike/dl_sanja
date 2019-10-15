#%%
import os, glob, shutil
import cv2
import uuid
import tensorflow as tf
from PIL import Image, ImageTk
from skimage.io import imsave, imread
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential


#%%
path="/Volumes/LaCie_DataStorage/PerlmutterData/training"
labelname = 'cell_membrane'

#%%
id_01 = uuid.uuid4()
print(id_01.hex)

#%%
