from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from imutils import paths
from PIL import Image, ImageTk
import numpy as np
from data import load_test_data

path_pedict = r"D:\PerlmutterData\training\history\model-test-1.h5"
model_p = load_model(path_pedict)

test_path = r"D:\PerlmutterData\predict\Test"
imgpath = list(paths.list_images(test_path))
print(imgpath[0])

img = Image.open(imgpath[0])
img_arr = np.array(img)