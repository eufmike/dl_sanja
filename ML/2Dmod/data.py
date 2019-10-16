#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 14:46:32 2018

@author: bertinetti
"""
from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave, imread
import numpy as np
from sklearn.feature_extraction import image

def load_train_data(folder,train_n,labels_n):
    imgs_train = np.load(os.path.join(folder,train_n))
    imgs_mask_train = np.load(os.path.join(folder,labels_n))
    return imgs_train, imgs_mask_train

def create_train_data(data_path,train_p,label_p,fext):
    train_data_path = os.path.join(data_path, train_p)
    train_images = [f for f in sorted(os.listdir(train_data_path)) if fext in f]
    label_data_path=os.path.join(data_path, label_p)
    label_images=[f for f in sorted(os.listdir(label_data_path)) if fext in f]
    total = len(train_images)
    if total != len(label_images):
        print('!'*50)
        print('Number of training images and labels do not match')
        print('!'*50)
    tmp=imread(os.path.join(train_data_path, train_images[0]), as_grey=True)
    image_rows,image_cols=tmp.shape[0],tmp.shape[1]
    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Loading training images...')
    print('-'*30)
    for image_name in train_images:
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        img = np.array([img])
        imgs[i] = img
        if i % 10 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('-'*30)
    print('Loading training labels...')
    print('-'*30)
    i = 0
    for image_name in label_images:
        img = imread(os.path.join(label_data_path, image_name), as_grey=True)
        img = np.array([img])
        imgs_mask[i] = img
        if i % 10 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    
    print('Loading done.')

    np.save(os.path.join(data_path,'imgs_train.npy'), imgs)
    print('Train data saved to:',os.path.join(data_path,'imgs_train.npy'))
    np.save(os.path.join(data_path,'imgs_labels.npy'), imgs_mask)
    print('Label data saved to:',os.path.join(data_path,'imgs_labels.npy'))
    print('Saving to .npy files done.')

def load_test_data(folder,name_im,name_id):
    imgs_test = np.load(os.path.join(folder,name_im))
    imgs_id = np.load(os.path.join(folder,name_id))
    return imgs_test, imgs_id

def create_test_data(data_path,test_p,fext):
    test_data_path = os.path.join(data_path, test_p)
    images = [f for f in sorted(os.listdir(test_data_path)) if fext in f]
    total = len(images)
    tmp=imread(os.path.join(test_data_path, images[0]), as_grey=True)
    image_rows,image_cols=tmp.shape[0],tmp.shape[1]

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Loading test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0][-4:].lstrip('0'))
        img = imread(os.path.join(test_data_path, image_name), as_grey=True)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 10 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(os.path.join(data_path,'imgs_test.npy'), imgs)
    np.save(os.path.join(data_path,'imgs_id_test.npy'), imgs_id)
    print('Test data saved to:',os.path.join(data_path,'imgs_train.npy'))
    print('Index file saved to:',os.path.join(data_path,'imgs_id_test.npy'))
    print('Saving to .npy files done.')
    
def crop_no_black(trim,labim,size):
    seed=np.random.randint(10000)
    for i,patch in enumerate(image.extract_patches_2d(trim,size,max_patches=0.2,random_state=seed)):
        if not 0 in patch:
            return patch,image.extract_patches_2d(labim,size,max_patches=0.2,random_state=seed)[i]
        break

def preprocess(imgs,img_rows, img_cols):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], ( img_rows, img_cols), preserve_range=True)    
    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

def getSegmentationArr(array , nClasses ):
    n,height,width,tmp=array.shape
    seg_labels = np.zeros(( n,  height , width  , nClasses ))
    for i in range(n):
        for c in range(nClasses):
            try:
                img = array[i,:,:,0]
                seg_labels[i, : , : , c ] = (img == c ).astype(int)
#                print(seg_labels.shape)
            except Exception as e:
                print(e)
    seg_labels = np.reshape(seg_labels, ( n, width*height , nClasses ))
    return seg_labels