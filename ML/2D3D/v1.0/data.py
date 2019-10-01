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
from sklearn.feature_extraction import image
from scipy.ndimage import zoom
from numpy.random import randint
import numpy as np
import tables
import copy

def load_train_data(folder,train_n,labels_n):
    imgs_train = np.load(os.path.join(folder,train_n))
    imgs_mask_train = np.load(os.path.join(folder,labels_n))
    return imgs_train, imgs_mask_train

def fetch_data_1dir(path,fext):
    training_data_files = list()
    for f in sorted(os.listdir(path)): 
        if fext in f:
            training_data_files.append(os.path.join(path,f))
    return training_data_files

def write_3Ddata_to_file(patches3D, out_file, imtype, truth_dtype=np.uint8, subject_ids=None,
                       normalize=False, crop=False):
    """
    Takes in a set of training images and writes those images to an hdf5 file.
    :param file_list: List containing the data files. 
    :param out_file: Where the hdf5 file will be written to.
    :return: Location of the hdf5 file with the image data written to it. 
    """
    n_samples = patches3D.shape[0]
    n_channels=1
    image_shape = patches3D.shape[1:]
#    print('image shape:',image_shape)
    try:
        # create a hdf5 container
        hdf5_file, storage, filters = create_data_file(out_file,n_samples,image_shape,imtype)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(out_file)
        raise e

    
    write_3Dimage_data_to_file(patches3D, storage, image_shape, n_channels)
    if subject_ids:
        hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids)
    if normalize:
        normalize_data_storage(data_storage)
    hdf5_file.close()
    return filters

def write_data_to_file(file_list, out_file, imtype, truth_dtype=np.uint8, subject_ids=None,
                       normalize=False, crop=False):
    """
    Takes in a set of training images and writes those images to an hdf5 file.
    :param file_list: List containing the data files. 
    :param out_file: Where the hdf5 file will be written to.
    :return: Location of the hdf5 file with the image data written to it. 
    """
    n_samples = len(file_list)
    n_channels=1
    image_shape = read_image(file_list[0]).shape
#    print('image shape:',image_shape)
    try:
        # create a hdf5 container
        hdf5_file, storage, filters = create_data_file(out_file,n_samples,image_shape,imtype)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(out_file)
        raise e

    
    write_image_data_to_file(file_list, storage, image_shape, n_channels)
    if subject_ids:
        hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids)
    if normalize:
        normalize_data_storage(data_storage)
    hdf5_file.close()
    return filters

def create_data_file(out_file, n_samples, image_shape, imtype):
    n_channels=1
    if os.path.isfile(out_file):
        os.remove(out_file)
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='zlib')
    print('Compression details for '+imtype+' images :',filters)
    data_shape = tuple([0] + list(image_shape) + [n_channels])
    storage = hdf5_file.create_earray(hdf5_file.root, imtype, tables.UInt8Atom(), shape=data_shape,
                                           filters=filters, expectedrows=n_samples)
    return hdf5_file, storage, filters

def write_image_data_to_file(in_files, data_storage, image_shape, n_channels, truth_dtype=np.uint8, crop=False):
    images = read_image_files(in_files)
    subject_data = [image for image in images]
    add_data_to_storage(data_storage, subject_data, n_channels)
    return data_storage

def write_3Dimage_data_to_file(patches, data_storage, image_shape, n_channels, truth_dtype=np.uint8, crop=False):
    subject_data = [image for image in patches]
    add_data_to_storage(data_storage, subject_data, n_channels)
    return data_storage

def read_image_files(image_files):
    image_list = list()
    for image in image_files:
        image_list.append(read_image(image))
    return image_list

def add_data_to_storage(data_storage, subject_data, n_channels):
    data_storage.append(np.asarray(subject_data)[...,np.newaxis])
    
def read_image(in_file):
#    print("Reading: {0}".format(in_file))
    image = imread(in_file,as_grey=True)
    return image

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

def preprocess3D(patch,img_depth,img_rows, img_cols):
    imgs_p = np.ndarray((patch.shape[0],img_depth, img_rows, img_cols), dtype=np.uint8)
    for i in range(patch.shape[0]):
        imgs_p[i] = resize(patch[i], (img_depth, img_rows, img_cols), preserve_range=True)    
    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

def getSegmentationArr(array , nClasses ):
    if len(array.shape)==4:
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
    if len(array.shape)==5:
        n,depth,height,width,tmp=array.shape
        seg_labels = np.zeros(( n,  depth, height , width  , nClasses ))
        for i in range(n):
            for c in range(nClasses):
                try:
                    img = array[i,:,:,:,0]
                    seg_labels[i, : , :, : , c ] = (img == c ).astype(int)
    #                print(seg_labels.shape)
                except Exception as e:
                    print(e)
        seg_labels = np.reshape(seg_labels, ( n, depth*width*height , nClasses ))
    return seg_labels

def open_hdf5_file(filename, readwrite="r"):
#    plt.imshow(datafile.root.data[1])
#    print(tables.open_file(filename, readwrite).root.data[1].shape)
    stack=tables.open_file(filename, readwrite)
    return stack

def extract_3D_patches(stack,patch_size,origins):
    (origin_row, origin_col, origin_dep)=origins
    patches=[]
    for o_r, o_c, o_d in zip(origin_row, origin_col, origin_dep):
        patches.append(stack[o_r:o_r+patch_size, o_c:o_c+patch_size, o_d:o_d+patch_size])
    return np.array(patches)

def sweep_3D_patches(stack,patch_size,origins):
    patches=[]
    for origin in origins:
        patches.append(stack[origin[0]:origin[0]+patch_size, origin[1]:origin[1]+patch_size, origin[2]:origin[2]+patch_size])
    return np.array(patches)
        
def transform_3Dpatch(data, truth, scale_deviation=None, flip=True):
    n_dim = len(truth.shape)
    if scale_deviation:
        scale_factor = random_scale_factor(1, std=scale_deviation)
    else:
        scale_factor = None
    if flip:
        flip_axis = random_flip_dimensions(n_dim)
    else:
        flip_axis = None
    trd,trt=distort_image(data,truth, flip_axis=flip_axis,scale_factor=scale_factor)

    return trd,trt

def scale_image(d,t, scale_factor):
    scale_factor=round(scale_factor[0],3)
    if scale_factor>1:
        sf=np.random.uniform(low=1, high=scale_factor,size=3)
        scd=zoom(d,sf)[0:d.shape[0],0:d.shape[1],0:d.shape[2]]
        sct=zoom(t,sf)[0:t.shape[0],0:t.shape[1],0:t.shape[2]]
        
    if scale_factor<=1:
        sf=(scale_factor,scale_factor,scale_factor)
        tmpd=zoom(d,sf)
        tmpt=zoom(t,sf)
        dif=d.shape[0]-tmpd.shape[0]
        if dif//2==dif/2:
            scd=np.pad(tmpd,dif//2,pad_with)
            sct=np.pad(tmpt,dif//2,pad_with)
        else:
            tmpd=tmpd[1:,1:,1:]
            tmpt=tmpt[1:,1:,1:]
            scd=np.pad(tmpd,(dif+1)//2,pad_with)
            sct=np.pad(tmpt,(dif+1)//2,pad_with)
    return scd,sct


def flip_image(d,t, axis):
    new_data = np.copy(d)
    new_truth = np.copy(t)
    for axis_index in axis:
        new_data = np.flip(new_data, axis=axis_index)
        new_truth = np.flip(new_truth, axis=axis_index)
    return new_data,new_truth


def random_flip_dimensions(n_dimensions):
    axis = list()
    for dim in range(n_dimensions):
        if random_boolean():
            axis.append(dim)
    return axis


def random_scale_factor(n_dim=3, mean=1, std=0.25):
    return np.random.normal(mean, std, n_dim)


def random_boolean():
    return np.random.choice([True, False])


def distort_image(d,t, flip_axis=None, scale_factor=None):
    if flip_axis:
        trd,trt = flip_image(d,t, flip_axis)
        if scale_factor is not None:
            trd,trt = scale_image(trd,trt, scale_factor)
    elif scale_factor is not None:
        trd,trt = scale_image(d,t, scale_factor)
    return trd,trt

def pad_with(vector, pad_width, iaxis, kwargs):    
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector