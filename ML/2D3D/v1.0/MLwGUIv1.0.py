#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 14:53:58 2018

@author: bertinetti
"""
from __future__ import print_function
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
from multiprocessing import Process,Queue,Pool
from collections import Counter
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from scipy import fftpack, ndimage, misc, signal
from scipy.ndimage.interpolation import shift
from scipy.misc import imrotate
from skimage import restoration,color
from skimage.io import imsave, imread
from skimage.transform import resize
from sklearn.feature_extraction import image
import cv2
from functools import partial
from PIL import Image, ImageTk
import webbrowser
import platform
import math
import os, glob, shutil
import sys, struct, gzip, random
import tensorflow as tf

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras import backend as K

from data import load_train_data, load_test_data, crop_no_black, preprocess,preprocess3D, getSegmentationArr, fetch_data_1dir, write_data_to_file, write_3Ddata_to_file, open_hdf5_file, read_image, extract_3D_patches, sweep_3D_patches,transform_3Dpatch
import Nmodels
import Nmodels3D

mfw,mfh=600,450
nbfw,nbfh=550,40
config = dict()
config['tr_p']='train'
config['test_p']='test'
config['selfold']='Augmented'
config['augm_tmpf']='tmp'
config['image_p']=os.path.join(config['tr_p'],'image')
config['label_p']=os.path.join(config['tr_p'],'label')
config['model_p']=os.path.join(config['tr_p'],'models')
config['trimg_npy']='train_images.hdf5'
config['trlab_npy']='train_labels.hdf5'
config['test_npy']='test_imgs.hdf5'
config['test_id_npy']='imgs_id_test.npy'

tr_p=config['tr_p']
test_p=config['test_p']
selfold=config['selfold']
tmpf=config['augm_tmpf']
image_p=config['image_p']
label_p=config['label_p']
model_p=config['model_p']

train_npy=config['trimg_npy']
labels_npy=config['trlab_npy']
test_npy=config['test_npy']
test_id_npy=config['test_id_npy']


class App(tk.Frame):
    def __init__(self,master):
        tk.Frame.__init__(self,master)
        self.master.title("Segmentation with neuronal networks")
        self.master.resizable(False,False)
        self.master.tk_setPalette(background='#ececec')
        x=int((self.master.winfo_screenwidth()-self.master.winfo_reqwidth())/1.5)
        y=int((self.master.winfo_screenheight()-self.master.winfo_reqheight())/2)
        self.master.geometry('{}x{}'.format(x,y))
        
        #Create the common frame
        self.bfiles=tk.Frame(master, width = mfw, height=mfh*len(mainfields))
        self.fpath=tk.Label(self.bfiles,text=fieldsprep[0][0])
        self.fp=tk.StringVar(self.bfiles)
        self.fpath_val=tk.Entry(self.bfiles,textvariable=self.fp)
        self.browse_button = tk.Button(self.bfiles,text="Browse", fg="green",command=self.browseSt)
    
        self.fpath.grid(row=0,column=0)
        self.fpath_val.grid(row=0,column=1)
        self.browse_button.grid(row=0,column=2)
        

        self.fext=tk.Label(self.bfiles,text=fieldsprep[1][0])
        self.fe=tk.StringVar(self.bfiles,value=fieldsprep[1][1])
        self.fext_val=tk.Entry(self.bfiles,textvariable=self.fe)
        self.fext.grid(row=2,column=0)
        self.fext_val.grid(row=2,column=1)


        self.bfiles.pack(side="top")
        
        
        #Create Notebooks
        self.nbfr=tk.Frame(master, width = mfw, height=mfh)
        self.nbfr.pack(side="top")
        self.n=ttk.Notebook(self.nbfr)
        self.prep_frame=tk.Frame(self.n, width = mfw, height=mfh-40)   
        self.train_frame=tk.Frame(self.n, width = mfw, height=mfh-40)
        self.n.add(self.prep_frame, text='Data Preparation')
        self.n.add(self.train_frame, text='Train and Predict')
        self.n.pack()
        
        
        # Create the main containers to pre-process the data
        tk.Label(self.prep_frame,text="Data Prepration").grid(row=0)
        self.cen_frame_prep=tk.Frame(self.prep_frame, width = nbfw, height=nbfh*len(prepfields[1:]), pady=3)
        self.btm_frame_prep=tk.Frame(self.prep_frame, width = nbfw, height=nbfh, pady=3)
        
        # Layout the containers
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.cen_frame_prep.grid(row=1, sticky="ew")
        self.btm_frame_prep.grid(row=2, sticky="ew")

        self.entsprepr=self.makeFrame(self.cen_frame_prep,fieldsprep)
        
        
        # Widgets of the bottom frame
        # Widgets of the bottom frame
        self.augm_button = tk.Button(self.btm_frame_prep,text="Augment", fg="Red",command=partial(self.AugmentDS,self.entsprepr))
        self.augm_3Dbutton = tk.Button(self.btm_frame_prep,text="Augment 3D", fg="Red",command=partial(self.AugmentDS3D,self.entsprepr))
        self.quit_button = tk.Button(self.btm_frame_prep, text='Quit', command=self.cancel_b)
        self.save_button= tk.Button(self.btm_frame_prep, text='Create train .hdf5', command=partial(self.Savehdf5,self.entsprepr,'trlab'))
        self.savetst_button= tk.Button(self.btm_frame_prep, text='Create test .hdf5', command=partial(self.Savehdf5,self.entsprepr,'test'))
        
        self.augm_button.grid(row=0,column=0)
        self.augm_3Dbutton.grid(row=0,column=1)
        self.save_button.grid(row=0,column=2)
        self.savetst_button.grid(row=0,column=3)
        self.quit_button.grid(row=0,column=4)

        root.bind('<Return>', (lambda event: self.fetch(self.entsprepr))) 

        # Create the main containers for the FT destriping notebook
        tk.Label(self.train_frame,text="Train Model").grid(row=0)
        self.cen_frame_train=tk.Frame(self.train_frame, width = nbfw, height=nbfh*len(trfields[1:]), pady=3)
        self.trmodel_frame=tk.Frame(self.train_frame, width = nbfw, height=nbfh, pady=3)
        self.btm_frame_train=tk.Frame(self.train_frame, width = nbfw, height=nbfh, pady=3)
        
        # Layout the containers
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.cen_frame_train.grid(row=1, sticky="ew")
        self.trmodel_frame.grid(row=2, sticky="ew")
        self.btm_frame_train.grid(row=3, sticky="ew")

        self.trmodellab=tk.Label(self.trmodel_frame,text='Model:')
        self.trmodelstr=tk.StringVar(self.trmodel_frame)
        self.trmodelstr.set('MksNet')
        self.trmodelopt = tk.OptionMenu(self.trmodel_frame, self.trmodelstr, 'MksNet', 'UNet','UNet2', 'unet_model_3d','isensee2017_3D','MksNet3D', 'SegNet','isensee2017', 'VGGSegNet','FCN8Net','FCN32Net')
        self.trmodellab.grid(row=0,column=0)
        self.trmodelopt.grid(row=0,column=1)
        
        self.entstrain=self.makeFrame(self.cen_frame_train,fieldstrain)
        
        # Widgets of the bottom frame
        self.train_button = tk.Button(self.btm_frame_train,text="Train", fg="Red",command=partial(self.TrainModel,self.entstrain))
        self.train3D_button = tk.Button(self.btm_frame_train,text="Train 3D", fg="Red",command=partial(self.TrainModel3D,self.entstrain))
        self.quit_button = tk.Button(self.btm_frame_train, text='Quit', command=self.cancel_b)
        self.predict_button = tk.Button(self.btm_frame_train,text="Predict", fg="Red",command=partial(self.Predict,self.entstrain))
        self.predict3D_button = tk.Button(self.btm_frame_train,text="Predict 3D", fg="Red",command=partial(self.Predict3D,self.entstrain))
        
        self.train_button.grid(row=0,column=0)
        self.train3D_button.grid(row=0,column=1)
        self.quit_button.grid(row=0,column=2)
        self.predict_button.grid(row=0,column=3)
        self.predict3D_button.grid(row=0,column=4)
        
        root.bind('<Tab>', (lambda event: self.fetch(self.entstrain))) 

# Create .npy train dataset


    def cancel_b(self):
        self.quit()
        self.master.destroy()
        

    def browseSt(self):
        idir='/'
        if 'Win' in platform.system():
            idir = 'W:/'
        if 'Darwin' in platform.system():
            idir = "/Volumes/Data/Luca_Work/MPI/Science/Coding/Python/Segm"
        if 'Linux' in platform.system():
            idir = '/usr/people/home/bertinetti/Data/test_ML'
        dirname = tk.filedialog.askdirectory(initialdir = idir ,title = 'Select root path for images')
        if dirname:
            self.fp.set(dirname)
            
    def AugmentDS(self,Augentries):
        datagen = ImageDataGenerator(
#            featurewise_center=False,
#            featurewise_std_normalization=False,
#            samplewise_center=False,
#            samplewise_std_normalization=False,
#            zca_whitening=True,
#            rescale=None,
            rotation_range=3,
            width_shift_range=0.08,
            height_shift_range=0.08,
            shear_range=0.07,
            zoom_range=0.07,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='constant',
            cval=0.
            )
        args=self.arggen(Augentries)
#        trimgs,labimgs,data_path,num
        config['data_path'],config['fext'],num,cr=args[0],args[1],int(args[2]),float(args[3])
        fext=config['fext']
        data_path=config['data_path']
        self.Savefiles(data_path,fext,'trlab')
        imhdf5=open_hdf5_file(config['image_hdf5_path'])
        trimgs=np.squeeze(imhdf5.root.data,axis=3)
        imhdf5.close()
        labhdf5=open_hdf5_file(config['label_hdf5_path'])
        labimgs=np.squeeze(labhdf5.root.truth,axis=3)
        labhdf5.close()
        nLabels=np.max(labimgs)
        print('estimated number of lables:',nLabels)
        if nLabels>1:
            labimgstmp=[]
            for i in range(1,nLabels+1):
                labimgstmp.append(np.ma.masked_not_equal(labimgs,i).filled(0)/i)
            labimgstmp=np.array(labimgstmp)
                
        imgshape=trimgs[0].shape
        print(imgshape)
        print('-'*30)
        print('Augmenting train and labels dataset: ',num,'replica per image...')
        print('-'*30)
    #    seed = np.random.randint(10000)
        seed=np.random.randint(10000,size=2*len(trimgs)*num)
        if tmpf in sorted(os.listdir(config['image_path'])):
           shutil.rmtree(os.path.join(config['image_path'],tmpf), ignore_errors=True)
           shutil.rmtree(os.path.join(config['label_path'],tmpf), ignore_errors=True)
        os.makedirs(os.path.join(config['image_path'],tmpf))
        os.makedirs(os.path.join(config['label_path'],tmpf))
        global batchdata
        batchdata=[]
        j=0
        for x in trimgs:
            x[x==0]=1
            x = x.reshape((1,) + x.shape+(1,))
            # the .flow() command below generates batches of randomly transformed images
            # and saves the results to the `preview/` directory
            i = 0
            
            for batch in datagen.flow(x, batch_size=1,seed=seed[j]):
                self.save_tif(data_path,os.path.join(image_p,tmpf),'img',batch[0,:,:,0].astype('uint8'),seed[i+j*2*num],fext)
                i += 1
                if i >= 2*num:
                    break  # otherwise the generator would loop indefinitely
            j +=1

        if nLabels>1:
            for k in range(1,nLabels+1):
                os.makedirs(os.path.join(config['label_path'],tmpf,str(k)))
                j=0
                for y in labimgstmp[k-1]:
                    y = y.reshape((1,) + y.shape+(1,))
                    i = 0
                    for batch in datagen.flow(y, batch_size=1,seed=seed[j]):
                        self.save_tif(data_path,os.path.join(label_p,tmpf,str(k)),'img',batch[0,:,:,0].astype('uint8'),seed[i+j*2*num],fext)
                        batchdata.append(batch[0,:,:,0])
                        i += 1
                        if i >= 2*num:
                            break  # otherwise the generator would loop indefinitely
                    j +=1
            imglist=[f for f in sorted(os.listdir(os.path.join(config['image_path'],tmpf))) if fext in f]
            for n in range(len(imglist)):
                tmp=sum(read_image(os.path.join(config['label_path'],tmpf,str(k),imglist[n]))*k for k in range(1,nLabels+1))
                self.save_tif(data_path,os.path.join(label_p,tmpf),'img',tmp.astype('uint8'),imglist[n].split('.')[0][-4:],fext)
            for k in range(1,nLabels+1):
                shutil.rmtree(os.path.join(config['label_path'],tmpf,str(k)), ignore_errors=True)                
        else:
            j=0
            for y in labimgs:
                y = y.reshape((1,) + y.shape+(1,))
                i = 0
                for batch in datagen.flow(y, batch_size=1,seed=seed[j]):
    
                    self.save_tif(data_path,os.path.join(label_p,tmpf),'img',batch[0,:,:,0].astype('uint8'),seed[i+j*2*num],fext)
                    batchdata.append(batch[0,:,:,0])
                    i += 1
                    if i >= 2*num:
                        break  # otherwise the generator would loop indefinitely
                j +=1
        self.Savefiles(data_path,fext,'trlab',subtask='augtmp')
#        create_train_data(data_path,os.path.join(image_p,tmpf),os.path.join(label_p,tmpf),fext)
        imhdf5=open_hdf5_file(config['image_hdf5_path'])
        tmptr=np.squeeze(imhdf5.root.data,axis=3)
        imhdf5.close()
        labhdf5=open_hdf5_file(config['label_hdf5_path'])
        tmplab=np.squeeze(labhdf5.root.truth,axis=3)
        labhdf5.close()
        print(imgshape,cr)
        lencrop=int(((imgshape[0]*cr)//16)*16),int(((imgshape[1]*cr)//16)*16)
        print(lencrop)
#        delta=imgshape[0]-lencrop[0],imgshape[1]-lencrop[1]
#        print(delta)
        seltr=[]
        sellab=[]
        j=0
        for i,img in enumerate(tmptr):
            tmpres=crop_no_black(tmptr[i],tmplab[i],lencrop)
            if tmpres is not None:
                seltr.append(tmpres[0])
                sellab.append(tmpres[1])
                j += 1
                if j > len(trimgs)*(num+1):
                    break
        seltr=np.array(seltr)
        sellab=np.array(sellab)
        print(seltr.shape,sellab.shape)
        if selfold in sorted(os.listdir(os.path.join(data_path,image_p))):
           shutil.rmtree(os.path.join(data_path,image_p,selfold), ignore_errors=True)
           shutil.rmtree(os.path.join(data_path,label_p,selfold), ignore_errors=True)        
        os.makedirs(os.path.join(data_path,image_p,selfold))
        os.makedirs(os.path.join(data_path,label_p,selfold))
        for i in range(len(seltr)):
            self.save_tif(data_path,os.path.join(image_p,selfold),'img',seltr[i],i,fext)
            self.save_tif(data_path,os.path.join(label_p,selfold),'img',sellab[i],i,fext)
#        create_train_data(data_path,image_p,label_p,fext)
        if tmpf in sorted(os.listdir(os.path.join(data_path,image_p))):
           shutil.rmtree(os.path.join(data_path,image_p,tmpf), ignore_errors=True)
           shutil.rmtree(os.path.join(data_path,label_p,tmpf), ignore_errors=True)     
        self.Savefiles(data_path,fext,'trlab',subtask='augm')
        print('Done')
        return

    def AugmentDS3D(self,Augentries):
        args=self.arggen(Augentries)
#        trimgs,labimgs,data_path,num
        config['data_path'],config['fext'],num,cr=args[0],args[1],int(args[2]),float(args[3])
        fext=config['fext']
        data_path=config['data_path']    
        fold3D='3D'
        self.Savefiles(data_path,fext,'trlab')
        imhdf5=open_hdf5_file(config['image_hdf5_path'])
        trimgs=np.squeeze(imhdf5.root.data,axis=3)
        imhdf5.close()
        labhdf5=open_hdf5_file(config['label_hdf5_path'])
        labimgs=np.squeeze(labhdf5.root.truth,axis=3)
        labhdf5.close()
        nLabels=np.max(labimgs)
        print('estimated number of lables:',nLabels)
        print('stack shape',trimgs.shape)
        (nd,nr,nc)=trimgs.shape
        print(trimgs.shape)
        patch_size=64
        origin_row = np.random.randint(0, nr-patch_size, num)
        origin_col = np.random.randint(0, nc-patch_size, num)
        origin_dep = np.random.randint(0, nd-patch_size, num)
        origins=np.array((origin_dep,origin_row,origin_col))
        trimgs_patches=extract_3D_patches(trimgs,patch_size,origins)  
        labs_patches=extract_3D_patches(labimgs,patch_size,origins)
        if nLabels>1:
            for i in range(trimgs_patches.shape[0]):
                    trimgs_patches[i],labs_patches[i]= transform_3Dpatch(trimgs_patches[i],labs_patches[i])
        if nLabels==1:
            for i in range(trimgs_patches.shape[0]):
                    trimgs_patches[i],labs_patches[i]= transform_3Dpatch(trimgs_patches[i],labs_patches[i],scale_deviation=0.15)
        print('patches shape',trimgs_patches.shape)
        if fold3D in sorted(os.listdir(os.path.join(data_path,image_p))):
           shutil.rmtree(os.path.join(data_path,image_p,fold3D), ignore_errors=True)
           shutil.rmtree(os.path.join(data_path,label_p,fold3D), ignore_errors=True)        
        os.makedirs(os.path.join(data_path,image_p,fold3D))
        os.makedirs(os.path.join(data_path,label_p,fold3D))
        for i in range(trimgs_patches.shape[0]):
            os.makedirs(os.path.join(data_path,image_p,fold3D,str(i)))
            os.makedirs(os.path.join(data_path,label_p,fold3D,str(i)))
            for k in range(trimgs_patches.shape[1]):
                self.save_tif(data_path,os.path.join(image_p,fold3D,str(i)),'img',trimgs_patches[i,k],k,fext)
                self.save_tif(data_path,os.path.join(label_p,fold3D,str(i)),'img',labs_patches[i,k],k,fext)
        imglabpatches=np.array((trimgs_patches,labs_patches))
        self.Savefiles3D(data_path,imglabpatches,fext)
        
        return
        
    def Savehdf5(self,entries,task):
        args=self.arggen(entries)
        config['data_path'],config['fext']=args[0],args[1]
        fext=config['fext']
        data_path=config['data_path']
        self.Savefiles(data_path,fext,task)

    def Savefiles(self,data_path,fext,task,subtask=None):
        if task =='trlab':
            config['image_path']=os.path.join(data_path,image_p)
            config['label_path']=os.path.join(data_path,label_p)
            outim=config['image_hdf5_path']=os.path.join(config['data_path'],config['trimg_npy'])
            outlab=config['label_hdf5_path']=os.path.join(config['data_path'],config['trlab_npy'])
            if subtask == None:
                if selfold in sorted(os.listdir(config['image_path'])):
                    print('Augmented data found. Saving augmented data instead of original ones')
                    imgs=fetch_data_1dir(os.path.join(config['image_path'],selfold),fext)
                    lbls=fetch_data_1dir(os.path.join(config['label_path'],selfold),fext)
    #                create_train_data(data_path,os.path.join(image_p,selfold),os.path.join(label_p,selfold),fext)             
                else:
                    imgs=fetch_data_1dir(config['image_path'],fext)
                    lbls=fetch_data_1dir(config['label_path'],fext)
            if subtask=='augm':
                    imgs=fetch_data_1dir(os.path.join(config['image_path'],selfold),fext)
                    lbls=fetch_data_1dir(os.path.join(config['label_path'],selfold),fext)            
            if subtask=='augtmp': 
                    imgs=fetch_data_1dir(os.path.join(config['image_path'],tmpf),fext)
                    lbls=fetch_data_1dir(os.path.join(config['label_path'],tmpf),fext)  
#                create_train_data(data_path,image_p,label_p,fext)
            config['img_comp']=write_data_to_file(imgs,outim,'data')
            config['lab_comp']=write_data_to_file(lbls,outlab,'truth')
            print('Training images hdf5 file written to:',outim)
            print('Training labels hdf5 file written to:',outlab)
        
        if task =='test':
            config['test_path']=os.path.join(data_path,test_p)
            outtest=os.path.join(config['data_path'],config['test_npy'])
            test=fetch_data_1dir(config['test_path'],fext)
            config['test_comp']=write_data_to_file(test,outtest,'test')
            print('Test images hdf5 file written to:',outtest)
            imgs_id = np.ndarray((len(test), ), dtype=np.int32)
            i=0
            for image_name in test:
                img_id = int(image_name.split('.')[0][-4:].lstrip('0'))
                imgs_id[i] = img_id
                i += 1
            np.save(os.path.join(data_path,'imgs_id_test.npy'), imgs_id)
#            create_test_data(data_path,test_p,fext)

    def Savefiles3D(self,data_path,patches,fext):
        config['image_path']=os.path.join(data_path,image_p)
        config['label_path']=os.path.join(data_path,label_p)
        outim=config['image_hdf5_path']=os.path.join(config['data_path'],config['trimg_npy'])
        outlab=config['label_hdf5_path']=os.path.join(config['data_path'],config['trlab_npy'])  
        imgs=patches[0]
        labs=patches[1]
        write_3Ddata_to_file(imgs,outim,'data')
        write_3Ddata_to_file(labs,outlab,'truth')
        
            
    def TrainModel(self,TMentries):
        args=self.arggen(TMentries)
        data_path,mName,epchs,vspl,nCl,lr,bs=args[0],args[2],int(args[3]),float(args[4]),int(args[5]),float(args[6]),int(args[7])
        if nCl>1:
            nLab= nCl +1
        else: 
            nLab=nCl
        config['data_path']=data_path
        config['image_path']=os.path.join(data_path,image_p)
        config['label_path']=os.path.join(data_path,label_p)
        config['image_hdf5_path']=os.path.join(config['data_path'],config['trimg_npy'])
        config['label_hdf5_path']=os.path.join(config['data_path'],config['trlab_npy']) 
        print('-'*30)
        print('Loading and preprocessing train data...')
        print('-'*30)
        imgsopen=open_hdf5_file(config['image_hdf5_path'])
        imgs_train = np.squeeze(imgsopen.root.data,axis=3)
        imgsopen.close()
        labopen=open_hdf5_file(config['label_hdf5_path'])
        imgs_mask_train = np.squeeze(labopen.root.truth,axis=3)
        labopen.close()
        img_rows, img_cols = imgs_train[0].shape[0],imgs_train[0].shape[1]
        
        imgs_train = preprocess(imgs_train,img_rows, img_cols)
        imgs_mask_train = preprocess(imgs_mask_train,img_rows, img_cols)
        imgs_train = imgs_train.astype('float32')
        mean = np.mean(imgs_train)  # mean for data centering
        std = np.std(imgs_train)  # std for data normalization
    
        imgs_train -= mean
        imgs_train /= std
    
#        imgs_mask_train = imgs_mask_train.astype('float32')
#        imgs_mask_train /= 255.  # scale masks to [0, 1]
#        imgs_mask_train *= (nCl)  # generates the labes as integers
        imgs_mask_train = imgs_mask_train.astype('uint8')
        if np.max(imgs_mask_train) != nCl:
            print('Warning: the number of classes does not match the intesities of the label images')
        if nLab>1:
            imgs_mask_train = getSegmentationArr(imgs_mask_train , nLab)
#            global imgs_mask_train2
#            imgs_mask_train2 = np.copy(imgs_mask_train)
        else:
            imgs_mask_train[imgs_mask_train > 0.5] = 1
            imgs_mask_train[imgs_mask_train <= 0.5] = 0
        print(imgs_mask_train.shape)
        print('-'*30)
        print('Creating and compiling model...')
        print('-'*30)
        
        
    #    model = get_unet(imgs_train[0].shape)
    #    model = Model4(imgs_train[0].shape)
        if os.path.exists(os.path.join(data_path,model_p)) and len([x for x in os.listdir(os.path.join(data_path,model_p)) if ('.hdf5') in x])>0:
            print('loading weights and compiling the model')
            latest=max(glob.glob(os.path.join(data_path,model_p,'*.hdf5')),key=os.path.getctime)
            model= getattr(Nmodels,mName)(nLab,imgs_train[0].shape,latest,lr)
        else:
            if not os.path.exists(os.path.join(data_path,model_p)):
                os.makedirs(os.path.join(data_path,model_p))
            model= getattr(Nmodels,mName)(nLab,imgs_train[0].shape,'',lr)
        
    #    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True) 
        model_checkpoint = ModelCheckpoint(os.path.join(data_path,model_p,mName+'weights.ep{epoch:02d}-il{loss:.3f}-vl{val_loss:.3f}.hdf5'), monitor='loss',verbose=1, save_best_only=True)
    
        print('-'*30)
        print('Fitting model...')
        print('-'*30)
    #    model.fit(imgs_train, imgs_mask_train, batch_size=34, nb_epoch=20, verbose=1, shuffle=True,
    #              validation_split=0.2,
    #              callbacks=[model_checkpoint])         
        model.fit(imgs_train, imgs_mask_train, batch_size=bs, epochs=epchs, verbose=1,
                  validation_split=vspl, 
                  shuffle=True, 
                  callbacks=[model_checkpoint])
        return

    def TrainModel3D(self,TMentries):
        args=self.arggen(TMentries)
        data_path,mName,epchs,vspl,nCl,lr,bs=args[0],args[2],int(args[3]),float(args[4]),int(args[5]),float(args[6]),int(args[7])
        if nCl>1:
            nLab= nCl +1
        else: 
            nLab=nCl
        config['data_path']=data_path
        config['image_path']=os.path.join(data_path,image_p)
        config['label_path']=os.path.join(data_path,label_p)
        config['image_hdf5_path']=os.path.join(config['data_path'],config['trimg_npy'])
        config['label_hdf5_path']=os.path.join(config['data_path'],config['trlab_npy']) 
        print('-'*30)
        print('Loading and preprocessing train data...')
        print('-'*30)
        imgsopen=open_hdf5_file(config['image_hdf5_path'])
        imgs_train = np.squeeze(imgsopen.root.data,axis=4)
        imgsopen.close()
        labopen=open_hdf5_file(config['label_hdf5_path'])
        imgs_mask_train = np.squeeze(labopen.root.truth,axis=4)
        labopen.close()
        (img_depth, img_rows, img_cols) = (imgs_train[0].shape)
        
        imgs_train2 = []
        for patch in imgs_train: 
            patchtmp=preprocess(patch, img_rows, img_cols)
            imgs_train2.append(patchtmp)
        imgs_mask_train2 = []
        for patch in imgs_mask_train:
            patchtmp=preprocess(patch, img_rows, img_cols)
            imgs_mask_train2.append(patchtmp)
        imgs_train = np.array(imgs_train2).astype('float32')
        mean = np.mean(imgs_train)  # mean for data centering
        std = np.std(imgs_train)  # std for data normalization
    
        imgs_train -= mean
        imgs_train /= std
    
#        imgs_mask_train = imgs_mask_train.astype('float32')
#        imgs_mask_train /= 255.  # scale masks to [0, 1]
#        imgs_mask_train *= (nCl)  # generates the labes as integers
        imgs_mask_train = np.array(imgs_mask_train2).astype('uint8')
        print('Size of the training data:',imgs_mask_train.shape)
        if np.max(imgs_mask_train) != nCl:
            print('Warning: the number of classes does not match the intesities of the label images')
        if nLab>1:
            imgs_mask_train = getSegmentationArr(imgs_mask_train , nLab)
#            global imgs_mask_train2
#            imgs_mask_train2 = np.copy(imgs_mask_train)
        else:
            imgs_mask_train[imgs_mask_train > 0.5] = 1
            imgs_mask_train[imgs_mask_train <= 0.5] = 0
        print(imgs_mask_train.shape)
        print('-'*30)
        print('Creating and compiling model...')
        print('-'*30)
        
        
    #    model = get_unet(imgs_train[0].shape)
    #    model = Model4(imgs_train[0].shape)
        if os.path.exists(os.path.join(data_path,model_p)) and len([x for x in os.listdir(os.path.join(data_path,model_p)) if ('.hdf5') in x])>0:
            print('loading weights and compiling the model')
            latest=max(glob.glob(os.path.join(data_path,model_p,'*.hdf5')),key=os.path.getctime)
            model= getattr(Nmodels3D,mName)(nLab,imgs_train[0].shape,latest,lr)
        else:
            if not os.path.exists(os.path.join(data_path,model_p)):
                os.makedirs(os.path.join(data_path,model_p))
            model= getattr(Nmodels3D,mName)(nLab,imgs_train[0].shape,'',lr)
        
    #    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True) 
        model_checkpoint = ModelCheckpoint(os.path.join(data_path,model_p,mName+'weights.ep{epoch:02d}-il{loss:.3f}-vl{val_loss:.3f}.hdf5'), monitor='loss',verbose=1, save_best_only=True)
    
        print('-'*30)
        print('Fitting model...')
        print('-'*30)
    #    model.fit(imgs_train, imgs_mask_train, batch_size=34, nb_epoch=20, verbose=1, shuffle=True,
    #              validation_split=0.2,
    #              callbacks=[model_checkpoint])         
        model.fit(imgs_train, imgs_mask_train, batch_size=bs, epochs=epchs, verbose=1,
                  validation_split=vspl, 
                  shuffle=True, 
                  callbacks=[model_checkpoint])
        return
    
    def Predict(self,TMentries):
        args=self.arggen(TMentries)
        data_path,mName,epchs,vspl,nCl,lr,bs=args[0],args[2],int(args[3]),float(args[4]),int(args[5]),float(args[6]),int(args[7])
        if nCl>1:
            nLab= nCl +1
        else: 
            nLab=nCl
        print('-'*30)
        print('Loading and preprocessing test data...')
        print('-'*30)
        
        test_open=open_hdf5_file(os.path.join(data_path,config['test_npy']))
        imgs_test_tot = np.squeeze(test_open.root.test,axis=3)
        test_open.close()
        imgs_id_test_tot=np.load(os.path.join(data_path,test_id_npy))
        img_rows, img_cols = imgs_test_tot[0].shape
        print('-'*30)
        print('Loading saved weights...')
        print('-'*30)
        print(max(glob.glob(os.path.join(data_path,model_p,'*.hdf5')),key=os.path.getctime))
        latest=max(glob.glob(os.path.join(data_path,model_p,'*.hdf5')),key=os.path.getctime)
        imgtmp=imgs_test_tot[0].reshape(img_rows, img_cols,1)
        model= getattr(Nmodels,mName)(nLab,imgtmp.shape,latest,lr)
        i=0
        for i in range(len(imgs_test_tot)):
            print('image',i+1,'of',len(imgs_test_tot))
            imgs_test = preprocess(imgs_test_tot[i].reshape(1,img_rows, img_cols),img_rows, img_cols)
            imgs_id_test=imgs_id_test_tot[i]
            print('processing image',imgs_id_test)
        
            imgs_test = imgs_test.astype('float32')
            mean = np.mean(imgs_test)  # mean for data centering
            std = np.std(imgs_test)  # std for data normalization
            imgs_test -= mean
            imgs_test /= std
        
        #    model.load_weights(os.path.join(data_path,model_p,'weights.h5'))
        #    print(len(imgs_test))
        #    op_shape=model.output_shape
        #    t=list(op_shape)
        #    t[0]=3
        #    op_shape=tuple(t)
        
            print('-'*30)
            print('Predicting labels on test data...')
            print('-'*30)
            imgs_mask_test=model.predict(imgs_test, verbose=1)
        
            print('-' * 30)
            print('Saving predicted labels to files...')
            print('-' * 30)
            pred_dir = os.path.join(data_path,'preds_'+mName+'_'+latest.split('weights',1)[1])
            if not os.path.exists(pred_dir):
                os.mkdir(pred_dir)

            if nLab>1:
                imgs_mask_test = imgs_mask_test.reshape(( img_rows, img_cols , nLab ) )
                imgs_mask_test= imgs_mask_test.argmax( axis=2 ).astype(np.uint8)
#                    imagep = (image * 255.).astype(np.uint8)
            else:
                imgs_mask_test = (imgs_mask_test * 255.).astype(np.uint8)
            imsave(os.path.join(pred_dir, str(imgs_id_test) + '_pred.tif'), imgs_mask_test)

    def Predict3D(self,TMentries):
        args=self.arggen(TMentries)
        data_path,mName,epchs,vspl,nCl,lr,bs=args[0],args[2],int(args[3]),float(args[4]),int(args[5]),float(args[6]),int(args[7])   
        if nCl>1:
            nLab= nCl +1
        else: 
            nLab=nCl
        print('-'*30)
        print('Loading and preprocessing test data...')
        print('-'*30)
        
        test_open=open_hdf5_file(os.path.join(data_path,config['test_npy']))
        imgs_test_tot = np.squeeze(test_open.root.test,axis=3)
        test_open.close()
        img_depth, img_rows, img_cols = imgs_test_tot.shape
        print('-'*30)
        print('Loading saved weights...')
        print('-'*30)
        print(max(glob.glob(os.path.join(data_path,model_p,'*.hdf5')),key=os.path.getctime))
        latest=max(glob.glob(os.path.join(data_path,model_p,'*.hdf5')),key=os.path.getctime)
        pred_dir = os.path.join(data_path,'preds_'+mName+'_'+latest.split('weights',1)[1])

        CPU=False
        GPU=True
        num_cores =  os.cpu_count()-2
        if GPU:
            num_GPU = 1
            num_CPU = 1
        if CPU:
            num_CPU = 1
            num_GPU = 0
        
        configK = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
                inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
                device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
        session = tf.Session(config=configK)
        K.set_session(session)
        if not GPU:
            vol_test = preprocess3D(imgs_test_tot.reshape(1,img_depth, img_rows, img_cols),img_depth, img_rows, img_cols)
            vol_test = vol_test.astype('float32')
            mean = np.mean(vol_test)  # mean for data centering
            std = np.std(vol_test)  # std for data normalization
            vol_test -= mean
            vol_test /= std
            voltmp=vol_test.reshape(img_depth, img_rows, img_cols,1)
            model= getattr(Nmodels3D,mName)(nLab,voltmp.shape,latest)
            print('-'*30)
            print('Predicting labels on test data...')
            print('-'*30)
            imgs_mask_test=model.predict(vol_test, verbose=1)
            if not os.path.exists(pred_dir):
                os.mkdir(pred_dir)
            if nLab>1:
                imgs_mask_test = imgs_mask_test.reshape(( img_depth, img_rows, img_cols, nLab ) )
                vol_mask= imgs_mask_test.argmax( axis=3 ).astype(np.uint8)
            else:
                vol_mask = np.squeeze((imgs_mask_test * 255.).astype(np.uint8),axis=4)
        else:
            p_size=64
            imgs_test_totshft=np.roll(np.roll(np.roll(imgs_test_tot,p_size//2,axis=2),p_size//2,axis=1),p_size//2,axis=0)
            volmask1=self.pred_vol(imgs_test_tot,nLab,mName,p_size,latest)
            volmask2=self.pred_vol(imgs_test_totshft,nLab,mName,p_size,latest)
            volmask2shft=np.roll(np.roll(np.roll(volmask2,-p_size//2,axis=2),-p_size//2,axis=1),-p_size//2,axis=0)
            vol_mask=volmask1+volmask2shft
            vol_mask[vol_mask>0.1]=255
#            vol_mask=volmask2shft
            
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        for i in range(vol_mask.shape[0]):
            imsave(os.path.join(pred_dir, str(i) + '_pred.tif'), vol_mask[i].astype(np.uint8))
            i+=1

    def pred_vol(self,volume,nLab,mName,ps,latest):
        img_depth, img_rows, img_cols=volume.shape
        patch_size=ps
        origin_row = np.array(range(img_rows//patch_size))*patch_size
        origin_col = np.array(range(img_cols//patch_size))*patch_size
        origin_dep = np.array(range(img_depth//patch_size))*patch_size
        if img_rows//patch_size!=img_rows/patch_size:
            origin_row=np.append(origin_row,img_rows-patch_size)
        if img_cols//patch_size!=img_cols/patch_size:
            origin_col=np.append(origin_col,img_cols-patch_size)
        if img_depth//patch_size!=img_depth/patch_size:
            origin_dep=np.append(origin_dep,img_depth-patch_size)
           
        origins=[]
        for o_d in origin_dep:
            for o_r in origin_row:
                for o_c in origin_col:
                    origins.append(np.array((o_d,o_r,o_c)))
        origins=np.array(origins)
        patches3D=sweep_3D_patches(volume,patch_size,origins)
        pdepth,prows,pcols=patches3D[0].shape
        imgtmp=patches3D[0].reshape(pdepth,prows,pcols,1)
        model= getattr(Nmodels3D,mName)(nLab,imgtmp.shape,latest)
        i=0
        vol_mask = np.zeros(( img_depth, img_rows, img_cols ))
        for patch in patches3D:
            print('patch',i+1,'of',len(patches3D))
            patch_test = preprocess3D(patches3D[i].reshape(1,pdepth,prows,pcols),pdepth,prows,pcols)
            patch_test = patch_test.astype('float32')
            mean = np.mean(patch_test)  # mean for data centering
            std = np.std(patch_test)  # std for data normalization
            patch_test -= mean
            patch_test /= std
        
        #    model.load_weights(os.path.join(data_path,model_p,'weights.h5'))
        #    print(len(imgs_test))
        #    op_shape=model.output_shape
        #    t=list(op_shape)
        #    t[0]=3
        #    op_shape=tuple(t)
        
            print('-'*30)
            print('Predicting labels on test data...')
            print('-'*30)
            imgs_mask_test=model.predict(patch_test, verbose=1)
            if nLab>1:
                imgs_mask_test = imgs_mask_test.reshape(( pdepth,prows,pcols, nLab ) )
                imgs_mask_test= imgs_mask_test.argmax( axis=3 ).astype(np.uint8)
                vol_mask[origins[i][0]:origins[i][0]+patch_size, origins[i][1]:origins[i][1]+patch_size, origins[i][2]:origins[i][2]+patch_size]=imgs_mask_test
#                    imagep = (image * 255.).astype(np.uint8)
            else:
                b=np.squeeze(imgs_mask_test,axis=4)
                b[b>0.4]=1
                vol_mask[origins[i][0]:origins[i][0]+patch_size, origins[i][1]:origins[i][1]+patch_size, origins[i][2]:origins[i][2]+patch_size]=b
            i+=1
        i=0
        return vol_mask

    def makeFrame(self,parent,fieldz):
        entries=[]
        entries.append((fieldz[0][0],self.fpath_val))
        entries.append((fieldz[1][0],self.fext_val))        
        if len(fieldz)>2 and fieldz[2][0]==fieldstrain[2][0]:
            entries.append(('Train Model',self.trmodelstr))
        for i in range(2,len(fieldz)):
           lab = tk.Label(parent, width=25, text=fieldz[i][0], anchor='w')
           ent_txt=tk.StringVar(parent,value=fieldz[i][1])
           ent = tk.Entry(parent,textvariable=ent_txt)
           ent.config(justify=tk.RIGHT)
           lab.grid(row=i,column=0)
           ent.grid(row=i,column=1)
           entries.append((fieldz[i][0], ent))
        return entries

    def fetch(self,fieldz):
#        print('%s: "%s"' % (fields[0][0],self.fp.get()))
        for entry in fieldz[0:]:
           field = entry[0]
           text  = entry[1].get()
           print('%s: "%s"' % (field, text))
        print("----------")
        return

    def arggen(self,fieldz):
        args=[]
        for entry in fieldz:
            field = entry[0]
            text  = entry[1].get()
            args.append(text)
        return args
    
    def save_tif(self,folder,subdir,imname,im,n,ext):
        suf='0000'
        fp=os.path.join(folder,subdir,imname.split('.')[0]+'_'+suf[:-len(str(n))]+str(n)+ext)
        image=Image.fromarray(im)
        image.save(fp)



smooth = 1.

def check_gpu():
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)
    with tf.Session() as sess:
        print (sess.run(c))


if __name__ == '__main__':
    initdir="/Volumes/Data/Luca_Work/MPI/Science/Coding/Python/Segm "
    mainfields=('Root path','Files extension')
    maindeftexts=('/','.tif')
    prepfields=('Replica per image','Crop Ratio')
    prepdeftxt=(3,0.8)
    trfields=('N epochs','Validation Split','N. of labels','Learning rate','Batch size')
    trdeftxt=(500,0.15,1,0.00001,16)
    
    fieldsprep=[]
    fieldstrain=[]
    if len(mainfields)==len(maindeftexts) and len(prepfields)==len(prepdeftxt):
        for i in range(len(mainfields)):
            tmp=(mainfields[i],maindeftexts[i])
            fieldsprep.append(tmp)
        for i in range(len(prepfields)):
            tmp=(prepfields[i],prepdeftxt[i])
            fieldsprep.append(tmp)
    if len(mainfields)==len(maindeftexts) and len(trfields)==len(trdeftxt): 
        for i in range(len(mainfields)):
            tmp=(mainfields[i],maindeftexts[i])
            fieldstrain.append(tmp)
        for i in range(len(trfields)):
            tmp=(trfields[i],trdeftxt[i])
            fieldstrain.append(tmp)
    root=tk.Tk()
    app=App(root)
    app.mainloop()

#    create_train_data(data_path,train_p,label_p,fext)
#    augment_DS(load_train_data(data_path,train_npy,labels_npy)[0],load_train_data(data_path,train_npy,labels_npy)[1],data_path,5)
#    create_train_data(data_path,train_p,label_p,fext)
#    create_test_data(data_path,test_p,fext)
#    train(data_path,train_npy,labels_npy)
#    predict(data_path,test_npy,test_id_npy)

#    tmp=load_train_data(data_path,train_npy,labels_npy)
#    tmp2=load_test_data(data_path,test_npy,test_id_npy)
