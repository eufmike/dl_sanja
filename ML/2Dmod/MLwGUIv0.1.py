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
# from scipy.misc import imrotate
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

from data import load_train_data, load_test_data, create_train_data, create_test_data, crop_no_black, preprocess, getSegmentationArr
import Nmodels

mfw,mfh=600,450
nbfw,nbfh=550,40
tr_p='train'
train_p=os.path.join(tr_p,'image')
label_p=os.path.join(tr_p,'label')
model_p=os.path.join(tr_p,'models')
train_npy='imgs_train.npy'
labels_npy='imgs_labels.npy'
test_p='test'
test_npy='imgs_test.npy'
test_id_npy='imgs_id_test.npy'
selfold='Augmented'

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
        self.augm_button = tk.Button(self.btm_frame_prep,text="Augment", fg="Red",command=partial(self.AugmentDS,self.entsprepr))
        self.quit_button = tk.Button(self.btm_frame_prep, text='Quit', command=self.cancel_b)
        self.save_button= tk.Button(self.btm_frame_prep, text='Create train .npys', command=partial(self.SaveNpy,self.entsprepr,'trlab'))
        self.savetst_button= tk.Button(self.btm_frame_prep, text='Create test .npys', command=partial(self.SaveNpy,self.entsprepr,'test'))
        
        self.augm_button.grid(row=0,column=0)
        self.save_button.grid(row=0,column=1)
        self.savetst_button.grid(row=0,column=2)
        self.quit_button.grid(row=0,column=3)

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
        self.trmodelopt = tk.OptionMenu(self.trmodel_frame, self.trmodelstr, 'MksNet', 'UNet', 'SegNet', 'VGGSegNet','FCN32Net')
        self.trmodellab.grid(row=0,column=0)
        self.trmodelopt.grid(row=0,column=1)
        
        self.entstrain=self.makeFrame(self.cen_frame_train,fieldstrain)
        
        # Widgets of the bottom frame
        self.train_button = tk.Button(self.btm_frame_train,text="Train", fg="Red",command=partial(self.TrainModel,self.entstrain))
        self.quit_button = tk.Button(self.btm_frame_train, text='Quit', command=self.cancel_b)
        self.predict_button = tk.Button(self.btm_frame_train,text="Predict", fg="Red",command=partial(self.Predict,self.entstrain))
        
        self.train_button.grid(row=0,column=0)
        self.quit_button.grid(row=0,column=1)
        self.predict_button.grid(row=0,column=2)

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
        dirname = tk.filedialog.askdirectory(initialdir = idir ,title = 'Select root path for images')
        if dirname:
            self.fp.set(dirname)
            
    def AugmentDS(self,Augentries):
        datagen = ImageDataGenerator(
#            featurewise_center=False,
#            featurewise_std_normalization=False,
#            samplewise_center=False,
#            samplewise_std_normalization=False,
            rescale=None,
            rotation_range=3,
            width_shift_range=0.08,
            height_shift_range=0.08,
            shear_range=0.07,
            zoom_range=0.07,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='constant',
            cval=0.)
        args=self.arggen(Augentries)
#        trimgs,labimgs,data_path,num
        data_path,fext,num,cr=args[0],args[1],int(args[2]),float(args[3])
        create_train_data(data_path,train_p,label_p,fext)
        imgs=load_train_data(data_path,train_npy,labels_npy)
        nLabels=np.max(imgs[1])
        trimgs,labimgs=imgs[0],imgs[1]
        imgshape=trimgs[0].shape
        print('-'*30)
        print('Augmenting train and labels dataset: ',num,'replica per image...')
        print('-'*30)
    #    seed = np.random.randint(10000)
        seed=np.random.randint(10000,size=2*len(trimgs)*num)
        tmpf='tmp'
        if tmpf in sorted(os.listdir(os.path.join(data_path,train_p))):
           shutil.rmtree(os.path.join(data_path,train_p,tmpf), ignore_errors=True)
           shutil.rmtree(os.path.join(data_path,label_p,tmpf), ignore_errors=True)
        os.makedirs(os.path.join(data_path,train_p,tmpf))
        os.makedirs(os.path.join(data_path,label_p,tmpf))
        global batchdata
        batchdata=[]
        j=0
        for x in trimgs:
            x[x==0]=1
            x = x.reshape((1,) + x.shape+(1,))
            # the .flow() command below generates batches of randomly transformed images
            # and saves the results to the `preview/` directory
            i = 0
            
            for batch in datagen.flow(x, batch_size=1,
                                      seed=seed[j]):
                self.save_tif(data_path,os.path.join(train_p,tmpf),'img',batch[0,:,:,0].astype('uint8'),seed[i+j*2*num],fext)
                i += 1
                if i >= 2*num:
                    break  # otherwise the generator would loop indefinitely
            j +=1
        j=0
        for y in labimgs:
            y = y.reshape((1,) + y.shape+(1,))
            i = 0
            for batch in datagen.flow(y, batch_size=1,
                                       seed=seed[j]):

                self.save_tif(data_path,os.path.join(label_p,tmpf),'img',batch[0,:,:,0].astype('uint8'),seed[i+j*2*num],fext)
                batchdata.append(batch[0,:,:,0])
                i += 1
                if i >= 2*num:
                    break  # otherwise the generator would loop indefinitely
            j +=1
        create_train_data(data_path,os.path.join(train_p,tmpf),os.path.join(label_p,tmpf),fext)
        tmpimgs=load_train_data(data_path,train_npy,labels_npy)
        tmptr=tmpimgs[0]
        tmplab=tmpimgs[1]
        print(imgshape,cr)
        lencrop=int(((imgshape[0]*cr)//16)*16),int(((imgshape[1]*cr)//16)*16)
        print(lencrop)
        delta=imgshape[0]-lencrop[0],imgshape[1]-lencrop[1]
        print(delta)
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
        np.save(os.path.join(data_path,'imgs_train.npy'), seltr)
        print('Augmented train data saved to:',os.path.join(data_path,'imgs_train.npy'))
        np.save(os.path.join(data_path,'imgs_labels.npy'), sellab)
        print('Augmented label data saved to:',os.path.join(data_path,'imgs_labels.npy'))
        if selfold in sorted(os.listdir(os.path.join(data_path,train_p))):
           shutil.rmtree(os.path.join(data_path,train_p,selfold), ignore_errors=True)
           shutil.rmtree(os.path.join(data_path,label_p,selfold), ignore_errors=True)        
        os.makedirs(os.path.join(data_path,train_p,selfold))
        os.makedirs(os.path.join(data_path,label_p,selfold))
        for i in range(len(seltr)):
            self.save_tif(data_path,os.path.join(train_p,selfold),'img',seltr[i],i,fext)
            self.save_tif(data_path,os.path.join(label_p,selfold),'img',sellab[i],i,fext)
#        create_train_data(data_path,train_p,label_p,fext)
        if tmpf in sorted(os.listdir(os.path.join(data_path,train_p))):
           shutil.rmtree(os.path.join(data_path,train_p,tmpf), ignore_errors=True)
           shutil.rmtree(os.path.join(data_path,label_p,tmpf), ignore_errors=True)        
        print('Done')
        return

    def SaveNpy(self,entries,sel):
        args=self.arggen(entries)
        data_path,fext,num,cr=args[0],args[1],int(args[2]),float(args[3])
        if sel =='trlab':
            if selfold in sorted(os.listdir(os.path.join(data_path,train_p))):
                print('Augmented data found. Saving augmented data instead of original ones')
                create_train_data(data_path,os.path.join(train_p,selfold),os.path.join(label_p,selfold),fext)
            else:
                create_train_data(data_path,train_p,label_p,fext)                
        if sel =='test':
            create_test_data(data_path,test_p,fext)
            
    def TrainModel(self,TMentries):
        args=self.arggen(TMentries)
        data_path,mName,epchs,vspl,nCl=args[0],args[2],int(args[3]),float(args[4]),int(args[5]) 
        if nCl>1:
            nLab= nCl +1
        else: 
            nLab=nCl
        print('-'*30)
        print('Loading and preprocessing train data...')
        print('-'*30)
        imgs_train, imgs_mask_train = load_train_data(data_path,train_npy,labels_npy)
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
            print('Warning: the number of classes does not matches with the intesities of the label images')
        if nLab>1:
            imgs_mask_train = getSegmentationArr(imgs_mask_train , nLab  )
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
            model= getattr(Nmodels,mName)(nLab,imgs_train[0].shape,latest)
        else:
            if not os.path.exists(os.path.join(data_path,model_p)):
                os.makedirs(os.path.join(data_path,model_p))
            model= getattr(Nmodels,mName)(nLab,imgs_train[0].shape,'')
        
    #    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True) 
        model_checkpoint = ModelCheckpoint(os.path.join(data_path,model_p,'weights.ep{epoch:02d}-il{loss:.3f}-vl{val_loss:.3f}.hdf5'), monitor='loss',verbose=1, save_best_only=True)
    
        print('-'*30)
        print('Fitting model...')
        print('-'*30)
    #    model.fit(imgs_train, imgs_mask_train, batch_size=34, nb_epoch=20, verbose=1, shuffle=True,
    #              validation_split=0.2,
    #              callbacks=[model_checkpoint])         
        model.fit(imgs_train, imgs_mask_train, batch_size=10, epochs=epchs, verbose=1,
                  validation_split=vspl, 
                  shuffle=True, 
                  callbacks=[model_checkpoint])
        return
    
    def Predict(self,TMentries):
        args=self.arggen(TMentries)
        data_path,fext,mName,epchs,vspl,nCl=args[0],args[1],args[2],int(args[3]),float(args[4]),int(args[5])   
        if nCl>1:
            nLab= nCl +1
        else: 
            nLab=nCl
        print('-'*30)
        print('Loading and preprocessing test data...')
        print('-'*30)
        imgs_test_tot, imgs_id_test_tot = load_test_data(data_path,test_npy,test_id_npy)
        img_rows, img_cols = imgs_test_tot[0].shape[0],imgs_test_tot[0].shape[1]
        print('-'*30)
        print('Loading saved weights...')
        print('-'*30)
        print(max(glob.glob(os.path.join(data_path,model_p,'*.hdf5')),key=os.path.getctime))
        latest=max(glob.glob(os.path.join(data_path,model_p,'*.hdf5')),key=os.path.getctime)
        model= getattr(Nmodels,mName)(nLab,imgs_test_tot[0].shape,latest)
        i=0
        for i in range(len(imgs_test_tot)-1):
            print('image',i,'of',len(imgs_test_tot))
            imgs_test = preprocess(imgs_test_tot[i:i+1],img_rows, img_cols)
            imgs_id_test=imgs_id_test_tot[i:i+1]
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
            global predout
            predout=np.copy(imgs_mask_test)
            np.save('imgs_mask_test.npy', imgs_mask_test)
        
            print('-' * 30)
            print('Saving predicted labels to files...')
            print('-' * 30)
            pred_dir = os.path.join(data_path,'preds_'+mName+'_'+latest.split('weights',1)[1])
            if not os.path.exists(pred_dir):
                os.mkdir(pred_dir)
            for image, image_id in zip(imgs_mask_test, imgs_id_test):
                if nLab>1:
                    image = image.reshape(( img_rows, img_cols , nLab ) )
                    image= image.argmax( axis=2 ).astype(np.uint8)
#                    image = (image * 255.).astype(np.uint8)
                else:
                    image = (image[:, :, 0] * 255.).astype(np.uint8)
                imsave(os.path.join(pred_dir, str(image_id) + '_pred.tif'), image)


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
    trfields=('N epochs','Validation Split','N. of labels')
    trdeftxt=(500,0.15,1)
    
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
