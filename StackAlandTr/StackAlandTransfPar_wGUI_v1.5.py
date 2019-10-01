#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 18:24:19 2017

@author: Luca
"""
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
from tkinter import messagebox
import os
import sys
from multiprocessing import Process,Queue,Pool
from collections import Counter
import numpy as np
import copy
import os
import cv2
from PIL import Image
import skimage.transform 
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import time
import platform
import math
import traceback
#import imreg_dft

mfw,mfh=600,350
nbfw,nbfh=550,40

def count_files_with_ext(ext,folder):
    fnum=Counter(ext in fname for fname in os.listdir(folder))[1]
    return fnum

def save_tif(folder,subdir,imname,im,ext):
    fp=os.path.join(folder,subdir,imname.split('.')[0]+'_Al'+ext)
    image=Image.fromarray(im)
    image.save(fp)

def res_canv(img,newshape,shx,shy): 
    newImage = np.zeros(newshape,np.uint8)
    yst=int(round(shy))
    xst=int(round(shx))
    newImage[yst:img.shape[0]+yst,xst:img.shape[1]+xst]=img
    return(newImage)

#def transform(im,TrM,nsz,shx,shy,stfolder,sfn,fnames,i,ext):
#    #the first image is not translated but only canvas resized
#    if i==0:
#        TrMatNull=np.array([[ 1.,0.,0.],[ 0.,1.,0.]])
#        tmp = cv2.warpAffine(res_canv(im,nsz,shx,shy), TrMatNull, (nsz[1],nsz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#        #rescaling the intensity of the images to 8-bit gray 
#        save_tif(stfolder,sfn,fnames[i],tmp,ext)
#    else:
#        #appliyng the transformation to the images after the first
#        tmp = cv2.warpAffine(res_canv(im,nsz,shx,shy), TrM[i-1], (nsz[1],nsz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#        save_tif(stfolder,sfn,fnames[i],tmp,ext)
#    return

def transform(im,TrM,fname,nsz,shx,shy,stfolder,sfn,ext):
    if not sfn in sorted(os.listdir(stfolder)):
        os.makedirs(os.path.join(stfolder,sfn))
        print('directory created')
    tmp = cv2.warpAffine(res_canv(im,nsz,shx,shy), TrM, (nsz[1],nsz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    save_tif(stfolder,sfn,fname,tmp,ext)
    return

def align(im1,im2,wma,wmo,cr,red,i):
    sz = im1.shape
#        (cc, warp_matrix) = cv2.findTransformECC (im2[int(sz[0]/2)-int(sz[0]/3):int(sz[0]/2)+int(sz[0]/3),int(sz[1]/2)-int(sz[1]/4):int(sz[1]/2)+int(sz[1]/4)],im1[int(sz[0]/2)-int(sz[0]/3):int(sz[0]/2)+int(sz[0]/3),int(sz[1]/2)-int(sz[1]/4):int(sz[1]/2)+int(sz[1]/4)],wma, wmo, cr)
#        (cc, warp_matrix) = cv2.findTransformECC (im2[:,int(sz[1]/2)-int(sz[1]/4):int(sz[1]/2)+int(sz[1]/4)],im1[:,int(sz[1]/2)-int(sz[1]/4):int(sz[1]/2)+int(sz[1]/4)],wma, wmo, cr)
    im1r,im2r = img_as_ubyte(skimage.transform.rescale(im1, 1/red)),img_as_ubyte(skimage.transform.rescale(im2, 1/red))
    (cc, warp_matrix) = cv2.findTransformECC(im2r,im1r,wma, wmo, cr)
#    print(i)
#    print(cc)
        #if findTransformECC fails to find the transform, use match template to help
    if cc<0.85:
        print('cannot align images',i+1,'and',i+2,':correlation too low --',cc)
        print('trying with match template')
        cormap=cv2.matchTemplate(im2r, im1r[int(2*sz[0]/5/red):int(3*sz[0]/5/red),int(2*sz[1]/5/red):int(3*sz[1]/5/red)], cv2.TM_CCOEFF_NORMED)
        #finds the pixel where the correlation map is max
        cormax=np.unravel_index(cormap.argmax(),cormap.shape)
        xsh=cormap.shape[1]/2-cormax[1]
        ysh=cormap.shape[0]/2-cormax[0]
        TrMatComp=np.array([[ 1.,0.,xsh],[ 0.,1.,ysh]])
        im1rtmp=cv2.warpAffine(np.copy(im1r), TrMatComp, (im1r.shape[1],im1r.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        (cc2, warp_matrix2) = cv2.findTransformECC(im2r,im1rtmp,wma, wmo, cr)
        print('calculated approximate shift: x',xsh,' and  y',ysh,' pixels')
        print('new correlation:',cc2,'with a warp matrix of')
        print(warp_matrix2)
        warp_matrix[0][2]=warp_matrix2[0][2]+xsh
        warp_matrix[1][2]=warp_matrix2[1][2]+ysh
        cc=cc2
    warp_matrix[0][2]=(red*warp_matrix[0][2])
    warp_matrix[1][2]=(red*warp_matrix[1][2])  
#        print(warp_matrix,' ',i)
    return np.copy(warp_matrix),np.copy(cc)

def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))

def outliers_mad(points, thresh=1.2):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return np.where(modified_z_score > thresh)

def init_feature(name,matcher,fik,nt,cr):
    if name == 'sift':
        detector = cv2.xfeatures2d.SIFT_create(nOctaveLayers=1, edgeThreshold=7, contrastThreshold=0.04,sigma=1.8)
#        detector = cv2.xfeatures2d.SIFT_create()
        norm = cv2.NORM_L2
    elif name == 'surf':
        detector = cv2.xfeatures2d.SURF_create(nOctaves=1, nOctaveLayers=1,extended=True, upright=True)
        norm = cv2.NORM_L2
    elif name == 'orb':
        detector = cv2.ORB_create()
        norm = cv2.NORM_HAMMING
    elif name == 'akaze':
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif name == 'brisk':
        detector = cv2.BRISK_create(octaves=0,thresh=35)
        norm = cv2.NORM_HAMMING
    else:
        return None, None
    if 'flann' in matcher:
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm = fik, trees = nt)
        else:
            flann_params= dict(algorithm = 6,
                               table_number = 24, # 6 - 12
                               key_size = 15,     # 12 - 20
                               multi_probe_level = 2) # 1 - 2
        search_params = dict(checks = cr)
        matcher = cv2.FlannBasedMatcher(flann_params, search_params) # bug : need to pass empty dict (#1329)
            
    else:
        matcher = cv2.BFMatcher(norm)
    return detector, matcher

def alignSIFT(im0,im1,fik,ntree,cr,red,dsc,mtch,imn):
    warp_matrix = np.array([[ 1.,0.,0.],[ 0.,1.,0.]])
    im0r,im1r = img_as_ubyte(skimage.transform.rescale(im0, 1/red)),img_as_ubyte(skimage.transform.rescale(im1, 1/red))
#    print(i)
    det=dsc
    matc=mtch
    detector,matcher=init_feature(det,matc,fik,ntree,cr)
    kp0,des0 = detector.detectAndCompute(im0r,None)
    kp1,des1 = detector.detectAndCompute(im1r,None)
    matches = matcher.knnMatch(des0,des1,k=2)
    good=[]
    # find the good matching descriptors
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append(m)    
    # points coordinates for good matches
    src_pts = np.float32([ kp0[m.queryIdx].pt for m in good ])
    dst_pts = np.float32([ kp1[m.trainIdx].pt for m in good ])
    # filter out the outliers
    distarr=np.array([math.hypot(i[0],i[1]) for i in src_pts-dst_pts])
    outl_arr=np.array(outliers_mad(distarr))
    src_good,dst_good=np.array([x for i,x in enumerate(src_pts) if i not in outl_arr]),np.array([x for i,x in enumerate(dst_pts) if i not in outl_arr])
    distarr_good=[math.hypot(i[0],i[1]) for i in src_good-dst_good]
    # calculate the mean shift
    print(len(src_good),"matching features in slice",str(imn)+". Number of outliers:",len(outl_arr[0]))
#    if(len(src_good)==0):
#        messagebox.showerror('Error','Could not find features')
#        root.destroy()
    # calculate the average shift
    xsh,ysh=np.average(src_good[:,0]-dst_good[:,0]),np.average(src_good[:,1]-dst_good[:,1])
    # define the warp matrix
    warp_matrix[0][2]=(red*xsh)
    warp_matrix[1][2]=(red*ysh)#+1
    # estimation of the accuracy of the shift
    stde=np.std(distarr_good)/math.sqrt(len(src_good))
    return np.copy(warp_matrix),np.copy(stde)

def alignDFT(im0,im1,red):
    im0r,im1r = np.array(img_as_ubyte(skimage.transform.rescale(im0, 1/red)),dtype='float32'),np.array(img_as_ubyte(skimage.transform.rescale(im1, 1/red)),dtype='float32')
    results=cv2.phaseCorrelate(im0r,im1r)
    warp_matrix = np.array([[ 1.,0.,red*(-float(results[0][0]))],[ 0.,1.,red*(-float(results[0][1]))]])
    stde=float(results[1])
    return np.copy(warp_matrix),np.copy(stde)

def alignDFT2(im0,im1,red):
    im0r,im1r = img_as_ubyte(skimage.transform.rescale(im0, 1/red)),img_as_ubyte(skimage.transform.rescale(im1, 1/red))
    results=imreg_dft.imreg.translation(im0r,im1r)
    warp_matrix = np.array([[ 1.,0.,red*(float(results['tvec'][1]))],[ 0.,1.,red*(float(results['tvec'][0]))]])
    stde=float(results['success'])
    return np.copy(warp_matrix),np.copy(stde)
    

def worker(imgs,i,chunksize, out_q,func,*args):
       """ The worker function, invoked in a process. 'images' is a
           list of images to span the process upon. The results are placed in
           a dictionary that's pushed to a queue.
       """
       outdict = {}
       for imn in range(len(imgs)-1):
#           print(i*chunksize+imn)
           outdict[i*chunksize+imn] = func(imgs[imn],imgs[imn+1],*args[1:],i*chunksize+imn)
       out_q.put(outdict)

def workerDFT(imgs,i,chunksize, out_q,func,*args):
       """ The worker function, invoked in a process. 'images' is a
           list of images to span the process upon. The results are placed in
           a dictionary that's pushed to a queue.
       """
       outdict = {}
       for imn in range(len(imgs)-1):
#           print(i*chunksize+imn)
           outdict[i*chunksize+imn] = func(imgs[imn],imgs[imn+1],*args[1:])
       out_q.put(outdict)

def workerTR(imgs,TrM,fnames,i,chunksize, out_q,func,*args):
       """ The worker function, invoked in a process. 'images' is a
           list of images to span the process upon. The results are placed in
           a dictionary that's pushed to a queue.
       """
       outdict = {}
       for imn in range(len(imgs)):
#           print(i*chunksize+imn)
           outdict[i*chunksize+imn] = func(imgs[imn],TrM[imn],fnames[imn],*args)
       out_q.put(outdict)

class App(tk.Frame):
    def __init__(self,master):
        tk.Frame.__init__(self,master)
        self.master.title("Stack Registration App")
        self.master.resizable(False,False)
        self.master.tk_setPalette(background='#ececec')
        x=int((self.master.winfo_screenwidth()-self.master.winfo_reqwidth())/1.5)
        y=int((self.master.winfo_screenheight()-self.master.winfo_reqheight())/1.5)
        self.master.geometry('{}x{}'.format(x,y))
        
        self.n=ttk.Notebook(master)
        self.al_frame=tk.Frame(self.n, width = mfw, height=mfh)
        self.tr_frame=tk.Frame(self.n, width = mfw, height=mfh)
        
        self.n.add(self.al_frame, text='Stack registration')
        self.n.add(self.tr_frame, text='Stack Transformation')        
        self.n.pack()
        
        # Create the main containers for the alignment notebook
        tk.Label(self.al_frame,text="App to register image stacks").grid(row=0)
        self.top_frame=tk.Frame(self.al_frame, width = nbfw, height=nbfh, pady=3)
        self.cen_frame=tk.Frame(self.al_frame, width = nbfw, height=nbfh*len(params[1:]), pady=3)
        self.model_frame=tk.Frame(self.al_frame, width = nbfw, height=nbfh, pady=3)
        self.btm_frame=tk.Frame(self.al_frame, width = nbfw, height=nbfh, pady=3)
        
        # Layout the containers
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.top_frame.grid(row=0, sticky="ew")
        self.cen_frame.grid(row=1, sticky="ew")
        self.model_frame.grid(row=2, sticky="ew")
        self.btm_frame.grid(row=3, sticky="ew")
        
        self.fpath=tk.Label(self.top_frame,text=fields[0][0])
        self.fp=tk.StringVar(self.top_frame)
        self.fpath_val=tk.Entry(self.top_frame,textvariable=self.fp)
        self.browse_button = tk.Button(self.top_frame,text="Browse", fg="green",command=self.browseAlSt)
        
        # layout the widgets in the top frame
        self.fpath.grid(row=0)
        self.fpath_val.grid(row=0,column=1)
        self.browse_button.grid(row=0,column=2)
        
        self.ents=self.makeFrame(self.cen_frame)
        
        self.descrlab=tk.Label(self.model_frame,text='Descriptor:')
        self.descstr=tk.StringVar(self.model_frame)
        self.descstr.set('sift')
        self.descr = tk.OptionMenu(self.model_frame, self.descstr, 'ECC', 'sift', 'surf', 'orb', 'akaze','brisk','PhCorr_DFT')
        self.matcherlab=tk.Label(self.model_frame,text='       Matcher:')
        self.matcherstr=tk.StringVar(self.model_frame)
        self.matcherstr.set('flann')
        self.matcher = tk.OptionMenu(self.model_frame, self.matcherstr, 'flann', 'brute force')
        self.descr.grid(row=0,column=2)
        self.descrlab.grid(row=0,column=1)
        self.matcher.grid(row=0,column=4)
        self.matcherlab.grid(row=0,column=3)
        
        
        # Widgets of the bottom frame
#        self.run_button = tk.Button(self.btm_frame,text="Run", fg="Red",command=self.alignfuncSIFT)
        self.run_button = tk.Button(self.btm_frame,text="Run", fg="Red",command=self.alignfunc)
        self.quit_button = tk.Button(self.btm_frame, text='Quit', command=self.cancel_b)

        # layout the widgets in the bottom frame
        self.run_button.grid(row=0,column=0)
        self.quit_button.grid(row=0,column=2)
        
        # Create the main containers for the transformation notebook
        tk.Label(self.tr_frame,text="App to transform image stacks from a transf. matrix").grid(row=0)
        self.top_tr_frame=tk.Frame(self.tr_frame, width = nbfw, height=nbfh, pady=3)
        self.cen_tr_frame=tk.Frame(self.tr_frame, width = nbfw, height=nbfh*len(params[1:]), pady=3)
        self.btm_tr_frame=tk.Frame(self.tr_frame, width = nbfw, height=nbfh, pady=3)
        
#        # Layout the containers
#        self.grid_rowconfigure(0, weight=1)
#        self.grid_columnconfigure(0, weight=1)

        self.top_tr_frame.grid(row=0, sticky="ew")
        self.cen_tr_frame.grid(row=1, sticky="ew")
        self.btm_tr_frame.grid(row=2, sticky="ew")
        
        self.fpath_tr=tk.Label(self.top_tr_frame,text=fields[0][0])
        self.fp_tr=tk.StringVar(self.top_tr_frame)
        self.fpath_tr_val=tk.Entry(self.top_tr_frame,textvariable=self.fp_tr)
        self.browse_tr_button = tk.Button(self.top_tr_frame,text="Browse", fg="green",command=self.browseTrSt)
        self.trpath_tr=tk.Label(self.top_tr_frame,text='Path to Transforms.txt')
        self.trp_tr=tk.StringVar(self.top_tr_frame)
        self.trpath_tr_val=tk.Entry(self.top_tr_frame,textvariable=self.trp_tr)
        self.browse2_tr_button = tk.Button(self.top_tr_frame,text="Browse", fg="green",command=self.browseTrMat)
        
        # layout the widgets in the top frame
        self.fpath_tr.grid(row=0)
        self.fpath_tr_val.grid(row=0,column=1)
        self.browse_tr_button.grid(row=0,column=2)
        self.trpath_tr.grid(row=1)
        self.trpath_tr_val.grid(row=1,column=1)
        self.browse2_tr_button.grid(row=1,column=2)
        
        self.ents_tr=self.make_tr_Frame(self.cen_tr_frame)
        
        # Widgets of the bottom frame
        self.run_tr_button = tk.Button(self.btm_tr_frame,text="Run", fg="Red",command=self.transffunct)
        self.quit_tr_button = tk.Button(self.btm_tr_frame, text='Quit', command=self.cancel_b)

        # layout the widgets in the bottom frame
        self.run_tr_button.grid(row=0,column=0)
        self.quit_tr_button.grid(row=0,column=2)

        root.bind('<Return>', (lambda event: self.fetch_tr())) 
        root.bind('<Tab>', (lambda event: self.fetch())) 

    def cancel_b(self):
        self.quit()
        self.master.destroy()
        

    def browseAlSt(self):
        idir='/'
        if 'Win' in platform.system():
            idir = 'W:/'
        if 'Darwin' in platform.system():
            idir = "/Volumes/Data/Luca_Work/MPI/Science/Coding/Python"
        dirname = tk.filedialog.askdirectory(initialdir = idir ,title = 'Select path to image stack to be aligned')
        if dirname:
            self.fp.set(dirname)
        return

    def browseTrSt(self):
        idir='/'
        if 'Win' in platform.system():
            idir = 'W:/'
        if 'Darwin' in platform.system():
            idir = "/Volumes/Data/Luca_Work/MPI/Science/Coding/Python"
        dirname = tk.filedialog.askdirectory(initialdir = idir ,title = 'Select path to image stack to be aligned')
        if dirname:
            self.fp_tr.set(dirname)
        return

    def browseTrMat(self):
        dirname = tk.filedialog.askdirectory(initialdir = 'W:/',title = 'Select path to Transforms.txt')
        if dirname:
            self.trp_tr.set(dirname)
            
    def alignfunc(self):
        if self.descstr.get()=='ECC':
            self.alignfuncECC()
        else:
            self.alignfuncSIFT()
            
    def alignfuncSIFT(self):
        # Read the images to be aligned
        args=self.arggen()
        stfolder,nC,ext,sfn,red,descriptor,matcher=args[0].replace("\\","/"),int(args[1]),args[2],args[3],int(args[4]),args[5],args[6] 
        noffiles=count_files_with_ext(ext,stfolder)
        alsteps=3
        self.loading=ProgWin(self.master,alsteps,noffiles)
        start = time.time()
        print("Image loading started on", time.asctime())
        self.loading.prg_status['text']="Image loading started on "+ str(time.asctime())
        self.update()
        stack,fnames=self.load_images_from_folder(stfolder,ext)
        stack=np.array(stack)
        print('Stack Size is:',sys.getsizeof(stack)//((1024**2)*10.24)/100,'Gb')
        end = time.time()
        secs = end-start
        print("Image loading took", secs)
        self.loading.prg_status['text']="Image loading took "+ str(secs)
        self.loading.progress['value']+=1
        self.update()
        nimages=len(stack)
        print("N. of Images: ",nimages)
        self.loading.prg_status['text']="N. of images in the stack "+str(nimages)
        self.update()
        
        sz = stack[0].shape 
        # Define the motion model
        FLANN_INDEX_KDTREE = 1
        nTREES = 5
        checks = 60
        
        # Run the ECC algorithm. The results are stored in TrMat.
        start = time.time()
        print("Alignment started on", time.asctime())
        self.loading.prg_status['text']="Alignment started on "+str(time.asctime())
        self.loading.progress2['value']=0
        self.update()
        global TrMat
        global ccmat
        TrMat = []
        ccmat=[]
#        tmp=self.multi_alignSIFT_wm(stack,FLANN_INDEX_KDTREE,nTREES,checks,nC,red,descriptor,matcher)
#        tmp=self.mp_pooler(nC,alignSIFT,stack,FLANN_INDEX_KDTREE,nTREES,checks,red,descriptor,matcher)
        if descriptor=='PhCorr_DFT':
            tmp=self.mp_processDFT(nC,alignDFT,stack,red)
        else:
            tmp=self.mp_process(nC,alignSIFT,stack,FLANN_INDEX_KDTREE,nTREES,checks,red,descriptor,matcher)
#        for i in range(len(tmp)):
#            TrMat.append(tmp[i].get()[0])
#            ccmat.append(tmp[i].get()[1])
        for i in range(len(tmp)):
            TrMat.append(tmp[i][0].astype(int))
            ccmat.append(tmp[i][1])
        TrMat=np.array(TrMat)
        ccmat=np.array(ccmat)
        
        print('successfully computed transformation matrices')
        self.loading.progress['value']+=1
        self.update()
        #align(stack,warp_matrix, warp_mode, criteria)
        
        #Compute the incremental transformation matrices
        print('computing incremental transformation matrices')
        self.loading.prg_status['text']="computing incremental transformation matrices"
        self.update()
        global TrMatInc
        TrMatInct=np.copy(TrMat)
        for j in range(len(TrMat)):
            TrMatInct[j,0,2]=-TrMat[0:j+1,0,2].sum()
            TrMatInct[j,1,2]=-TrMat[0:j+1,1,2].sum()
        TrMatNull=np.array([[[ 1.,0.,0.],[ 0.,1.,0.]]])
        TrMatInc=np.vstack((TrMatNull,TrMatInct))
#        print(len(stack),len(TrMatInc),len(fnames))
#        print(fnames)
        print('successfully computed incremental transformation matrices')
        end = time.time()
        secs = end-start
        print("Alignment took", secs)
        self.loading.prg_status['text']="Alignment took "+ str(secs)
        self.loading.progress['value']+=1
        self.update()
        
        #Computing maximum shifts for canvas resize
        # Maximum shift (neg x)
        minshx=TrMatInc[:,0,2].min()
        if minshx > 0:
            minshx = 0.
        # Maximum shift (pos x)
        maxshx=TrMatInc[:,0,2].max()
        if maxshx < 0:
            maxshx = 0.
        # Maximum shift (neg y)
        minshy=TrMatInc[:,1,2].min()
        if minshy > 0:
            minshy = 0.
        # Maximum shift (pos y)
        maxshy=TrMatInc[:,1,2].max()
        if maxshy < 0:
            maxshy = 0.    
        
        #new canvas size
        nsz=int(sz[0]+round(-minshy)+round(+maxshy)),int(sz[1]+round(-minshx)+round(maxshx))
        #Creating the new, shifted and resized, images
        self.loading.progress2['value']=0
        self.update()
        start=time.time()
        self.mp_processTR(nC,transform,stack,TrMatInc,fnames,nsz,maxshx,maxshy,stfolder,sfn,ext)
        transforms=[]
        for i in range(len(TrMatInc)):
            transforms.append([TrMatInc[i,0,2],TrMatInc[i,1,2]])
        np.savetxt(stfolder+'/'+sfn+'/Transforms.txt',transforms)
        end = time.time()
        secs = end-start
        print("The transformation took ", secs)
        self.loading.prg_status['text']="The transformation took "+ str(secs)
        self.update()
        print('Done')
        self.loading.destroy()
        return
    
    def alignfuncECC(self):
        # Read the images to be aligned
        args=self.arggen()
        stfolder,nC,ext,sfn,red,descriptor,matcher=args[0].replace("\\","/"),int(args[1]),args[2],args[3],int(args[4]),args[5],args[6] 
        noffiles=count_files_with_ext(ext,stfolder)
        alsteps=3
        self.loading=ProgWin(self.master,alsteps,noffiles)
        start = time.time()
        print("Image loading started on", time.asctime())
        self.loading.prg_status['text']="Image loading started on "+ str(time.asctime())
        self.update()
        stack,fnames=self.load_images_from_folder(stfolder,ext)
        stack=np.array(stack)
        end = time.time()
        secs = end-start
        print("Image loading took", secs)
        self.loading.prg_status['text']="Image loading took "+ str(secs)
        self.loading.progress['value']+=1
        self.update()
        nimages=len(stack)
        print("N. of Images: ",nimages)
        self.loading.prg_status['text']="N. of images in the stack "+str(nimages)
        self.update()
        
        sz = stack[0].shape 
        # Define the motion model
        warp_mode = cv2.MOTION_TRANSLATION
         
        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
         
        # Specify the number of iterations.
        number_of_iterations = 50000;
         
        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        termination_eps = 1e-10;
         
        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
         
        # Run the ECC algorithm. The results are stored in TrMat.
        start = time.time()
        print("Alignment started on", time.asctime())
        self.loading.prg_status['text']="Alignment started on "+str(time.asctime())
        self.loading.progress2['value']=0
        self.update()
        global TrMat
        global ccmat
        TrMat = []
        ccmat=[]
#        tmp=self.multi_align_wm(stack,warp_matrix, warp_mode, criteria,nC,red)
#        tmp=self.mp_pooler(nC,align,stack,warp_matrix, warp_mode, criteria,red)
        tmp=self.mp_process(nC,align,stack,warp_matrix, warp_mode, criteria,red)
#        for i in range(len(tmp)): 
#            TrMat.append(tmp[i].get()[0])
#            ccmat.append(tmp[i].get()[1])
        for i in range(len(tmp)): 
            TrMat.append(tmp[i][0].astype(int))
            ccmat.append(tmp[i][1])
        TrMat=np.array(TrMat)
        ccmat=np.array(ccmat)
        print('successfully computed transformation matrices')
        self.loading.progress['value']+=1
        self.update()
        #align(stack,warp_matrix, warp_mode, criteria)
        
        #Compute the incremental transformation matrices
        print('computing incremental transformation matrices')
        self.loading.prg_status['text']="computing incremental transformation matrices"
        self.update()
        global TrMatInc
        TrMatInc=np.copy(TrMat)
        for j in range(len(TrMat)):
            TrMatInct[j,0,2]=-TrMat[0:j+1,0,2].sum()
            TrMatInct[j,1,2]=-TrMat[0:j+1,1,2].sum()
        TrMatNull=np.array([[[ 1.,0.,0.],[ 0.,1.,0.]]])
        TrMatInc=np.vstack((TrMatNull,TrMatInct))
        print('successfully computed incremental transformation matrices')
        end = time.time()
        secs = end-start
        print("Alignment took", secs)
        self.loading.prg_status['text']="Alignment took "+ str(secs)
        self.loading.progress['value']+=1
        self.update()
        
        #Computing maximum shifts for canvas resize
        # Maximum shift (neg x)
        minshx=TrMatInc[:,0,2].min()
        if minshx > 0:
            minshx = 0.
        # Maximum shift (pos x)
        maxshx=TrMatInc[:,0,2].max()
        if maxshx < 0:
            maxshx = 0.
        # Maximum shift (neg y)
        minshy=TrMatInc[:,1,2].min()
        if minshy > 0:
            minshy = 0.
        # Maximum shift (pos y)
        maxshy=TrMatInc[:,1,2].max()
        if maxshy < 0:
            maxshy = 0.    
        
        #new canvas size
        nsz=int(sz[0]+round(-minshy)+round(+maxshy)),int(sz[1]+round(-minshx)+round(maxshx))
        #Creating the new, shifted and resized, images
        self.loading.progress2['value']=0
        self.update()
        start=time.time()
        
        self.mp_processTR(nC,transform,stack,TrMatInc,fnames,nsz,maxshx,maxshy,stfolder,sfn,ext)
        transforms=[]
        for i in range(len(TrMatInc)):
            transforms.append([TrMatInc[i,0,2],TrMatInc[i,1,2]])
        np.savetxt(stfolder+'/'+sfn+'/Transforms.txt',transforms)
        end = time.time()
        secs = end-start
        print("The transformation took ", secs)
        self.loading.prg_status['text']="The transformation took "+ str(secs)
        self.update()
        print('Done')
        self.loading.destroy()
        return

    def transffunct(self):
        # Read the images to be aligned
        args=self.arggen_tr()
        print(args)
        stfolder,trfolder,nC,ext,sfn=args[0].replace("\\","/"),args[1],int(args[2]),args[3],args[4] 
        noffiles=count_files_with_ext(ext,stfolder)
        alsteps=3
        self.loading=ProgWin(self.master,alsteps,noffiles)
        start = time.time()
        print("Image loading started on", time.asctime())
        self.loading.prg_status['text']="Image loading started on "+ str(time.asctime())
        self.update()
        stack,fnames=self.load_images_from_folder(stfolder,ext)
        stack=np.array(stack)
        sz = stack[0].shape
        end = time.time()
        secs = end-start
        print("Image loading took", secs)
        self.loading.prg_status['text']="Image loading took "+ str(secs)
        self.loading.progress['value']+=1
        self.update()
        nimages=len(stack)
        print("N. of Images: ",nimages)
        self.loading.prg_status['text']="N. of images in the stack "+str(nimages)
        self.update()
        print("Transforms loading started on", time.asctime())
        self.loading.prg_status['text']="Transform loading started on "+ str(time.asctime())
        self.update()
        if not sfn in sorted(os.listdir(stfolder)):
            os.makedirs(os.path.join(stfolder,sfn))
            print('directory created')
        global TrMatInc
        TrMatInc=[]
        trname='Transforms.txt'
        TrMatVals=np.loadtxt(os.path.join(trfolder,trname))
        for i in range(len(TrMatVals)):
            TrMattmp=[[ 1.,0.,0.],[ 0.,1.,0.]]
            TrMattmp[0][2]=TrMatVals[i,0]
            TrMattmp[1][2]=TrMatVals[i,1]
            TrMatInc.append(TrMattmp)
        TrMatInc=np.array(TrMatInc)        
        
        end = time.time()
        secs = end-start
        print("Loading of the Transformation matrices took", secs)
        self.loading.prg_status['text']="Image loading took "+ str(secs)
        self.loading.progress['value']+=1
        self.update()
        nimages=len(stack)
        print("N. of Images: ",nimages)
        self.loading.prg_status['text']="N. of images in the stack "+str(nimages)
        self.update()
        
        # Maximum shift (neg x)
        minshx=TrMatInc[:,0,2].min()
        if minshx > 0:
            minshx = 0.
        # Maximum shift (pos x)
        maxshx=TrMatInc[:,0,2].max()
        if maxshx < 0:
            maxshx = 0.
        # Maximum shift (neg y)
        minshy=TrMatInc[:,1,2].min()
        if minshy > 0:
            minshy = 0.
        # Maximum shift (pos y)
        maxshy=TrMatInc[:,1,2].max()
        if maxshy < 0:
            maxshy = 0.          

        nsz=int(sz[0]+round(-minshy)+round(+maxshy)),int(sz[1]+round(-minshx)+round(maxshx))        
        self.loading.progress2['value']=0
        self.update()
        start=time.time()
        self.mp_processTR(nC,transform,stack,TrMatInc,fnames,nsz,maxshx,maxshy,stfolder,sfn,ext)
        end = time.time()
        secs = end-start
        print("The transformation took ", secs)
        self.loading.prg_status['text']="The transformation took "+ str(secs)
        self.update()
        print('Done')
        self.loading.destroy()
        return        
    
    #Make the main Alignment Frame    
    def makeFrame(self,parent):
        entries=[]
        entries.insert(0,(fields[0][0],self.fpath_val))
        for i in range(len(fields[1:])):
           lab = tk.Label(parent, width=25, text=fields[i+1][0], anchor='w')
           ent_txt=tk.StringVar(parent,value=fields[i+1][1])
           ent = tk.Entry(parent,textvariable=ent_txt)
           ent.config(justify=tk.RIGHT)
           lab.grid(row=i,column=0)
           ent.grid(row=i,column=1)
           entries.append((fields[i+1][0], ent))
        return entries
    
    #Make the main Transformation Frame  
    def make_tr_Frame(self,parent):
        entriestr=[]
        entriestr.insert(0,(fields[0][0],self.fpath_tr_val))
        entriestr.insert(1,('Transformation matrix',self.trpath_tr_val))
        for i in range(len(fields[1:-1])):
           lab = tk.Label(parent, width=25, text=fields[i+1][0], anchor='w')
           ent_txt=tk.StringVar(parent,value=fields[i+1][1])
           enttr = tk.Entry(parent,textvariable=ent_txt)
           enttr.config(justify=tk.RIGHT)
           lab.grid(row=i,column=0)
           enttr.grid(row=i,column=1)
           entriestr.append((fields[i+1][0], enttr))
        return entriestr

    def fetch(self):
        for entry in self.ents:
           field = entry[0]
           text  = entry[1].get()
           print('%s: "%s"' % (field, text))
        print('%s "%s"' % (self.descrlab['text'],self.descstr.get()))
        print('%s "%s"' % (self.matcherlab['text'],self.matcherstr.get()))
        print("----------")

    def fetch_tr(self):
        for entry in self.ents_tr:
           field = entry[0]
           text  = entry[1].get()
           print('%s: "%s"' % (field, text))
        print("----------")

    def arggen(self):
        args=[]
        for entry in self.ents:
           text  = entry[1].get()
           args.append(text)
        args.append(self.descstr.get())
        args.append(self.matcherstr.get())
        return args

    def arggen_tr(self):
        args=[]
        for entry in self.ents_tr:
           text  = entry[1].get()
           args.append(text)
        return args
    
    def load_images_from_folder(self,stfolder,fext):
        images = []
        fnames=[]
        nbad=0
        for filename in sorted(os.listdir(stfolder)):
            if not fext in filename: 
                continue
            img = cv2.cvtColor(cv2.imread(os.path.join(stfolder,filename)),cv2.COLOR_BGR2GRAY)
            if img is not None:
                nzeros=np.count_nonzero(img>0)
                if nzeros/(img.shape[0]*img.shape[1])<0.2:
                    nbad+=1
                    print('This image appear to be too dark to be aligned:',filename)
                    if nbad==1:
                        decide=messagebox.askyesno('Warning','One or more images appear too dark to be aligned (check Python console for more info).'+ "\n" +'Do you want to try to align anyway?')
                        if decide==False:
                            root.destroy()
                images.append(img)
                fnames.append(filename)
                self.loading.progress2['value']+=1
                self.update()
        return images,fnames
    
    def mp_pooler(self,nCORES,func,*args):
        pool=Pool(nCORES)
        print('computing with',nCORES,'processes in parallel')
        results=[]
        for i in range(len(args[0])-1):
            results.append(pool.apply_async(func,(args[0][i],args[0][i+1],*args[1:],i,)))
            self.loading.progress2['value']+=1
            self.update()
        pool.close()
        pool.join()
        return results       

    def mp_process(self,nprocs,func,*args):
        images=args[0]
        out_q = Queue()
        chunksize = int(math.ceil((len(images)-1) / float(nprocs)))
        procs = []
        print("Chunks of size:",chunksize)
        for i in range(nprocs):
            if i == nprocs-1:
                p = Process(
                        target=worker,
                        args=(images[chunksize * i:len(images)],i,chunksize,out_q,func,*args))
                procs.append(p)
                p.start()
                self.loading.progress2['value']+=chunksize
                self.update()
            else:                
                p = Process(
                        target=worker,
                        args=(images[chunksize * i:chunksize * (i + 1)+1],i,chunksize,out_q,func,*args))
                procs.append(p)
                p.start()
                self.loading.progress2['value']+=chunksize
                self.update()

        # Collect all results into a single result dict. We know how many dicts
        # with results to expect.
        resultdict = {}
        for i in range(nprocs):
            resultdict.update(out_q.get())

        # Wait for all worker processes to finish
        for p in procs:
            p.join()
            
        results=[]
        for j in range(len(resultdict)):
            results.append(resultdict[j])

        return results

    def mp_processDFT(self,nprocs,func,*args):
        images=args[0]
        out_q = Queue()
        chunksize = int(math.ceil((len(images)-1) / float(nprocs)))
        procs = []
        print("Chunks of size:",chunksize)
        for i in range(nprocs):
            if i == nprocs-1:
                p = Process(
                        target=workerDFT,
                        args=(images[chunksize * i:len(images)],i,chunksize,out_q,func,*args))
                procs.append(p)
                p.start()
                self.loading.progress2['value']+=chunksize
                self.update()
            else:                
                p = Process(
                        target=workerDFT,
                        args=(images[chunksize * i:chunksize * (i + 1)+1],i,chunksize,out_q,func,*args))
                procs.append(p)
                p.start()
                self.loading.progress2['value']+=chunksize
                self.update()

        # Collect all results into a single result dict. We know how many dicts
        # with results to expect.

        resultdict = {}
        for i in range(nprocs):
            resultdict.update(out_q.get())

        # Wait for all worker processes to finish
        for p in procs:
            p.join()

            
        results=[]
        for j in range(len(resultdict)):
            results.append(resultdict[j])

        return results
    
    def mp_processTR(self,nprocs,func,*args):
        images=args[0]
        TrM=args[1]
        fnames=args[2]
        out_q = Queue()
        chunksize = int(math.ceil((len(images)-1) / float(nprocs)))
        procs = []
        print("Chunks of size:",chunksize)
        for i in range(nprocs):
            if i == nprocs-1:
                p = Process(
                        target=workerTR,
                        args=(images[chunksize * i:len(images)],TrM[chunksize * i:len(images)],fnames[chunksize * i:len(images)],i,chunksize,out_q,func,*args[3:]))
                procs.append(p)
                p.start()
                self.loading.progress2['value']+=chunksize
                self.update()
            else:                
                p = Process(
                        target=workerTR,
                        args=(images[chunksize * i:chunksize * (i + 1)],TrM[chunksize * i:chunksize * (i + 1)],fnames[chunksize * i:chunksize * (i + 1)],i,chunksize,out_q,func,*args[3:]))
                procs.append(p)
                p.start()
                self.loading.progress2['value']+=chunksize
                self.update()

        # Collect all results into a single result dict. We know how many dicts
        # with results to expect.
        resultdict = {}
        for i in range(nprocs):
            resultdict.update(out_q.get())
            
        # Wait for all worker processes to finish
        for p in procs:
            p.join()
            
        results=[]
        for j in range(len(resultdict)):
            results.append(resultdict[j])

        return results

    def multi_align_tr(self,imstack,TrM,nsz,shx,shy,stfolder,sfn,nCORES,fnames,ext):
        if not sfn in sorted(os.listdir(stfolder)):
            os.makedirs(os.path.join(stfolder,sfn))
            print('directory created')
        pool=Pool(nCORES)
        print('applying transformations with',nCORES,'processes in parallel ')
        results=[]
        for i in range(len(imstack)):
#            results.append(transform(imstack[i],TrM,nsz,shx,shy,stfolder,sfn,fnames,i,ext,))
            results.append(pool.apply_async(transform,(imstack[i],TrM,nsz,shx,shy,stfolder,sfn,fnames,i,ext,)))
            self.loading.progress2['value']+=1
            self.update()
        pool.close()
        pool.join()
        print('successfully transformed all the images in the stack')
        return results

class ProgWin(tk.Frame):
    def __init__(self,master,count,count2):
        tk.Frame.__init__(self,master,borderwidth=5,relief='groove')
        self.pack()
        
        self.deftxt=tk.Label(self,text="Alignment is in progress")
        self.prg_status=tk.Label(self,text="Loading Files")
        self.deftxt.pack()
        self.prg_status.pack()
        
        self.progress=ttk.Progressbar(self,orient='horizontal',length=250,mode='determinate')
        self.progress.pack()
        self.progress['value']=0
        self.progress['maximum']=count
        
        self.progress2=ttk.Progressbar(self,orient='horizontal',length=250,mode='determinate')
        self.progress2.pack()
        self.progress2['value']=0
        self.progress2['maximum']=count2
    
    
if __name__ == '__main__':
    params=('Stack path','Number of Cores (out of '+str(os.cpu_count())+')','Files extension','Subdir name','Img Reduction')
    deftexts=('/','3','.tif','Aligned','1')
    fields=[]
    if len(params)==len(deftexts):
        for i in range(len(params)):
            tmp=(params[i],deftexts[i])
            fields.append(tmp)
    root=tk.Tk()
    app=App(root)
    app.mainloop()