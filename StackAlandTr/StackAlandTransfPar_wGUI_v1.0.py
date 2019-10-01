#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 18:24:19 2017
v 1.0
@author: Luca
"""
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
from tkinter import messagebox
import os
from multiprocessing import Process
from multiprocessing import Pool
from collections import Counter
import numpy as np
import copy
import os
import cv2
import matplotlib.pyplot as plt
from scipy import misc
import time
import platform

mfw,mfh=600,350
nbfw,nbfh=550,40

def count_files_with_ext(ext,folder):
    fnum=Counter(ext in fname for fname in os.listdir(folder))[1]
    return fnum

def save_tif(folder,subdir,imname,im,ext):
    misc.imsave(os.path.join(folder,subdir,imname.split('.')[0]+'_Al'+ext),im)

def res_canv(img,newshape,shx,shy): 
    newImage = np.zeros(newshape,np.uint8)
    yst=int(round(shy))
    xst=int(round(shx))
    newImage[yst:img.shape[0]+yst,xst:img.shape[1]+xst]=img
    return(newImage)

def transform(im,TrM,nsz,shx,shy,stfolder,sfn,fnames,i,ext):
    #the first image is not translated but only canvas resized
    if i==0:
        TrMatNull=np.array([[ 1.,0.,0.],[ 0.,1.,0.]])
        tmp = cv2.warpAffine(res_canv(im,nsz,shx,shy), TrMatNull, (nsz[1],nsz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        #rescaling the intensity of the images to 8-bit gray 
        if np.max(tmp)>1:
            im2_aligned=misc.toimage(tmp,cmin=0,cmax=255)
        else:
            im2_aligned=misc.toimage(255*tmp,cmin=0,cmax=255)
        save_tif(stfolder,sfn,fnames[i],im2_aligned,ext)
    else:
        #appliyng the transformation to the images after the first
        tmp = cv2.warpAffine(res_canv(im,nsz,shx,shy), TrM[i-1], (nsz[1],nsz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        if np.max(tmp)>1:
            im2_aligned=misc.toimage(tmp,cmin=0,cmax=255)
        else:
            im2_aligned=misc.toimage(255*tmp,cmin=0,cmax=255)
        save_tif(stfolder,sfn,fnames[i],im2_aligned,ext)
    return

def align(im1,im2,wma,wmo,cr,red,i):
    sz = im1.shape
#        (cc, warp_matrix) = cv2.findTransformECC (im2[int(sz[0]/2)-int(sz[0]/3):int(sz[0]/2)+int(sz[0]/3),int(sz[1]/2)-int(sz[1]/4):int(sz[1]/2)+int(sz[1]/4)],im1[int(sz[0]/2)-int(sz[0]/3):int(sz[0]/2)+int(sz[0]/3),int(sz[1]/2)-int(sz[1]/4):int(sz[1]/2)+int(sz[1]/4)],wma, wmo, cr)
#        (cc, warp_matrix) = cv2.findTransformECC (im2[:,int(sz[1]/2)-int(sz[1]/4):int(sz[1]/2)+int(sz[1]/4)],im1[:,int(sz[1]/2)-int(sz[1]/4):int(sz[1]/2)+int(sz[1]/4)],wma, wmo, cr)
    im1r,im2r = misc.imresize(im1, 1/red),misc.imresize(im2, 1/red)
    (cc, warp_matrix) = cv2.findTransformECC(im2r,im1r,wma, wmo, cr)

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

class App(tk.Frame):
    def __init__(self,master):
        tk.Frame.__init__(self,master)
        self.master.title("Stack Registration App")
        self.master.resizable(False,False)
        self.master.tk_setPalette(background='#ececec')
        x=int((self.master.winfo_screenwidth()-self.master.winfo_reqwidth())/1.5)
        y=int((self.master.winfo_screenheight()-self.master.winfo_reqheight())/2)
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
        self.btm_frame=tk.Frame(self.al_frame, width = nbfw, height=nbfh, pady=3)
        
        # Layout the containers
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.top_frame.grid(row=0, sticky="ew")
        self.cen_frame.grid(row=1, sticky="ew")
        self.btm_frame.grid(row=2, sticky="ew")
        
        self.fpath=tk.Label(self.top_frame,text=fields[0][0])
        self.fp=tk.StringVar(self.top_frame)
        self.fpath_val=tk.Entry(self.top_frame,textvariable=self.fp)
        self.browse_button = tk.Button(self.top_frame,text="Browse", fg="green",command=self.browseAlSt)
        
        # layout the widgets in the top frame
        self.fpath.grid(row=0)
        self.fpath_val.grid(row=0,column=1)
        self.browse_button.grid(row=0,column=2)
        
        self.ents=self.makeFrame(self.cen_frame)
        
        # Widgets of the bottom frame
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
        # Read the images to be aligned
        args=self.arggen()
        stfolder,nC,ext,sfn,red=args[0].replace("\\","/"),int(args[1]),args[2],args[3],int(args[4]) 
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
        tmp=self.multi_align_wm(stack,warp_matrix, warp_mode, criteria,nC,red)
        for i in range(len(tmp)): 
            TrMat.append(tmp[i].get()[0])
            ccmat.append(tmp[i].get()[1])
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
            TrMatInc[j,0,2]=-TrMat[0:j+1,0,2].sum()
            TrMatInc[j,1,2]=-TrMat[0:j+1,1,2].sum()
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
        self.multi_align_tr(stack,TrMatInc,nsz,maxshx,maxshy,stfolder,sfn,nC,fnames,ext)
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
        self.multi_align_tr(stack,TrMatInc,nsz,maxshx,maxshy,stfolder,sfn,nC,fnames,ext)

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
           field = entry[0]
           text  = entry[1].get()
           args.append(text)
        return args

    def arggen_tr(self):
        args=[]
        for entry in self.ents_tr:
           field = entry[0]
           text  = entry[1].get()
           args.append(text)
        return args
    
    def load_images_from_folder(self,stfolder,fext):
        images = []
        fnames=[]
        nbad=0
        for filename in os.listdir(stfolder):
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

    def multi_align_wm(self,imstack,wma,wmo,cr,nCORES,red):
        pool=Pool(nCORES)
        print('computing tranlation matrix with',nCORES,'processes in parallel')
        results=[]
        for i in range(len(imstack)-1):
            results.append(pool.apply_async(align,(imstack[i],imstack[i+1],wma,wmo,cr,red,i,)))
            self.loading.progress2['value']+=1
            self.update()
        pool.close()
        pool.join()
        return results

    def multi_align_tr(self,imstack,TrM,nsz,shx,shy,stfolder,sfn,nCORES,fnames,ext):
        if not sfn in os.listdir(stfolder):
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
#        ws = self.master.winfo_screenwidth() # width of the screen
#        hs = self.master.winfo_screenheight()
#        x = (ws/2) - (550/2)
#        y = (hs/2) - (150/2)
#        self.geometry('%dx%d+%d+%d'% (550,150,x,y))
#        self.grid(row=0,column=0,sticky=tk.E)
        self.pack()
        
        self.deftxt=tk.Label(self,text="Alignment is in progress")
        self.prg_status=tk.Label(self,text="Loading Files")
        self.deftxt.pack()
        self.prg_status.pack()
#        self.deftxt.grid(row=0,padx=15,pady=10)
#        self.prg_status.grid(row=1,padx=15,pady=10)
        
        self.progress=ttk.Progressbar(self,orient='horizontal',length=250,mode='determinate')
        self.progress.pack()
#        self.progress.grid(row=2,padx=15,pady=10)
        self.progress['value']=0
        self.progress['maximum']=count
        
        self.progress2=ttk.Progressbar(self,orient='horizontal',length=250,mode='determinate')
        self.progress2.pack()
#        self.progress2.grid(row=3,padx=15,pady=10)
        self.progress2['value']=0
        self.progress2['maximum']=count2
    
    
if __name__ == '__main__':
    params=('Stack path','Number of Cores (out of '+str(os.cpu_count())+')','Files extension','Subdir name','Img Reduction')
    deftexts=('/',str(os.cpu_count()-2),'.tif','Aligned','1')
    fields=[]
    if len(params)==len(deftexts):
        for i in range(len(params)):
            tmp=(params[i],deftexts[i])
            fields.append(tmp)
    root=tk.Tk()
    app=App(root)
    app.mainloop()