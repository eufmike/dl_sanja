#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 22:46:00 2017

@author: Luca
"""
# I(y)=a*I(x)+b --> I(x)=(I(y)-b)/a -> BW
# I(x)=a*I(y)+b -> FW
#Solve[y0 == a x0 + b && y1 == a x1 + b, {a, b}] -> BW
#Solve[x0 == a y0 + b && x1 == a y1 + b, {a, b}] -> FW
#a -> (y0 - y1)/(x0 - x1), b -> -(x1 y0 - x0 y1)/(x0 - x1) -> BW
#a -> (x0 - x1)/(y0 - y1), b -> -(-x1 y0 + x0 y1)/(y0 - y1) -> FW

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
from multiprocessing import Process
from multiprocessing import Pool
from collections import Counter
import os
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
from scipy import misc,linspace, polyval, polyfit, sqrt, stats, randn
from functools import partial
import time
import random
from PIL import Image, ImageTk
from skimage import exposure
import platform

def count_files_with_ext(ext,folder):
    fnum=Counter(ext in fname for fname in os.listdir(folder))[1]
    return fnum

def save_tif(folder,subdir,imname,im,ext):
    misc.imsave(os.path.join(folder,subdir,imname.split('.')[0]+'_IC'+ext),im)

def save_tif2(folder,subdir,imname,im,ext):
    misc.imsave(os.path.join(folder,subdir,imname.split('.')[0]+'_CEn'+ext),im)
    
def denandred(img,red,gausskernel):
    imgred=cv2.resize(img,(int(np.rint(img.shape[1]/red)), int(np.rint(img.shape[0]/red))), interpolation = cv2.INTER_AREA)
    imgredblur=cv2.GaussianBlur(imgred,(gausskernel,gausskernel),0)
    return imgredblur
    
def abmatcalc(yv,xv,i,rev):
    indtmp=random.randint(0,len(xv)-1)
    if rev==0:
        if 20 < abs(xv[indtmp]-xv[i]) < 140 and 15<xv[i]<220:
            atmp=(yv[indtmp]-yv[i])/(xv[indtmp]-xv[i])
            btmp=(xv[indtmp]*yv[i]-xv[i]*yv[indtmp])/(xv[indtmp]-xv[i])
        else:
            atmp='None'
            btmp='None'
    if rev==1:
        if 20 < abs(yv[indtmp]-yv[i]) < 140 and 15<yv[i]<220:
            atmp=(xv[indtmp]-xv[i])/(yv[indtmp]-yv[i])
            btmp=(xv[i]*yv[indtmp]-xv[indtmp]*yv[i])/(yv[indtmp]-yv[i])
        else:
            atmp='None'
            btmp='None'
    return atmp,btmp

def clahefunct(img,ts,cl,stfolder,sfn,ext,fname,prev):
    clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=(ts,ts))
    tmp=clahe.apply(img)
    CEimg=misc.toimage(tmp,cmin=0,cmax=255)
    if prev=='N':
        save_tif2(stfolder,sfn,fname,CEimg,ext)
    return CEimg

def clahefunct2(img,ts,cl,nb,stfolder,sfn,ext,fname,prev):
    tmp = exposure.equalize_adapthist(img,kernel_size=ts,clip_limit=cl/100,nbins=nb)
    CEimg=misc.toimage(255*tmp,cmin=0,cmax=255)
    if prev=='N':
        save_tif2(stfolder,sfn,fname,CEimg,ext)
    return CEimg

class App(tk.Frame):
    wcounter=0
    prevs=dict()
    mainframes=dict()
    prevlab=dict()
    prevparams=dict()
    prev_c_orig=dict()
    imgp=dict()
    fig=dict()
    prev_c_CE=dict()
    cld=dict()
    figCE=dict()
    def __init__(self,master):
        tk.Frame.__init__(self,master)
        self.master.title("Stack Intensity correction App")
        self.master.resizable(False,False)
        self.master.tk_setPalette(background='#ececec')
        x=int((self.master.winfo_screenwidth()-self.master.winfo_reqwidth())/1.5)
        y=int((self.master.winfo_screenheight()-self.master.winfo_reqheight())/1.5)
        self.master.geometry('{}x{}'.format(x,y))
        
        self.n=ttk.Notebook(master)
        self.IC_frame=tk.Frame(self.n, width = mfw, height=mfh)
        self.CLAHE_frame=tk.Frame(self.n, width = mfw, height=mfh)
        
        self.n.add(self.IC_frame, text='Intensity Correction')
        self.n.add(self.CLAHE_frame, text='Contrast enhancement') 
        self.n.pack()
        
        # Create the main containers for the Intensity correction notebook
        tk.Label(self.IC_frame,text="App to correct for B&C changes in image stacks").grid(row=0)
        self.top_frame=tk.Frame(self.IC_frame, width = nbfw, height=nbfh, pady=3)
        self.cen_frame=tk.Frame(self.IC_frame, width = nbfw, height=nbfh*len(params[1:]), pady=3)
        self.btm_frame=tk.Frame(self.IC_frame, width = nbfw, height=nbfh, pady=3)
        
        # Create the main containers for the Intensity correction notebook
        tk.Label(self.CLAHE_frame,text="App to enhance local contrast in image stacks (CLAHE)").grid(row=0)
        self.top_frame_CE=tk.Frame(self.CLAHE_frame, width = nbfw, height=nbfh, pady=3)
        self.cen_frame_CE=tk.Frame(self.CLAHE_frame, width = nbfw, height=nbfh*len(paramsCE[1:]), pady=3)
        self.btm_frame_CE=tk.Frame(self.CLAHE_frame, width = nbfw, height=nbfh, pady=3)
        
        # Layout the containers
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.top_frame.grid(row=0, sticky="ew")
        self.cen_frame.grid(row=1, sticky="ew")
        self.btm_frame.grid(row=2, sticky="ew")

        self.fpath=tk.Label(self.top_frame,text=fields[0][0])
        self.fp=tk.StringVar(self.top_frame)
        self.fpath_val=tk.Entry(self.top_frame,textvariable=self.fp)
        self.browse_button = tk.Button(self.top_frame,text="Browse", fg="green",command=partial(self.browseSt,self.fp))
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.top_frame_CE.grid(row=0, sticky="ew")
        self.cen_frame_CE.grid(row=1, sticky="ew")
        self.btm_frame_CE.grid(row=2, sticky="ew")
                
        self.fpath_CE=tk.Label(self.top_frame_CE,text=fields[0][0])
        self.fp_CE=tk.StringVar(self.top_frame_CE)
        self.fpath_val_CE=tk.Entry(self.top_frame_CE,textvariable=self.fp_CE)
        self.browse_button_CE = tk.Button(self.top_frame_CE,text="Browse", fg="green",command=partial(self.browseSt,self.fp_CE))

        # layout the widgets in the top frame
        self.fpath.grid(row=0)
        self.fpath_val.grid(row=0,column=1)
        self.browse_button.grid(row=0,column=2)

        self.fpath_CE.grid(row=0)
        self.fpath_val_CE.grid(row=0,column=1)
        self.browse_button_CE.grid(row=0,column=2)
        
        self.ents=self.makeFrame(self.cen_frame,fields,self.fp)
        
        self.ents_CE=self.makeFrame(self.cen_frame_CE,fieldsCE,self.fp_CE)
        
        # Widgets of the bottom frame
        self.run_button = tk.Button(self.btm_frame,text="Run", fg="Red",command=partial(self.CorrInt,self.ents))
        self.quit_button = tk.Button(self.btm_frame, text='Quit', command=self.cancel_b)

        self.run_button.grid(row=0,column=0)
        self.quit_button.grid(row=0,column=2)

        self.run_button = tk.Button(self.btm_frame_CE,text="Run", fg="Red",command=partial(self.calcCLAHE,self.ents_CE))
        self.quit_button = tk.Button(self.btm_frame_CE, text='Quit', command=self.cancel_b)
        self.prev_button= tk.Button(self.btm_frame_CE, text='Preview', command=partial(self.preview_window_CLAHE,self.ents_CE))
        
        self.run_button.grid(row=0,column=0)
        self.quit_button.grid(row=0,column=2)
        self.prev_button.grid(row=0,column=3)

        root.bind('<Return>', (lambda event: self.fetch(self.ents))) 
        root.bind('<Tab>', (lambda event: self.fetch(self.ents_CE))) 

    def cancel_b(self):
        self.quit()
        self.master.destroy()

    def browseSt(self,fpval):
        idir='/'
        if 'Win' in platform.system():
            idir = 'W:/'
        if 'Darwin' in platform.system():
            idir = "/Volumes/Data/Luca_Work/MPI/Science/Coding/Python"
        dirname = tk.filedialog.askdirectory(initialdir = idir ,title = 'Select path to image stack to be aligned')
        if dirname:
            fpval.set(dirname)
        return

            
    def makeFrame(self,parent,fieldz,fpathval):
        entries=[]
        entries.insert(0,(fieldz[0][0],fpathval))
        for i in range(1,len(fieldz)):
           lab = tk.Label(parent, width=25, text=fieldz[i][0], anchor='w')
           ent_txt=tk.StringVar(parent,value=fieldz[i][1])
           ent = tk.Entry(parent,textvariable=ent_txt)
           ent.config(justify=tk.RIGHT)
           lab.grid(row=i,column=0)
           ent.grid(row=i,column=1)
           entries.append((fieldz[i][0], ent))
        return entries

    def fetch(self,fieldz):
        print('%s: "%s"' % (fields[0][0],self.fp.get()))
        for entry in fieldz[1:]:
           field = entry[0]
           text  = entry[1].get()
           print('%s: "%s"' % (field, text))
        print("----------")

    def arggen(self,fieldz):
        args=[]
        for entry in fieldz:
            field = entry[0]
            text  = entry[1].get()
            args.append(text)
        return args

    def load_images_from_folder(self,stfolder,fext):
        images = []
        fnames=[]
        for filename in os.listdir(stfolder):
            if not fext in filename: 
                continue
            img = cv2.cvtColor(cv2.imread(os.path.join(stfolder,filename)),cv2.COLOR_BGR2GRAY)
            if img is not None:
                images.append(img)
                fnames.append(filename)
                self.loading.progress2['value']+=1
                self.update()
        return images,fnames            

    def load_image_from_folder(self,stfolder,fext,fnum):
        nfiles=self.count_files_with_ext(fext,stfolder)
        fnames=[]
        for filename in os.listdir(stfolder):
            if not fext in filename: 
                continue
            fnames.append(filename)
        for i in range(nfiles):
            if i==fnum:
                img = cv2.cvtColor(cv2.imread(os.path.join(stfolder,fnames[fnum])),cv2.COLOR_BGR2GRAY)
        return img    

    def count_files_with_ext(self,ext,folder):
        fnum=Counter(ext in fname for fname in os.listdir(folder))[1]
        return fnum

    def preview_window_CLAHE(self,entries):
        args=self.arggen(entries)
        stfolder,NC,ext,sfn,ts,cl,nb=args[0],int(args[1]),args[2],args[3],int(args[4]),float(args[5]),int(args[6])
        imgarray=self.load_image_from_folder(stfolder,ext,0)
        self.wcounter += 1
        self.prevs[self.wcounter] = tk.Toplevel(self,width=imgarray.shape[1],height=imgarray.shape[0]/2)
        self.prevs[self.wcounter].wm_title("Window #%s" % self.wcounter)
        self.mainframes[self.wcounter] = tk.Frame(self.prevs[self.wcounter],width=imgarray.shape[1],height=imgarray.shape[0]/2)
        self.mainframes[self.wcounter].pack()
        self.prevlab[self.wcounter] = tk.Label(self.mainframes[self.wcounter], text="Preview for CLAHE")
        self.prevlab[self.wcounter].pack(side='top')
        self.prevparams[self.wcounter] = tk.Label(self.mainframes[self.wcounter], text="Parameters: Tile Size="+str(ts)+" - Clip Limit="+str(cl)+" - Nbins="+str(nb))
        self.prevparams[self.wcounter].pack(side='top')
        prev_rat=1
        if 2*imgarray.shape[1] > self.master.winfo_screenwidth():
            prev_rat=int(imgarray.shape[1]/self.master.winfo_screenwidth())+1
        prevsize=(imgarray.shape[0]/2/prev_rat,imgarray.shape[1]/2/prev_rat)
        self.prev_c_orig[self.wcounter]=tk.Canvas(self.mainframes[self.wcounter],width=prevsize[1],height=prevsize[0])
        self.prev_c_orig[self.wcounter].pack(side='left')
        self.imgp[self.wcounter]=Image.fromarray(imgarray)
        self.fig[self.wcounter]=ImageTk.PhotoImage(self.imgp[self.wcounter])
        self.prev_c_orig[self.wcounter].create_image(0,0,image=self.fig[self.wcounter],anchor=tk.NW)
        
        self.prev_c_CE[self.wcounter]=tk.Canvas(self.mainframes[self.wcounter],width=prevsize[1],height=prevsize[0])
        self.prev_c_CE[self.wcounter].pack(side='left')
#        self.cld[self.wcounter]=clahefunct2(imgarray,ts,cl,nb,stfolder,sfn,ext,'','Y')
        self.cld[self.wcounter]=clahefunct(imgarray,ts,cl,stfolder,sfn,ext,'','Y')
        self.figCE[self.wcounter]=ImageTk.PhotoImage(self.cld[self.wcounter])
        self.prev_c_CE[self.wcounter].create_image(0,0,image=self.figCE[self.wcounter],anchor=tk.NW)

    def intcorrcalc_mp(self,img1,img0,rev,nCORES):
        x=img0.flatten().astype(np.float64)
        y=img1.flatten().astype(np.float64)
        am=[]
        bm=[]
        results=[]
        pool=Pool(nCORES)
        for i in range(len(x)-1):
            results=pool.apply_async(abmatcalc,(y,x,i,rev,))
            if results.get()[0] != 'None':
                am.append(results.get()[0])
                bm.append(results.get()[1])
                print(am)
        pool.close()
        pool.join()
        return np.average(am),np.average(bm),np.std(am),np.std(bm)

    def intcorrcalc(self,img1,img0,rev,nCORES):
        x=img0.flatten().astype(np.float64)
        y=img1.flatten().astype(np.float64)
#        print(np.average(x)-np.average(y),np.std(x),np.std(y),2*(np.std(x)-np.std(y))/(np.std(x)+np.std(y)))
        am=[]
        bm=[]
        results=[]
        for i in range(len(x)-1):
            results=abmatcalc(y,x,i,rev)
            if results[0] != 'None':
                am.append(results[0])
                bm.append(results[1])
#        plt.scatter(linspace(0,len(am),len(am)),am)
#        plt.show()
        if abs(np.average(x)-np.average(y)) < (np.std(x)+np.std(y))/2/15 and abs(2*(np.std(x)-np.std(y))/(np.std(x)+np.std(y)))<0.02:
            amt,bmt,asdm,bsdm=1-abs(1-np.average(am))/5,np.average(bm)/5,np.std(am),np.std(bm)/5
        else:
            amt,bmt,asdm,bsdm=np.average(am),np.average(bm),np.std(am),np.std(bm)
#        if abs(1-amt)<asdm/6:
#            amt,bmt,asdm,bsdm=1,0,0,0
#        print(amt,bmt)
        return amt,bmt,asdm,bsdm

    def imgcorrection(self,im1,im0,redf,gaussks,fname,fext,stfolder,sfn,nCORES,rev):
        if not sfn in os.listdir(stfolder):
            os.makedirs(os.path.join(stfolder,sfn))
            print('directory created')
        im1r,im0r=denandred(im1,redf,gaussks),denandred(im0,redf,gaussks)
#        im1r,im0r=denandred(im1,redf,25),denandred(im0,redf,25)
        am,bm,asd,bsd=self.intcorrcalc(im1r,im0r,rev,nCORES)
        if rev==0:
            cimg=(im1-bm)/am
        if rev==1:
            cimg=im1*am+bm 
        corrimg=misc.toimage(cimg,cmin=0,cmax=255)
        save_tif(stfolder,sfn,fname,corrimg,fext)
        return am,bm,asd,bsd,cimg

    def corrimgint(self,imstack,fnames,ind,redf,gaussks,nCORES,fext,stfolder,sfn):
        amat=[]
        bmat=[]
        asdmat=[]
        bsdmat=[]
        rev=0
        print('computing intensity correction factors with',nCORES,'processes in parallel')
        for i in range(ind,len(imstack)-1):
            if i == ind:
                cprev=np.copy(imstack[i])
            results=[]
            results=self.imgcorrection(imstack[i+1],cprev,redf,gaussks,fnames[i+1],fext,stfolder,sfn,nCORES,rev)
            self.loading.progress2['value']+=1
            self.update()
            amat.append(results[0])
            bmat.append(results[1])
            asdmat.append(results[2])
            bsdmat.append(results[3])
            cprev=np.copy(results[4])
        results=self.imgcorrection(imstack[ind],imstack[ind],redf,gaussks,fnames[ind],fext,stfolder,sfn,nCORES,rev)
        amat.insert(0,results[0])
        bmat.insert(0,results[1])
        asdmat.insert(0,results[2])
        bsdmat.insert(0,results[3])
        print(fnames[ind],ind,results[0])
        rev=1
        for i in range(ind,0,-1):
            if i == ind:
                cnext=np.copy(imstack[i])
            results=[]
            results=self.imgcorrection(imstack[i-1],cnext,redf,gaussks,fnames[i-1],fext,stfolder,sfn,nCORES,rev)
            self.loading.progress2['value']+=1
            self.update()
            amat.insert(0,results[0])
            bmat.insert(0,results[1])
            asdmat.insert(0,results[2])
            bsdmat.insert(0,results[3])
            cnext=np.copy(results[4])
        amat,bmat,asdmat,bsdmat=np.array(amat),np.array(bmat),np.array(asdmat),np.array(bsdmat)
        return amat,bmat,asdmat,bsdmat


    def CorrInt(self,entries):
        global am,bm,asd,bsd
        # Read the images to be aligned
        args=self.arggen(entries)
        stfolder,nC,ext,sfn,red,rimgN=args[0],int(args[1]),args[2],args[3],int(args[4]),args[5] 
        noffiles=count_files_with_ext(ext,stfolder)
        Nsteps=2
        self.loading=ProgWin(self.master,Nsteps,noffiles)
        start = time.time()
        print("Image loading started on", time.asctime())
        self.loading.prg_status['text']="Image loading started on "+ str(time.asctime())
        self.update()
        stack,fnames=self.load_images_from_folder(stfolder,ext)
        stack=np.array(stack)
        rimind=fnames.index([i for i in fnames if rimgN in i][0])
        gks=2*int(np.rint(stack[0].shape[1]/red*0.025)/2)+1
        print(stack[0].shape[1]/red,stack[0].shape[0]/red,gks)
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
        start = time.time()
        msg='Image correction started on'
        print(msg, time.asctime())
        self.loading.prg_status['text']=msg+ str(time.asctime())
        self.loading.progress2['value']=0
        self.update()        
        am,bm,asd,bsd=self.corrimgint(stack,fnames,rimind,red,gks,nC,ext,stfolder,sfn)
        end = time.time()
        secs = end-start
        msg='Image correction took'
        print(msg, secs)
        self.loading.prg_status['text']=msg+ str(secs)
        self.loading.progress['value']+=1
        self.update()
        print('Done')
        self.loading.destroy() 
        return

#    def clahe_mp(self,imstack,fnames,ts,cl,nb,nCORES,ext,stfolder,sfn,prev):
    def clahe_mp(self,imstack,fnames,ts,cl,nCORES,ext,stfolder,sfn,prev):
        if not sfn in os.listdir(stfolder):
            os.makedirs(os.path.join(stfolder,sfn))
            print('directory created')
        pool=Pool(nCORES)
        print('computing contranst enhancement with',nCORES,'processes in parallel')
        vmin, vmax = np.min(imstack),np.max(imstack)
        if vmax-vmin <= 1:
            imstack = np.copy(imstack)/255
        results=[]
        for i in range(len(imstack)):
            results.append(pool.apply_async(clahefunct,(imstack[i],ts,cl,stfolder,sfn,ext,fnames[i],'N',)))
#            results.append(pool.apply_async(clahefunct2,(imstack[i],ts,cl,nb,stfolder,sfn,ext,fnames[i],'N',)))
            self.loading.progress2['value']+=1
            self.update()
        pool.close()
        pool.join()        
        return
    
    def calcCLAHE(self,entries):
        global am,bm,asd,bsd
        # Read the images to be aligned
        args=self.arggen(entries)
        stfolder,nC,ext,sfn,ts,cl,nb=args[0],int(args[1]),args[2],args[3],int(args[4]),float(args[5]),int(args[6])
        noffiles=count_files_with_ext(ext,stfolder)
        Nsteps=2
        self.loading=ProgWin(self.master,Nsteps,noffiles)
        start = time.time()
        msg='Image loading started on '
        print(msg, time.asctime())
        self.loading.prg_status['text']=msg+ str(time.asctime())
        self.update()
        stack,fnames=self.load_images_from_folder(stfolder,ext)
        stack=np.array(stack)
        end = time.time()
        secs = end-start
        msg='Image loading took '
        print(msg, secs)
        self.loading.prg_status['text']=msg+ str(secs)
        self.loading.progress['value']+=1
        self.update()
        nimages=len(stack)
        print("N. of Images: ",nimages)
        start = time.time()
        msg='Contrast enhancement started on '
        print(msg, time.asctime())
        self.loading.prg_status['text']=msg+str(time.asctime())
        self.loading.progress2['value']=0
        self.update()        
#        self.clahe_mp(stack,fnames,ts,cl,nb,nC,ext,stfolder,sfn,'N')
        self.clahe_mp(stack,fnames,ts,cl,nC,ext,stfolder,sfn,'N')
        end = time.time()
        secs = end-start
        msg='Contrast enhancement took '
        print(msg, secs)
        self.loading.prg_status['text']=msg+ str(secs)
        self.loading.progress['value']+=1
        self.update()
        print('Done')
        self.loading.destroy() 
        return

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
        
        self.deftxt=tk.Label(self,text="Correction is in progress")
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

mfw,mfh=600,350
nbfw,nbfh=550,40    
    
if __name__ == '__main__':
    initdir="/Volumes/Data/Luca_Work/MPI/Science/Coding/Python"
    params=('Stack path','Number of Cores (out of '+str(os.cpu_count())+')','Files extension','Subdir name','Img Reduction','Reference image contains')
    paramsCE=('Stack path','Number of Cores (out of '+str(os.cpu_count())+')','Files extension','Subdir name','Tile size','Clip Limit','Bins (not used ATM)')
    deftexts=('/',3,'.tif','IntCorr',6,'0090')
    deftextsCE=('/',3,'.tif','CLAHE',20,2,256)
    fields=[]
    if len(params)==len(deftexts):
        for i in range(len(params)):
            tmp=(params[i],deftexts[i])
            fields.append(tmp)
    fieldsCE=[]
    if len(paramsCE)==len(deftextsCE):
        for i in range(len(paramsCE)):
            tmp=(paramsCE[i],deftextsCE[i])
            fieldsCE.append(tmp)
    root=tk.Tk()
    app=App(root)
    app.mainloop()
