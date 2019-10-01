#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 18:54:59 2017

@author: Luca
"""

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
from multiprocessing import Process
from multiprocessing import Pool
from collections import Counter
import numpy as np
import copy
import os
import cv2
import matplotlib.pyplot as plt
from scipy import misc,fftpack
from functools import partial
import time
from PIL import Image, ImageTk
import platform


def save_tif(folder,subdir,imname,im,ext):
    misc.imsave(os.path.join(folder,subdir,imname.split('.')[0]+'_DS'+ext),im)
    
def denandred(img,red,gausskernel):
    imgred=cv2.resize(img,(int(np.rint(img.shape[1]/red)), int(np.rint(img.shape[0]/red))), interpolation = cv2.INTER_AREA)
    imgredblur=cv2.GaussianBlur(imgred,(gausskernel,gausskernel),0)
    return imgredblur

def RemCurt(im,wt,wd,stfolder,sfn,fname,prev,ext):
    H,W=im.shape
    fft2 = fftpack.fft2(im) #fft of img of the stack
    fft2sh = fftpack.fftshift(fft2)#shifted fft of img of the stack
    fft2shcorr=copy.copy(fft2sh)
    # calculation of the corrected FFT modulus of img, comment if using the simpler version of the filter
    for i in range(W):
        if abs(W/2 - i) > wd: 
            for k in range(2*wt+1):
                if abs(fft2sh[int(H/2-wt/2+k),i]) != 0:
                    fft2shcorr[int(H/2-wt/2+k),i]=0
#    fft2shcorrArr.append(fft2shcorr)
    ifft2shcorr=fftpack.ifftshift(fft2shcorr)
    ifft2corr = fftpack.ifft2(ifft2shcorr)
    CorrImg=abs(ifft2corr)
#    fft2shArr.append(fft2sh)
    if prev == 'N':
        save_tif(stfolder,sfn,fname,misc.toimage(255*CorrImg,cmin=0,cmax=255),ext)
#    print(j)
    return CorrImg


class App(tk.Frame):
    wcounter=0
    prevs=dict()
    mainframes=dict()
    prevlab=dict()
    prevparams=dict()
    prev_c_orig=dict()
    imgp=dict()
    figp=dict()
    prev_c_den=dict()
    imgden=dict()
    figden=dict()
    def __init__(self,master):
        tk.Frame.__init__(self,master)
        self.master.title("Remove curtaining from stack App")
        self.master.resizable(False,False)
        self.master.tk_setPalette(background='#ececec')
        x=int((self.master.winfo_screenwidth()-self.master.winfo_reqwidth())/1.5)
        y=int((self.master.winfo_screenheight()-self.master.winfo_reqheight())/2)
        self.master.geometry('{}x{}'.format(x,y))
        
        self.n=ttk.Notebook(master)
        self.RC_frame=tk.Frame(self.n, width = mfw, height=mfh)        
        self.n.add(self.RC_frame, text='Remove curtaining')
        self.n.pack()
        
        # Create the main containers for the Intensity correction notebook
        tk.Label(self.RC_frame,text="App to remove curtaining from image stacks").grid(row=0)
        self.top_frame=tk.Frame(self.RC_frame, width = nbfw, height=nbfh, pady=3)
        self.cen_frame=tk.Frame(self.RC_frame, width = nbfw, height=nbfh*len(params[1:]), pady=3)
        self.btm_frame=tk.Frame(self.RC_frame, width = nbfw, height=nbfh, pady=3)
        
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

        # layout the widgets in the top frame
        self.fpath.grid(row=0)
        self.fpath_val.grid(row=0,column=1)
        self.browse_button.grid(row=0,column=2)

        self.ents=self.makeFrame(self.cen_frame,fields,self.fp)
        
        # Widgets of the bottom frame
        self.run_button = tk.Button(self.btm_frame,text="Run", fg="Red",command=partial(self.RemCurt,self.ents))
        self.quit_button = tk.Button(self.btm_frame, text='Quit', command=self.cancel_b)
        self.prev_button= tk.Button(self.btm_frame, text='Preview', command=partial(self.preview_window_FFTfilt,self.ents))
        
        self.run_button.grid(row=0,column=0)
        self.quit_button.grid(row=0,column=2)
        self.prev_button.grid(row=0,column=3)

        root.bind('<Return>', (lambda event: self.fetch(self.ents))) 
        
    def cancel_b(self):
        self.quit()
        self.master.destroy()
        return

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
        return

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
        fcount=Counter(ext in fname for fname in os.listdir(folder))[1]
        return fcount

        
    def RemCurt(self,entries):
        # Read the images to be aligned
        args=self.arggen(entries)
        stfolder,nC,ext,sfn,wt,wd=args[0],int(args[1]),args[2],args[3],int(args[4]),int(args[5])
        noffiles=self.count_files_with_ext(ext,stfolder)
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
        msg='Curtaining removal started on '
        print(msg, time.asctime())
        self.loading.prg_status['text']=msg+str(time.asctime())
        self.loading.progress2['value']=0
        self.update()        
        self.CurtRem_mp(stack,fnames,wt,wd,nC,ext,stfolder,sfn)
        end = time.time()
        secs = end-start
        msg='Curtaining removal took '
        print(msg, secs)
        self.loading.prg_status['text']=msg+ str(secs)
        self.loading.progress['value']+=1
        self.update()
        print('Done')
        self.loading.destroy() 
        return
    
    def CurtRem_mp(self,imstack,fnames,wt,wd,nCORES,ext,stfolder,sfn):
        if not sfn in os.listdir(stfolder):
            os.makedirs(os.path.join(stfolder,sfn))
            print('directory created')
        print('running with',nCORES,'processes in parallel')
        pool=Pool(nCORES)
        print('computing FFT with',nCORES,'processes in parallel')
        vmin, vmax = np.min(imstack),np.max(imstack)
        if vmax > 1:
            imstack = np.copy(imstack)/255
        results=[]
        for i in range(len(imstack)):
            pool.apply_async(RemCurt,(imstack[i],wt,wd,stfolder,sfn,fnames[i],'N',ext,))
            self.loading.progress2['value']+=1
            self.update()            
        pool.close()
        pool.join()
        return results 

    def make_prev_win(self,counter,imgarray,ltext,txttitle,funct,*args):
        self.prevs[counter] = tk.Toplevel(self,width=imgarray.shape[1],height=imgarray.shape[0]/2)
        self.prevs[counter].wm_title("Window #%s" % counter)
        self.mainframes[counter] = tk.Frame(self.prevs[counter],width=imgarray.shape[1],height=imgarray.shape[0]/2)
        self.mainframes[counter].pack()
        self.prevlab[counter] = tk.Label(self.mainframes[counter], text=txttitle)
        self.prevlab[counter].pack(side='top')
        self.prevparams[counter] = tk.Label(self.mainframes[counter], text=ltext)
        self.prevparams[counter].pack(side='top')
        prev_rat=1
        if 2*imgarray.shape[1] > self.master.winfo_screenwidth():
            prev_rat=int(imgarray.shape[1]/self.master.winfo_screenwidth())+1
        prevsize=(imgarray.shape[0]/2/prev_rat,imgarray.shape[1]/2/prev_rat)
        self.prev_c_orig[counter]=tk.Canvas(self.mainframes[counter],width=prevsize[1],height=prevsize[0])
        self.prev_c_orig[counter].pack(side='left')
        self.imgp[counter]=Image.fromarray(imgarray)
        self.figp[counter]=ImageTk.PhotoImage(self.imgp[counter])
        self.prev_c_orig[counter].create_image(0,0,image=self.figp[counter],anchor=tk.CENTER)
        self.prev_c_den[counter]=tk.Canvas(self.mainframes[counter],width=prevsize[1],height=prevsize[0])
        self.prev_c_den[counter].pack(side='left')
        if 'den_nlm' in str(funct):
            vmax = np.max(args[0])
            if vmax > 1:
                newargs=np.copy(args)
                newargs[0] = np.copy(args[0])/255
                self.imgden[self.wcounter]=Image.fromarray(funct(*newargs))
        else:
            self.imgden[self.wcounter]=Image.fromarray(funct(*args))
        self.figden[self.wcounter]=ImageTk.PhotoImage(self.imgden[self.wcounter])
        self.prev_c_den[self.wcounter].create_image(0,0,image=self.figden[self.wcounter],anchor=tk.CENTER)
        return    

    def preview_window_FFTfilt(self,entries):
        args=self.arggen(entries)
        stfolder,nC,ext,sfn,wt,wd=args[0],int(args[1]),args[2],args[3],int(args[4]),int(args[5])
        im=self.load_image_from_folder(stfolder,ext,0)
        self.wcounter += 1
        text='Destriping Preview'
        texttolabel="Parameters: Window Thickness="+str(wt)+" - Distance from center="+str(wd)
        self.make_prev_win(self.wcounter,im,texttolabel,text,RemCurt,im,wt,wd,stfolder,sfn,'','Y',ext)
        return
    

class ProgWin(tk.Frame):
    def __init__(self,master,count,count2):
        tk.Frame.__init__(self,master,borderwidth=5,relief='groove')
        self.pack()
        
        self.deftxt=tk.Label(self,text="Denoising is in progress")
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
        
mfw,mfh=600,350
nbfw,nbfh=550,40    
    
if __name__ == '__main__':
    initdir="/Volumes/Data/Luca_Work/MPI/Science/Coding/Python"
    params=('Stack path','Number of Cores (out of '+str(os.cpu_count())+')','Files extension','Subdir name','Window Thickness','Distance from the center')
    deftexts=('/',3,'.tif','Destr',6,6)
    fields=[]
    if len(params)==len(deftexts):
        for i in range(len(params)):
            tmp=(params[i],deftexts[i])
            fields.append(tmp)
    root=tk.Tk()
    app=App(root)
    app.mainloop()