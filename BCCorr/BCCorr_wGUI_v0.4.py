#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 22:46:00 2017

@author: Luca
"""

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
from multiprocessing import Process,Queue
from collections import Counter
import os
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
from scipy import misc,linspace, polyval, polyfit, sqrt, stats, randn
from functools import partial
import time
from PIL import Image, ImageTk
from skimage import exposure
import platform
import math

def count_files_with_ext(ext,folder):
    fnum=Counter(ext in fname for fname in os.listdir(folder))[1]
    return fnum

def save_tif(folder,subdir,imname,im,ext):
    fp=os.path.join(folder,subdir,imname.split('.')[0]+'_IC'+ext)
    image=Image.fromarray(im)
    image.save(fp)
    
def save_tif2(folder,subdir,imname,im,ext):
    fp=os.path.join(folder,subdir,imname.split('.')[0]+'_CEn'+ext)
    im.save(fp)

def im2vec(im):
    vec=[]
    vectmp=im.flatten().astype(np.float64)
    vec[:]=[x for x in vectmp if x>4 and x<250]
    return vec

def clahefunct(img,fname,ts,cl,stfolder,sfn,ext,prev):
    clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=(ts,ts))
    tmp=clahe.apply(img)
    CEimg=Image.fromarray(np.uint8(np.clip(tmp,0,255)),'L')
    if prev=='N':
        save_tif2(stfolder,sfn,fname,CEimg,ext)
    return CEimg

def clahefunct2(img,ts,cl,nb,stfolder,sfn,ext,fname,prev):
    tmp = exposure.equalize_adapthist(img,kernel_size=ts,clip_limit=cl/100,nbins=nb)
    CEimg=Image.fromarray(np.uint8(np.clip(tmp,0,255)),'L')
    if prev=='N':
        save_tif2(stfolder,sfn,fname,CEimg,ext)
    return CEimg

def hist_match(source, template,fname,fext,stfolder,sfn):
    """
    adapted from https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    corrimg=interp_t_values[bin_idx].reshape(oldshape)
    save_tif(stfolder,sfn,fname,corrimg.astype('uint8'),fext)

    return corrimg

def worker(images, fnames, out_q,func,*args):
    """ The worker function, invoked in a process. 'images' is a
        list of images to span the process upon. The results are placed in
        a dictionary that's pushed to a queue.
    """
    outdict = {}
    for imn in range(len(images)):
        outdict[imn] = ''
        func(images[imn],fnames[imn],*args[2:],'N')
    out_q.put(outdict)

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
        self.canv_speed = int(imgarray.shape[1]/4)
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
        self.prev_c_orig[self.wcounter].create_image(0,0,image=self.fig[self.wcounter],anchor=tk.NW,tags='orig')
        self.prev_c_orig[self.wcounter].config(scrollregion=(0,0,imgarray.shape[1],imgarray.shape[0]))
        self.prev_c_CE[self.wcounter]=tk.Canvas(self.mainframes[self.wcounter],width=prevsize[1],height=prevsize[0])
        self.prev_c_CE[self.wcounter].config(scrollregion=(0,0,imgarray.shape[1],imgarray.shape[0]))
        self.prev_c_CE[self.wcounter].pack(side='left')
#        self.cld[self.wcounter]=clahefunct2(imgarray,ts,cl,nb,stfolder,sfn,ext,'','Y')
        self.cld[self.wcounter]=clahefunct(imgarray,'',ts,cl,stfolder,sfn,ext,'Y')
        self.figCE[self.wcounter]=ImageTk.PhotoImage(self.cld[self.wcounter])
        self.prev_c_CE[self.wcounter].create_image(0,0,image=self.figCE[self.wcounter],anchor=tk.NW,tags='mod')
        
        self.prevs[self.wcounter].bind('<KeyPress>', lambda event, ct=self.wcounter: self.move_canvas_arrowkey(event,ct))
        self.prev_c_orig[self.wcounter].tag_bind('orig','<Button1-Motion>', lambda event, ct=self.wcounter: self.move_canvas_mouse(event,ct))
        self.prev_c_orig[self.wcounter].tag_bind('orig','<ButtonRelease-1>', lambda event, ct=self.wcounter: self.release_canvas_mouse(event,ct))
        self.move_flag = False
        return    

    def move_canvas_arrowkey(self, event, counter):
#        print(event.keysym)
        if event.keysym == "Up":
            self.prev_c_orig[counter].move('orig', 0, -self.canv_speed)
            self.prev_c_CE[counter].move('mod', 0, -self.canv_speed)
        elif event.keysym == "Down":
            self.prev_c_orig[counter].move('orig', 0, self.canv_speed)
            self.prev_c_CE[counter].move('mod', 0, self.canv_speed)
        elif event.keysym == "Left":
            self.prev_c_orig[counter].move('orig', -self.canv_speed, 0)
            self.prev_c_CE[counter].move('mod', -self.canv_speed, 0)
        elif event.keysym == "Right":
            self.prev_c_orig[counter].move('orig', self.canv_speed, 0)
            self.prev_c_CE[counter].move('mod', self.canv_speed, 0)
    
    def move_canvas_mouse(self, event, counter):
        if self.move_flag:
            new_xpos, new_ypos = event.x, event.y
             
            self.prev_c_orig[counter].move('orig',new_xpos-self.mouse_xpos ,new_ypos-self.mouse_ypos)
            self.prev_c_CE[counter].move('mod',new_xpos-self.mouse_xpos ,new_ypos-self.mouse_ypos)
             
            self.mouse_xpos = new_xpos
            self.mouse_ypos = new_ypos
        else:
            self.move_flag = True
            self.prev_c_orig[counter].tag_raise('orig')
            self.prev_c_CE[counter].tag_raise('mod')
            self.mouse_xpos = event.x
            self.mouse_ypos = event.y
 
    def release_canvas_mouse(self, event, counter):
        self.move_flag = False

    def corrimgint(self,imstack,fnames,ind,fext,stfolder,sfn):
        if not sfn in os.listdir(stfolder):
            os.makedirs(os.path.join(stfolder,sfn))
            print('directory created')
        cprev,cnext=np.copy(imstack[ind]),np.copy(imstack[ind])
        cstack=[]
        print('computing intensity correction factors with 1 core')
        for i in range(ind,len(imstack)-1):
            result=hist_match(imstack[i+1],cprev,fnames[i+1],fext,stfolder,sfn)
            cstack.append(result)
            self.loading.progress2['value']+=1
            self.update()
            cprev=np.copy(result)
        result=hist_match(imstack[ind],imstack[ind],fnames[ind],fext,stfolder,sfn)
        cstack.insert(0,result)
        for i in range(ind,0,-1):
            result=hist_match(imstack[i-1],cnext,fnames[i-1],fext,stfolder,sfn)
            cstack.insert(0,result)
            self.loading.progress2['value']+=1
            self.update()
            cnext=np.copy(result)
        return 


    def CorrInt(self,entries):
        global am,bm,asd,bsd
        # Read the images to be aligned
        args=self.arggen(entries)
        stfolder,ext,sfn,rimgN=args[0],args[1],args[2],args[3] 
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
        self.corrimgint(stack,fnames,rimind,ext,stfolder,sfn)
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
        self.mp_process(nC,clahefunct,stack,fnames,ts,cl,stfolder,sfn,ext)
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

    def mp_process(self,nprocs,func,*args):
        images=args[0]
        fnames=args[1]
        stfolder=args[4]
        sfn=args[5]
        if not sfn in os.listdir(stfolder):
           os.makedirs(os.path.join(stfolder,sfn))
           print('directory created')

        out_q = Queue()
        chunksize = int(math.ceil(len(images) / float(nprocs)))
        procs = []
    
        for i in range(nprocs):
            p = Process(
                    target=worker,
                    args=(images[chunksize * i:chunksize * (i + 1)],fnames[chunksize * i:chunksize * (i + 1)],out_q,func,*args))
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
    
        return resultdict

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
    params=('Stack path','Files extension','Subdir name','Reference image contains')
    paramsCE=('Stack path','Number of Cores (out of '+str(os.cpu_count())+')','Files extension','Subdir name','Tile size','Clip Limit','Bins (not used ATM)')
    deftexts=('/','.tif','IntCorr','0090')
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


