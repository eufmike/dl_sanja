#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 15:52:03 2017

@author: bertinetti
"""

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
from tkinter import messagebox
import os
from multiprocessing import Process,Queue
from multiprocessing import Pool
from collections import Counter
import numpy as np
import copy
import os
import cv2
from PIL import Image, ImageTk
from functools import partial
#import skimage.transform 
from skimage import img_as_ubyte
from skimage.filters import threshold_otsu, threshold_local, rank, threshold_niblack, threshold_sauvola
from skimage.morphology import disk,remove_small_objects, skeletonize
#from skimage.feature import peak_local_max
#from skimage.morphology import watershed
#from skimage import measure
import matplotlib.pyplot as plt
import time
import platform
import math

def count_files_with_ext(ext,folder):
    fnum=Counter(ext in fname for fname in os.listdir(folder))[1]
    return fnum

def save_tif(folder,subdir,imname,im,ttype,ext):
    fp=os.path.join(folder,subdir,imname.split('.')[0]+'_LT_'+ttype+ext)
#    image=Image.fromarray(im)
    im.save(fp)

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

def thr_conn(img,fname,stfolder,sfn,thrtyp,bs,mp,kv,rv,inv,ext,prev):
    ofs=-5
    if thrtyp=='Gauss':
        tmp=cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, ofs)
        thrconnimg=Image.fromarray(tmp,mode='L')
    if thrtyp=='Mean':
        tmp=cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, bs, ofs)
        thrconnimg=Image.fromarray(tmp,mode='L')
    if thrtyp=='Gauss Sk':
        ad_thr = threshold_local(img, bs,method='gaussian', offset=ofs)
        tmp = 255*(img > ad_thr).astype('uint8')
        thrconnimg=Image.fromarray(tmp,mode='L')
    if thrtyp=='Mean Sk':
        ad_thr = threshold_local(img, bs,method='mean', offset=ofs)
        tmp = 255*(img > ad_thr).astype('uint8')
        thrconnimg=Image.fromarray(tmp,mode='L')
    if thrtyp=='Median Sk':
        ad_thr = threshold_local(img, bs,method='median', offset=ofs)
        tmp = 255*(img > ad_thr).astype('uint8')
        thrconnimg=Image.fromarray(tmp,mode='L')
    if thrtyp=='Otsu':
        ret,tmp=cv2.threshold(img, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thrconnimg=Image.fromarray(tmp,mode='L')
    if thrtyp=='Otsu Loc':
        radius = bs
        selem = disk(radius)
        local_otsu = rank.otsu(img, selem)
        tmp = 255*(img >= local_otsu).astype('uint8')
        thrconnimg=Image.fromarray(tmp,mode='L')
    if thrtyp=='Niblack':
        ad_thr = threshold_niblack(img, window_size=bs, k=kv)
        tmp = 255*(img > ad_thr).astype('uint8')
        thrconnimg=Image.fromarray(tmp,mode='L')
    if thrtyp=='Sauvola':
        ad_thr = threshold_sauvola(img,  window_size=bs, k=kv, r=rv)
        tmp = 255*(img > ad_thr).astype('uint8')
        thrconnimg=Image.fromarray(tmp,mode='L')
        
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
    openimg = Image.fromarray(cv2.morphologyEx(tmp,cv2.MORPH_OPEN,kernel1),mode='L')
#    openimg2 = Image.fromarray(remove_small_objects((tmp/255).astype('uint8'),mp),mode='L')
    opimg=255*remove_small_objects((tmp/255).astype('uint8').astype('bool'),mp).astype('uint8')
    openimg2 = Image.fromarray(opimg,mode='L')
#    skel=255*skeletonize((opimg/255).astype('uint8'))
#    skelimg=Image.fromarray(skel,mode='L')
    if prev=='N':
        save_tif(stfolder,sfn,fname,openimg2,thrtyp,ext)
    return openimg2#thrconnimg

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
        mfw,mfh=int(self.master.winfo_screenwidth()/2.5),int(root.winfo_screenheight()/2.5)
        nbfw,nbfh=mfw-5,40
        self.master.title("Local Threshold App")
        self.master.resizable(False,False)
        self.master.tk_setPalette(background='#ececec')
        x=int((self.master.winfo_screenwidth()-self.master.winfo_reqwidth())/1.7)
        y=int((self.master.winfo_screenheight()-self.master.winfo_reqheight())/1.7)
        self.master.geometry('{}x{}'.format(x,y))
        
        self.n=ttk.Notebook(master)
        self.cr_frame=tk.Frame(self.n)
        
        self.n.add(self.cr_frame, text='Local Threshold Computation')   
        self.n.pack()
        
        # Create the main containers for the connected regions notebook
        tk.Label(self.cr_frame,text="App to compute local threshold in an image stack").grid(row=0)
        self.top_frame=tk.Frame(self.cr_frame, width = nbfw, height=nbfh, pady=3)
        self.cen_frame=tk.Frame(self.cr_frame, width = nbfw, height=nbfh*len(params[1:]), pady=3)
        self.thr_frame=tk.Frame(self.cr_frame, width = nbfw, height=nbfh, pady=3)
        self.btm_frame=tk.Frame(self.cr_frame, width = nbfw, height=nbfh, pady=3)
        
        # Layout the containers
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.top_frame.grid(row=1, sticky="ew")
        self.cen_frame.grid(row=2, sticky="ew")
        self.thr_frame.grid(row=3, sticky="ew")
        self.btm_frame.grid(row=4, sticky="ew")
        
        self.fpath=tk.Label(self.top_frame,text=fields[0][0])
        self.fp=tk.StringVar(self.top_frame)
        self.fpath_val=tk.Entry(self.top_frame,textvariable=self.fp)
        self.browse_button = tk.Button(self.top_frame,text="Browse", fg="green",command=self.browseSt)
        
        # layout the widgets in the top frame
        self.fpath.grid(row=0)
        self.fpath_val.grid(row=0,column=1)
        self.browse_button.grid(row=0,column=2)
        
        self.ents=self.makeFrame(self.cen_frame)
        
        self.thrmetlab=tk.Label(self.thr_frame,text='Threshold:')
        self.thrmetstr=tk.StringVar(self.thr_frame)
        self.thrmetstr.set('Gauss')
        self.thrmet = tk.OptionMenu(self.thr_frame, self.thrmetstr, 'Gauss', 'Mean', 'Otsu','Gauss Sk','Mean Sk','Median Sk', 'Otsu Loc', 'Niblack', 'Sauvola')
        self.invvar = tk.IntVar()
        self.chkinv = tk.Checkbutton(self.thr_frame, text='Inverted Thr.', variable=self.invvar)
        self.thrmetlab.grid(row=0,column=1)
        self.thrmet.grid(row=0,column=2)
        self.chkinv.grid(row=0,column=3)
        
        
        # Widgets of the bottom frame
#        self.run_button = tk.Button(self.btm_frame,text="Run", fg="Red",command=self.alignfuncSIFT)
        self.run_button = tk.Button(self.btm_frame,text="Run", fg="Red",command=self.local_threshold)
        self.quit_button = tk.Button(self.btm_frame, text='Quit', command=self.cancel_b)
        self.prev_button= tk.Button(self.btm_frame, text='Preview', command=partial(self.preview_window,'Connected Regions',self.ents))


        # layout the widgets in the bottom frame
        self.run_button.grid(row=0,column=0)
        self.quit_button.grid(row=0,column=2)
        self.prev_button.grid(row=0,column=3)
        

        self.master.bind('<Tab>', (lambda event: self.fetch())) 

    def cancel_b(self):
        self.quit()
        self.master.destroy()

    def browseSt(self):
        idir='/'
        if 'Win' in platform.system():
            idir = 'W:/'
        if 'Darwin' in platform.system():
            idir = "/Volumes/Data/Luca_Work/MPI/Science/Coding/Python"
        dirname = tk.filedialog.askdirectory(initialdir = idir ,title = 'Select path to image stack to be aligned')
        if dirname:
            self.fp.set(dirname)
        return

    #Make the main Frame    
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
    
    def fetch(self):
        for entry in self.ents:
           field = entry[0]
           text  = entry[1].get()
           print('%s: "%s"' % (field, text))
        print('%s "%s"' % (self.thrmetlab['text'],self.thrmetstr.get()))
        print('%s "%s"' % (self.chkinv['text'],self.invvar.get()))
        print("----------")
        
    def arggen(self):
        args=[]
        for entry in self.ents:
           text  = entry[1].get()
           args.append(text)
        args.append(self.thrmetstr.get())
        args.append(self.invvar.get())
        return args
    
    def preview_window(self,text,entries):
        args=self.arggen()
        stfolder,NC,ext,sfn,bs,kv,rv,mp,thrtyp,inv=args[0],int(args[1]),args[2],args[3],int(args[4]),float(args[5]),float(args[6]),int(args[7]),args[8],int(args[9])
        imgarray=self.load_image_from_folder(stfolder,ext,0)
        self.wcounter += 1
        texttolabel="Parameters: Type: "+thrtyp+" - BS="+str(bs)+" - Min Pts="+str(mp)+" - k="+str(kv)+" - r="+str(rv)+" - Inv="+str(inv)
        self.make_prev_win(self.wcounter,imgarray,texttolabel,text,thr_conn,imgarray,'',stfolder,sfn,thrtyp,bs,mp,kv,rv,inv,ext,'Y')
        return

    def make_prev_win(self,counter,imgarray,ltext,txttitle,funct,*args):
        self.canv_speed = int(imgarray.shape[1]/4)
        self.prevs[counter] = tk.Toplevel(self,width=imgarray.shape[1],height=imgarray.shape[0]/2)
        self.prevs[counter].wm_title("Window #%s" % counter)
        self.mainframes[counter] = tk.Frame(self.prevs[counter],width=imgarray.shape[1],height=imgarray.shape[0]/2)
        self.mainframes[counter].pack()
        self.prevlab[counter] = tk.Label(self.mainframes[counter], text="Preview for "+txttitle+" evaluation")
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
        self.prev_c_orig[counter].create_image(0,0,image=self.figp[counter],anchor=tk.NW,tags='orig')
        self.prev_c_den[counter]=tk.Canvas(self.mainframes[counter],width=prevsize[1],height=prevsize[0])
        self.prev_c_den[counter].pack(side='left')
        if 'den_nlm' in str(funct):
            vmax = np.max(args[0])
            if vmax > 1:
                newargs=np.copy(args)
                newargs[0] = np.copy(args[0])/255
                self.imgden[self.wcounter]=funct(*newargs)
        else:
            self.imgden[self.wcounter]=funct(*args)
        self.figden[self.wcounter]=ImageTk.PhotoImage(self.imgden[self.wcounter])
        self.prev_c_den[self.wcounter].create_image(0,0,image=self.figden[self.wcounter],anchor=tk.NW,tags='mod')
        
        self.prevs[counter].bind('<KeyPress>', lambda event, ct=counter: self.move_canvas_arrowkey(event,ct))
        self.prev_c_orig[counter].tag_bind('orig','<Button1-Motion>', lambda event, ct=counter: self.move_canvas_mouse(event,ct))
        self.prev_c_orig[counter].tag_bind('orig','<ButtonRelease-1>', lambda event, ct=counter: self.release_canvas_mouse(event,ct))
        self.move_flag = False
        
        return
    
    def move_canvas_arrowkey(self, event, counter):
#        print(event.keysym)
        if event.keysym == "Up":
            self.prev_c_orig[counter].move('orig', 0, -self.canv_speed)
            self.prev_c_den[counter].move('mod', 0, -self.canv_speed)
        elif event.keysym == "Down":
            self.prev_c_orig[counter].move('orig', 0, self.canv_speed)
            self.prev_c_den[counter].move('mod', 0, self.canv_speed)
        elif event.keysym == "Left":
            self.prev_c_orig[counter].move('orig', -self.canv_speed, 0)
            self.prev_c_den[counter].move('mod', -self.canv_speed, 0)
        elif event.keysym == "Right":
            self.prev_c_orig[counter].move('orig', self.canv_speed, 0)
            self.prev_c_den[counter].move('mod', self.canv_speed, 0)
    
    def move_canvas_mouse(self, event, counter):
        if self.move_flag:
            new_xpos, new_ypos = event.x, event.y
             
            self.prev_c_orig[counter].move('orig',new_xpos-self.mouse_xpos ,new_ypos-self.mouse_ypos)
            self.prev_c_den[counter].move('mod',new_xpos-self.mouse_xpos ,new_ypos-self.mouse_ypos)
             
            self.mouse_xpos = new_xpos
            self.mouse_ypos = new_ypos
        else:
            self.move_flag = True
            self.prev_c_orig[counter].tag_raise('orig')
            self.prev_c_den[counter].tag_raise('mod')
            self.mouse_xpos = event.x
            self.mouse_ypos = event.y
 
    def release_canvas_mouse(self, event, counter):
        self.move_flag = False

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

    def local_threshold(self):
        args=self.arggen()
        stfolder,nC,ext,sfn,bs,kv,rv,mp,thrtyp,inv=args[0],int(args[1]),args[2],args[3],int(args[4]),float(args[5]),float(args[6]),int(args[7]),args[8],int(args[9])
        noffiles=count_files_with_ext(ext,stfolder)
        alsteps=2
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
        start = time.time()
        print("Local thresholding computation started on", time.asctime())
        self.loading.prg_status['text']="Alignment started on "+str(time.asctime())
        self.loading.progress2['value']=0
        self.update()
        self.mp_thr_conn(nC,thr_conn,stack,fnames,stfolder,sfn,thrtyp,bs,mp,kv,rv,inv,ext)
        print("Local thresholding computation took", secs)
        print('successfully thresholded all the images')
        self.loading.progress['value']+=1
        self.update()        
        print('Done')
        self.loading.destroy()
        return

    def mp_thr_conn(self,nprocs,func,*args):
        images=args[0]
        fnames=args[1]
        stfolder=args[2]
        sfn=args[3]
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
    
        return

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
    params=('Stack path','Number of Cores (out of '+str(os.cpu_count())+')','Files extension','Subdir name','Block Size (odd number)','k (Sauv NiBl)','r (Sauv)','Min points per region')
    deftexts=('/',str(os.cpu_count()-3),'.tif','LocThr','11','0.02','4','1000')
    fields=[]
    if len(params)==len(deftexts):
        for i in range(len(params)):
            tmp=(params[i],deftexts[i])
            fields.append(tmp)
    root=tk.Tk()
    app=App(root)
    app.mainloop()