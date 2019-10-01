#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 16:11:45 2017

@author: Luca
"""
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
import os
from multiprocessing import Process,Queue,Pool
from collections import Counter
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from scipy import fftpack, ndimage, misc
from skimage import restoration
import cv2
from functools import partial
from PIL import Image, ImageTk
import webbrowser
import platform
import math

mfw,mfh=600,450
nbfw,nbfh=550,40

def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    elif s == 'None':
        return 'None'
    else:
         raise ValueError # evil ValueError that doesn't tell you what the wrong value was

def save_tif(folder,subdir,imname,im,suffix,ext):
    fp=os.path.join(folder,subdir,imname.split('.')[0]+'_'+suffix+ext)
    im.save(fp)
    return

def den_Breg(img,fname,stfolder,sfn,wt,itn,epsv,iso,ext,prev):
    tmp=restoration.denoise_tv_bregman(img,wt,max_iter=itn,eps=epsv,isotropic=int(iso))
    corr=Image.fromarray(np.uint8(np.clip(tmp*255,0,255)),'L')
#    corr=misc.toimage(255*tmp,cmin=0,cmax=255)
    if prev=='N':
        save_tif(stfolder,sfn,fname,corr,'Den_TV_Br',ext)        
    return corr

def den_Ch(img,fname,stfolder,sfn,wt,itn,epsv,p3D,ext,prev):
    tmp=restoration.denoise_tv_chambolle(img,wt,n_iter_max=itn,eps=epsv,multichannel=False)
    if p3D==True and prev=='N':
        corr=[]
        for i in range(len(tmp)):
            tmp2=Image.fromarray(np.uint8(np.clip(tmp[i]*255,0,255)),'L')
            corr.append(tmp2)
    else:
        corr=Image.fromarray(np.uint8(np.clip(tmp*255,0,255)),'L')
    if prev=='N':
        if p3D==False:
            save_tif(stfolder,sfn,fname,corr,'Den_TV_Ch',ext)
        if p3D==True:
            for i in range(len(img)):
                save_tif(stfolder,sfn,fname[i],corr[i],'Den_TV_Ch_3D',ext)
    return corr

def den_BlF(img,fname,stfolder,sfn,ws,sc,ss,binsv,p3D,ext,prev):
    tmp=restoration.denoise_bilateral(img,win_size=ws,sigma_color=sc,sigma_spatial=ss,bins=binsv,mode='constant',cval=0,multichannel=False)
#    tmp=restoration.denoise_bilateral(img,win_size=ws,sigma_color=sc,sigma_spatial=ss,bins=binsv,mode='constant',cval=0)
    if p3D==True and prev=='N':
        corr=[]
        for i in range(len(tmp)):
            tmp2=Image.fromarray(np.uint8(np.clip(tmp[i]*255,0,255)),'L')
            corr.append(tmp2)
    else:
        corr=Image.fromarray(np.uint8(np.clip(tmp*255,0,255)),'L')
    if prev=='N':
        if p3D==False:
            save_tif(stfolder,sfn,fname,corr,'Den_BlF',ext)
        if p3D==True:
            for i in range(len(img)):
                save_tif(stfolder,sfn,fname[i],corr[i],'Den_BlF_3D',ext)
    return corr

def den_nlm(img,fname,stfolder,sfn,ps,pd,hv,p3D,fm,ext,prev):
    tmp=restoration.denoise_nl_means(img,patch_size=ps,patch_distance=pd,h=hv,multichannel=False,fast_mode=fm)
    if p3D==True and prev=='N':
        corr=[]
        for i in range(len(tmp)):
            tmp2=Image.fromarray(np.uint8(np.clip(tmp[i]*255,0,255)),'L')
            corr.append(tmp2)
    else:
        corr=Image.fromarray(np.uint8(np.clip(tmp*255,0,255)),'L')
    if prev=='N':
        if p3D==False:
            save_tif(stfolder,sfn,fname,corr,'Den_nlm',ext)
        if p3D==True:
            for i in range(len(img)):
                save_tif(stfolder,sfn,fname[i],corr[i],'Den_nlm_3D',ext)
    return corr

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

# Each process will get 'chunksize' nums and a queue to put his out
# dict into

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
        self.master.title("Stack Denoise App")
        self.master.resizable(False,False)
        self.master.tk_setPalette(background='#ececec')
        x=int((self.master.winfo_screenwidth()-self.master.winfo_reqwidth())/1.5)
        y=int((self.master.winfo_screenheight()-self.master.winfo_reqheight())/1.5)
        mfw,mfh=x-1,y-1
        nbfw,nbfh=mfw-50,40
        self.master.geometry('{}x{}'.format(x,y))
        
        self.bfiles=tk.Frame(master, width = mfw, height=mfh)
        self.info=tk.Label(self.bfiles,text="Click here access the plugins documentation", fg="blue", cursor="hand2")
        self.info.pack()
        self.info.bind("<Button-1>", self.callback)
        self.fpath=tk.Label(self.bfiles,text=fields[0][0])
        self.fp=tk.StringVar(self.bfiles)
        self.fpath_val=tk.Entry(self.bfiles,textvariable=self.fp)
        self.browse_button = tk.Button(self.bfiles,text="Browse", fg="green",command=self.browseSt)
        self.fpath.pack(side="left")
        self.fpath_val.pack(side="left")
        self.browse_button.pack(side="left")
        self.bfiles.pack(side="top")
        
        self.extfiles=tk.Frame(master, width = mfw, height=mfh)
        self.fext=tk.Label(self.extfiles,text=fields[1][0])
        self.fe=tk.StringVar(self.extfiles,value=fields[1][1])
        self.fext_val=tk.Entry(self.extfiles,textvariable=self.fe)
        self.fext.pack(side="left")
        self.fext_val.pack(side="left")    
        self.sfnlab=tk.Label(self.extfiles,text=fields[2][0])
        self.sfn=tk.StringVar(self.extfiles,value=fields[2][1])
        self.sfn_val=tk.Entry(self.extfiles,textvariable=self.sfn)
        self.sfn_val.pack(side="right")    
        self.sfnlab.pack(side="right")
        self.extfiles.pack(side="top")

        self.nbfr=tk.Frame(master, width = mfw, height=mfh)
        self.nbfr.pack(side="top")
        self.n=ttk.Notebook(self.nbfr)
        self.br_frame=tk.Frame(self.n, width = mfw, height=mfh-40)
        self.ch_frame=tk.Frame(self.n, width = mfw, height=mfh-40)
        self.bl_frame=tk.Frame(self.n, width = mfw, height=mfh-40)
        self.nlm_frame=tk.Frame(self.n, width = mfw, height=mfh-40)
        
        self.n.add(self.br_frame, text='TV Bregman Denoise')
        self.n.add(self.ch_frame, text='TV Chambolle Denoise')
        self.n.add(self.bl_frame, text='Bilateral Filter')  
        self.n.add(self.nlm_frame, text='Non Local Means Filter')  
        self.n.pack()

        # Create the main containers for the Bregman NB
        tk.Label(self.br_frame,text="Denoise image stacks with the TV Bregman algoritm").grid(row=0)
        self.cen_frame_br=tk.Frame(self.br_frame, width = nbfw, height=nbfh*len(fieldsTVB), pady=3)
        self.btm_frame_br=tk.Frame(self.br_frame, width = nbfw, height=nbfh, pady=3)
        
        # Layout the containers for the Bregman NB
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.cen_frame_br.grid(row=1, sticky="ew")
        self.btm_frame_br.grid(row=2, sticky="ew")
        self.entsTVB=self.makeFrame(self.cen_frame_br,fieldsTVB)
        
        # Widgets and layout of the Bregman bottom frame
        self.run_button = tk.Button(self.btm_frame_br,text="Run", fg="Red",command=self.TVBden)
        self.quit_button = tk.Button(self.btm_frame_br, text='Quit', command=self.cancel_b)
        self.prev_button= tk.Button(self.btm_frame_br, text='Preview', command=partial(self.preview_window_TVBr,'TV Bregman',self.entsTVB))

        self.run_button.grid(row=0,column=0)
        self.quit_button.grid(row=0,column=2)
        self.prev_button.grid(row=0,column=3)

        # Create the main containers for the Chambolle NB
        tk.Label(self.ch_frame,text="Denoise image stacks with the TV Chambolle algoritm").grid(row=0)
        self.cen_frame_ch=tk.Frame(self.ch_frame, width = nbfw, height=nbfh*len(fieldsTVCh), pady=3)
        self.btm_frame_ch=tk.Frame(self.ch_frame, width = nbfw, height=nbfh, pady=3)
        
        # Layout the containers for the Chambolle NB
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.cen_frame_ch.grid(row=1, sticky="ew")
        self.btm_frame_ch.grid(row=2, sticky="ew")
        self.entsTVCh=self.makeFrame(self.cen_frame_ch,fieldsTVCh)

        # Widgets and layout of the Chambolle bottom frame
        self.run_button = tk.Button(self.btm_frame_ch,text="Run", fg="Red",command=self.TVChden)
        self.quit_button = tk.Button(self.btm_frame_ch, text='Quit', command=self.cancel_b)
        self.prev_button= tk.Button(self.btm_frame_ch, text='Preview', command=partial(self.preview_window_TVCh,'TV Chambolle',self.entsTVCh))

        self.run_button.grid(row=0,column=0)
        self.quit_button.grid(row=0,column=2)
        self.prev_button.grid(row=0,column=3)

        # Create the main containers for the Bilateral NB
        tk.Label(self.bl_frame,text="Denoise image stacks with the Bilateral filter").grid(row=0)
        self.cen_frame_bl=tk.Frame(self.bl_frame, width = nbfw, height=nbfh*len(fieldsBlF), pady=3)
        self.btm_frame_bl=tk.Frame(self.bl_frame, width = nbfw, height=nbfh, pady=3)
        
        # Layout the containers for the Bilateral NB
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.cen_frame_bl.grid(row=1, sticky="ew")
        self.btm_frame_bl.grid(row=2, sticky="ew")
        self.entsBlF=self.makeFrame(self.cen_frame_bl,fieldsBlF)

        # Widgets and layout of the Bilateral bottom frame
        self.run_button = tk.Button(self.btm_frame_bl,text="Run", fg="Red",command=self.BlFden)
        self.quit_button = tk.Button(self.btm_frame_bl, text='Quit', command=self.cancel_b)
        self.prev_button= tk.Button(self.btm_frame_bl, text='Preview', command=partial(self.preview_window_BlF,'Bilateral Filter',self.entsBlF))

        self.run_button.grid(row=0,column=0)
        self.quit_button.grid(row=0,column=2)
        self.prev_button.grid(row=0,column=3)

        # Create the main containers for the NLM
        tk.Label(self.nlm_frame,text="Denoise image stacks with the Non Local Means algoritm").grid(row=0)
        self.cen_frame_nlm=tk.Frame(self.nlm_frame, width = nbfw, height=nbfh*len(fieldsNLM), pady=3)
        self.btm_frame_nlm=tk.Frame(self.nlm_frame, width = nbfw, height=nbfh, pady=3)
        
        # Layout the containers for the NLM
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.cen_frame_nlm.grid(row=1, sticky="ew")
        self.btm_frame_nlm.grid(row=2, sticky="ew")
        self.entsNLM=self.makeFrame(self.cen_frame_nlm,fieldsNLM)

        # Widgets and layout of the NLM bottom frame
        self.run_button = tk.Button(self.btm_frame_nlm,text="Run", fg="Red",command=self.NLMden)
        self.quit_button = tk.Button(self.btm_frame_nlm, text='Quit', command=self.cancel_b)
        self.prev_button= tk.Button(self.btm_frame_nlm, text='Preview', command=partial(self.preview_window_NLM,'Non Local Means',self.entsNLM))

        self.run_button.grid(row=0,column=0)
        self.quit_button.grid(row=0,column=2)
        self.prev_button.grid(row=0,column=3)

    def callback(self,event):
        webbrowser.open_new(r"http://scikit-image.org/docs/dev/api/skimage.restoration.html")

    def cancel_b(self):
        self.quit()
        self.master.destroy()
        return

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

    #Make the main Transformation Frame  
    def makeFrame(self,parent,fields):
        entries=[]
        for i in range(len(fields)):
           lab = tk.Label(parent, width=25, text=fields[i][0], anchor='w')
           ent_txt=tk.StringVar(parent,value=fields[i][1])
           ent = tk.Entry(parent,textvariable=ent_txt)
           ent.config(justify=tk.RIGHT)
           lab.grid(row=i,column=0)
           ent.grid(row=i,column=1)
           entries.append((fields[i][0], ent))
        return entries

    def make_prev_win(self,counter,imgarray,ltext,txttitle,funct,*args):
        self.prevs[counter] = tk.Toplevel(self,width=imgarray.shape[1],height=imgarray.shape[0]/2)
        self.prevs[counter].wm_title("Window #%s" % counter)
        self.mainframes[counter] = tk.Frame(self.prevs[counter],width=imgarray.shape[1],height=imgarray.shape[0]/2)
        self.mainframes[counter].pack()
        self.prevlab[counter] = tk.Label(self.mainframes[counter], text="Preview for "+txttitle+" denoise")
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

#    def yview(self):
#        sbarV1.config(command=self.prev_c_orig[counter].yview)
#        sbarV1.config(command=self.prev_c_den[counter].yview)
#        return
#    def xview(self, *args):
#        sbarH1.config(command=self.prev_c_orig[counter].xview)
#        sbarH1.config(command=self.prev_c_den[counter].xview)
#        return
#    
    def yview(self,*args):
        self.prev_c_orig[self.wcounter].yview(*args)
        self.prev_c_den[self.wcounter].yview(*args)
        return 
    def xview(self,*args):
        self.prev_c_orig[self.wcounter].xview(*args)
        self.prev_c_den[self.wcounter].xview(*args)
        return

    def preview_window_TVBr(self,text,entries):
        args=self.arggen(entries)
        stfolder,ext,sfn,wt,itn,epsv,iso,NC=args[0],args[1],args[2],float(args[3]),int(args[4]),float(args[5]),str_to_bool(args[6]),int(args[7])
        imgarray=self.load_image_from_folder(stfolder,ext,0)
        self.wcounter += 1
        texttolabel="Parameters: Wt="+str(wt)+" - Itn="+str(itn)+" - eps="+str(epsv)+" - Iso="+str(iso)
        self.make_prev_win(self.wcounter,imgarray,texttolabel,text,den_Breg,imgarray,'',stfolder,sfn,wt,itn,epsv,iso,ext,'Y')
        return

    def preview_window_TVCh(self,text,entries):
        args=self.arggen(entries)
        stfolder,ext,sfn,wt,itn,epsv,p3D,NC=args[0],args[1],args[2],float(args[3]),int(args[4]),float(args[5]),str_to_bool(args[6]),int(args[7])
        imgarray=self.load_image_from_folder(stfolder,ext,0)
        self.wcounter += 1
        texttolabel="Parameters: Wt="+str(wt)+" - Itn="+str(itn)+" - eps="+str(epsv)+" - 3D="+str(p3D)
        self.make_prev_win(self.wcounter,imgarray,texttolabel,text,den_Ch,imgarray,'',stfolder,sfn,wt,itn,epsv,p3D,ext,'Y')
        return

    def preview_window_BlF(self,text,entries):
        args=self.arggen(entries)
        stfolder,ext,sfn,ws,sc,ss,bins,p3D,NC=args[0],args[1],args[2],int(args[3]),float(args[4]),float(args[5]),int(args[6]),str_to_bool(args[7]),int(args[8])
        imgarray=self.load_image_from_folder(stfolder,ext,0)
        self.wcounter += 1
        texttolabel="Parameters: WS="+str(ws)+" - SC="+str(sc)+" - SS="+str(ss)+" - Bins="+str(bins)+" - 3D="+str(p3D)
        self.make_prev_win(self.wcounter,imgarray,texttolabel,text,den_BlF,imgarray,'',stfolder,sfn,ws,sc,ss,bins,p3D,ext,'Y')
        return

    def preview_window_NLM(self,text,entries):
        args=self.arggen(entries)
        stfolder,ext,sfn,ps,pd,h,p3D,fm,NC=args[0],args[1],args[2],int(args[3]),int(args[4]),float(args[5]),str_to_bool(args[6]),str_to_bool(args[7]),int(args[8])
        imgarray=self.load_image_from_folder(stfolder,ext,0)
        vmin, vmax = np.min(imgarray),np.max(imgarray)
        self.wcounter += 1
        texttolabel="Parameters: PS="+str(ps)+" - PD="+str(pd)+" - h="+str(h)+" - 3D="+str(p3D)+" - FM="+str(fm)
        self.make_prev_win(self.wcounter,imgarray,texttolabel,text,den_nlm,imgarray,'',stfolder,sfn,ps,pd,h,p3D,fm,ext,'Y')
        return

    def TVBden(self):
        args=self.arggen(self.entsTVB)
#        global wt,itn,eps,iso
        stfolder,ext,sfn,wt,itn,eps,iso,NC=args[0],args[1],args[2],float(args[3]),int(args[4]),float(args[5]),str_to_bool(args[6]),int(args[7])
        noffiles=self.count_files_with_ext(ext,stfolder)
        steps=2
        self.loading=ProgWin(self.master,steps,noffiles)
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
        print("filtering started on", time.asctime())
        self.loading.prg_status['text']="Filtering started on "+str(time.asctime())
        self.loading.progress2['value']=0
        self.update()
#        self.TVBden_multi_func(stack,wt,itn,eps,iso,NC,stfolder,sfn,fnames,ext)
        self.mp_process(NC,den_Breg,stack,fnames,stfolder,sfn,wt,itn,eps,iso,ext)
        end = time.time()
        secs = end-start
        print("Filternig took", secs)
        print('successfully filtered all the images')
        self.loading.progress['value']+=1
        self.update()        
        print('Done')
        self.loading.destroy()
        return

    def TVChden(self):
        args=self.arggen(self.entsTVCh)
#        global wt,itn,eps,iso
        stfolder,ext,sfn,wt,itn,eps,p3D,NC=args[0],args[1],args[2],float(args[3]),int(args[4]),float(args[5]),str_to_bool(args[6]),int(args[7])
        noffiles=self.count_files_with_ext(ext,stfolder)
        steps=2
        self.loading=ProgWin(self.master,steps,noffiles)
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
        print("filtering started on", time.asctime())
        self.loading.prg_status['text']="Filtering started on "+str(time.asctime())
        self.loading.progress2['value']=0
        self.update()
        if p3D==False:
#            self.TVChden_multi_func(stack,wt,itn,eps,p3D,NC,stfolder,sfn,fnames,ext)
            self.mp_process(NC,den_Ch,stack,fnames,stfolder,sfn,wt,itn,eps,p3D,ext)
        if p3D==True:
            if not sfn in sorted(os.listdir(stfolder)):
                os.makedirs(os.path.join(stfolder,sfn))
                print('directory created')
            den_Ch(stack,fnames,stfolder,sfn,wt,itn,eps,p3D,ext,'N')
        end = time.time()
        secs = end-start
        print("Filternig took", secs)
        print('successfully filtered all the images')
        self.loading.progress['value']+=1
        self.update()        
        print('Done')
        self.loading.destroy()
        return    
    
    def BlFden(self):
        args=self.arggen(self.entsBlF)
#        global wt,itn,eps,iso
        stfolder,ext,sfn,ws,sc,ss,bins,p3D,NC=args[0],args[1],args[2],int(args[3]),float(args[4]),float(args[5]),int(args[6]),str_to_bool(args[7]),int(args[8])
        noffiles=self.count_files_with_ext(ext,stfolder)
        steps=2
        self.loading=ProgWin(self.master,steps,noffiles)
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
        print("filtering started on", time.asctime())
        self.loading.prg_status['text']="Filtering started on "+str(time.asctime())
        self.loading.progress2['value']=0
        self.update()
        if p3D==False:
#            self.BlFden_multi_func(stack,ws,sc,ss,bins,p3D,NC,stfolder,sfn,fnames,ext)
            self.mp_process(NC,den_BlF,stack,fnames,stfolder,sfn,ws,sc,ss,bins,p3D,ext)
        if p3D==True:
            if not sfn in sorted(os.listdir(stfolder)):
                os.makedirs(os.path.join(stfolder,sfn))
                print('directory created')
            den_BlF(stack,fnames,stfolder,sfn,ws,sc,ss,bins,p3D,ext,'N')
        end = time.time()
        secs = end-start
        print("Filternig took", secs)
        print('successfully filtered all the images')
        self.loading.progress['value']+=1
        self.update()        
        print('Done')
        self.loading.destroy()
        return    

    def NLMden(self):
        args=self.arggen(self.entsNLM)
#        global wt,itn,eps,iso
        stfolder,ext,sfn,ps,pd,h,p3D,fm,NC=args[0],args[1],args[2],int(args[3]),int(args[4]),float(args[5]),str_to_bool(args[6]),str_to_bool(args[7]),int(args[8])
        noffiles=self.count_files_with_ext(ext,stfolder)
        steps=2
        self.loading=ProgWin(self.master,steps,noffiles)
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
        print("filtering started on", time.asctime())
        self.loading.prg_status['text']="Filtering started on "+str(time.asctime())
        self.loading.progress2['value']=0
        self.update()
        vmin, vmax = np.min(stack),np.max(stack)
        if vmax > 1:
            stack = np.copy(stack)/255
        if p3D==False:
#            self.NLMden_multi_func(stack,ps,pd,h,p3D,fm,NC,stfolder,sfn,fnames,ext)
            self.mp_process(NC,den_nlm,stack,fnames,stfolder,sfn,ps,pd,h,p3D,fm,ext)
        if p3D==True:
            if not sfn in sorted(os.listdir(stfolder)):
                os.makedirs(os.path.join(stfolder,sfn))
                print('directory created')
            den_nlm(stack,fnames,stfolder,sfn,ps,pd,h,p3D,fm,ext,'N')
        end = time.time()
        secs = end-start
        print("Filternig took", secs)
        print('successfully filtered all the images')
        self.loading.progress['value']+=1
        self.update()        
        print('Done')
        self.loading.destroy()
        return

    def arggen(self,dentype):
        args=[]
        args.append(self.fpath_val.get())
        args.append(self.fext_val.get())
        args.append(self.sfn_val.get())
        for entry in dentype:
           field = entry[0]
           text  = entry[1].get()
           args.append(text)
        return args    

    def load_images_from_folder(self,stfolder,fext):
        images = []
        fnames=[]
        for filename in sorted(os.listdir(stfolder)):
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
        for filename in sorted(os.listdir(stfolder)):
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

    def mp_process(self,nprocs,func,*args):
        if 'den_nlm' in str(func):
            vmax = np.max(args[0])
            if vmax > 1:
                images = np.copy(args[0])/255
            else:
                images=args[0]
        else:
            images=args[0]
        fnames=args[1]
        stfolder=args[2]
        sfn=args[3]
        if not sfn in sorted(os.listdir(stfolder)):
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

    def TVBden_multi_func(self,imstack,wt,itn,eps,iso,nCORES,stfolder,sfn,fnames,ext):
        if not sfn in sorted(os.listdir(stfolder)):
            os.makedirs(os.path.join(stfolder,sfn))
            print('directory created')
        print('running with',nCORES,'processes in parallel')
        pool=Pool(nCORES)
        print('computing denoise with',nCORES,'processes in parallel')
        vmin, vmax = np.min(imstack),np.max(imstack)
        if vmax > 1:
            imstack = np.copy(imstack)/255
        results=[]
        for i in range(len(imstack)):
            results.append(pool.apply_async(den_Breg,(imstack[i],fnames[i],stfolder,sfn,wt,itn,eps,iso,'N',ext,)))
            results[0].get()
            self.loading.progress2['value']+=1
            self.update()
        pool.close()
        pool.join()
        return results

    def TVChden_multi_func(self,imstack,wt,itn,eps,p3D,nCORES,stfolder,sfn,fnames,ext):
        if not sfn in sorted(os.listdir(stfolder)):
            os.makedirs(os.path.join(stfolder,sfn))
            print('directory created')
        print('running with',nCORES,'processes in parallel')
        pool=Pool(nCORES)
        print('computing denoise with',nCORES,'processes in parallel')
        vmin, vmax = np.min(imstack),np.max(imstack)
        if vmax > 1:
            imstack = np.copy(imstack)/255
        results=[]
        for i in range(len(imstack)):
            results.append(pool.apply_async(den_Ch,(imstack[i],fnames[i],stfolder,sfn,wt,itn,eps,p3D,'N',ext,)))
            self.loading.progress2['value']+=1
            self.update()
        pool.close()
        pool.join()
        return results    

    def BlFden_multi_func(self,imstack,ws,sc,ss,bins,p3D,nCORES,stfolder,sfn,fnames,ext):
        if not sfn in sorted(os.listdir(stfolder)):
            os.makedirs(os.path.join(stfolder,sfn))
            print('directory created')
        print('running with',nCORES,'processes in parallel')
        pool=Pool(nCORES)
        print('computing denoise with',nCORES,'processes in parallel')
        vmin, vmax = np.min(imstack),np.max(imstack)
        if vmax > 1:
            imstack = np.copy(imstack)/255
        results=[]
        for i in range(len(imstack)):
            results.append(pool.apply_async(den_BlF,(imstack[i],fnames[i],stfolder,sfn,ws,sc,ss,bins,p3D,'N',ext,)))
            self.loading.progress2['value']+=1
            self.update()
        pool.close()
        pool.join()
        return results 
        
    def NLMden_multi_func(self,imstack,ps,pd,h,p3D,fm,nCORES,stfolder,sfn,fnames,ext):
        if not sfn in sorted(os.listdir(stfolder)):
            os.makedirs(os.path.join(stfolder,sfn))
            print('directory created')
        print('running with',nCORES,'processes in parallel')
        pool=Pool(nCORES)
        print('computing denoise with',nCORES,'processes in parallel')
        results=[]
        for i in range(len(imstack)):
            results.append(pool.apply_async(den_nlm,(imstack[i],fnames[i],stfolder,sfn,ps,pd,h,p3D,fm,'N',ext,)))
            self.loading.progress2['value']+=1
            self.update()
        pool.close()
        pool.join()
        return results

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

if __name__ == '__main__':
    fields=([('Stack path','/'),('Images ext.','.tif'),('Subdir name','Denoise')])
    TVBpar=('Weight','Iterations','eps','Isotropic','Number of Cores (out of '+str(os.cpu_count())+')')
    TVBdef=('7','100000','0.0001','True',str(os.cpu_count()-2))
    TVChpar=('Weight','Iterations','eps','3D (will use 1 Core)','Number of Cores (out of '+str(os.cpu_count())+')')
    TVChdef=('0.1','100000','0.001','False',str(os.cpu_count()-2))
    BlFpar=('Window Size','Sigma Color','Sigma spatial','bins','3D (will use 1 Core)','Number of Cores (out of '+str(os.cpu_count())+')')
    BlFdef=('10','0.1','5','50','False',str(os.cpu_count()-2))
    NLMpar=('Patch Size','Patch Distance','Grayscale Cut-Off','3D (will use 1 Core)','Fast Mode','Number of Cores (out of '+str(os.cpu_count())+')')
    NLMdef=('7','10','0.1','False','True',str(os.cpu_count()-2))
    fieldsTVB=[]
    fieldsTVCh=[]
    fieldsBlF=[]
    fieldsNLM=[]
    if len(TVBpar)==len(TVBdef):
        for i in range(len(TVBpar)):
            tmp=(TVBpar[i],TVBdef[i])
            fieldsTVB.append(tmp)
    if len(TVChpar)==len(TVChdef):
        for i in range(len(TVChpar)):
            tmp=(TVChpar[i],TVChdef[i])
            fieldsTVCh.append(tmp)
    if len(BlFpar)==len(BlFdef):
        for i in range(len(BlFpar)):
            tmp=(BlFpar[i],BlFdef[i])
            fieldsBlF.append(tmp)
    if len(NLMpar)==len(NLMdef):
        for i in range(len(NLMpar)):
            tmp=(NLMpar[i],NLMdef[i])
            fieldsNLM.append(tmp)
    root=tk.Tk()
    app=App(root)
    app.mainloop()