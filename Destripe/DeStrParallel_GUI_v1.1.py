#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 18:54:59 2017

@author: Luca
"""

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
from multiprocessing import Process,Queue,Pool
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
import pywt
import math


def save_tif(folder,subdir,imname,im,ext):
    fp=os.path.join(folder,subdir,imname.split('.')[0]+'_DS'+ext)
    im.save(fp)

def remove_stripe_ft(im,fname,stfolder,sfn,wt,wd,ext,prev):
    H,W=im.shape
    fft2 = fftpack.fft2(im) #fft of img of the stack
    fft2sh = fftpack.fftshift(fft2)#shifted fft of img of the stack
    fft2shcorr=copy.copy(fft2sh)
    # calculation of the corrected FFT modulus of img, comment if using the simpler version of the filter
#    for i in range(W):
#        if abs(W/2 - i) > wd: 
#            for k in range(2*wt+1):
#                if abs(fft2sh[int(H/2-wt/2+k),i]) != 0:
#                    fft2shcorr[int(H/2-wt/2+k),i]=0
    for k in range(int(H/2)-wt,int(H/2)+wt):
        for i in range(W):
            if abs(W/2-i)>=0:
                fft2shcorr[k][i]=fft2sh[k][i]*np.exp(-abs(int(W/2-i))/wd)
#    fft2shcorrArr.append(fft2shcorr)
    ifft2shcorr=fftpack.ifftshift(fft2shcorr)
    ifft2corr = fftpack.ifft2(ifft2shcorr)
    CorrImg=abs(ifft2corr)
#    fft2shArr.append(fft2sh)
    if prev == 'N':
        save_tif(stfolder,sfn,fname,Image.fromarray(np.uint8(np.clip(CorrImg,0,255)),'L'),ext)
#    print(j)
    return CorrImg

#Adapted from
#Stripe and ring artifact removal with combined wavelet — Fourier filtering
#Beat Munch†, Pavel Trtik†, Federica Marone, Marco Stampanoni
#Optics Express, 17(10):8567–8591, 2009.
def remove_stripe_fw(tomo,fname,stfolder, sfn,level, wname, sigma,ext,prev):
    if len(tomo.shape)==2:
        dz=1
        dx, dy = tomo.shape
    else:
        dz, dx, dy = tomo.shape
    print(dx,dy)
    num_slices = dz
    tomofilt=tomo.copy()
    for m in range(num_slices):
        sli = np.zeros((dy, dx), dtype='float32')
        if num_slices==1:
            sli = tomofilt
        else:
            sli = tomofilt[m]

        # Wavelet decomposition.
        cH = []
        cV = []
        cD = []
        for n in range(level):
            sli, (cHt, cVt, cDt) = pywt.dwt2(sli, wname)
            cH.append(cHt)
            cV.append(cVt)
            cD.append(cDt)
#            print('cV and sli shapes at level ',n,cV[n].shape,' ',sli.shape)

        # FFT transform of horizontal frequency bands.
        for n in range(level):
            # FFT
            fcV = fftpack.fftshift(fftpack.fft2(cV[n]))
            my, mx = fcV.shape
#            print('mx, my at level',n,'=',mx,my)

            # Damping of vertical stripes information.
            x_hat = (np.arange(-my, my, 2, dtype='float32') + 1) / 2
            damp = -np.expm1(-np.square(x_hat) / (2 * np.square(sigma)))
            fcV *= np.transpose(np.tile(damp, (mx, 1)))
#            print('fcV shape at level',str(n)+'=',fcV.shape)

            # Inverse FFT.
            cV[n] = np.real(fftpack.ifft2(fftpack.ifftshift(fcV)))

        # Wavelet reconstruction.
        for n in range(level)[::-1]:
            sli = sli[0:cH[n].shape[0], 0:cH[n].shape[1]]
            if m==0 and n==level-1:
                print('slice shape at level',n,'= ',sli.shape)
            sli = pywt.idwt2((sli, (cH[n], cV[n], cD[n])), wname)
#            print('slice shape after wl reconstruction at level',n,'= ',sli.shape)
        if num_slices==1:
            tomofilt = sli
        else:
            tomofilt[m] = sli
        if prev == 'N':
            save_tif(stfolder,sfn,fname,Image.fromarray(np.uint8(np.clip(sli,0,255)),'L'),ext)
    return tomofilt

def pltWL(wn):
    famlist=pywt.wavelist(family=wn, kind='discrete')
    for i in range(len(famlist)):
        wpln=wn+str(int(famlist[0].replace(wn,''))+i)
        wavelet = pywt.Wavelet(wpln)
        phi, psi, x = wavelet.wavefun(level=8)
        print(wpln)
        plt.plot(x,phi)
        plt.plot(x,psi)
        plt.show()

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
    figp=dict()
    prev_c_den=dict()
    imgden=dict()
    figden=dict()
    sbarV=dict()
    sbarH=dict()
    sbarV1=dict()
    sbarH1=dict()
    def __init__(self,master):
        tk.Frame.__init__(self,master)
        self.master.title("Remove curtaining from stack App")
        self.master.resizable(False,False)
        self.master.tk_setPalette(background='#ececec')
        x=int((self.master.winfo_screenwidth()-self.master.winfo_reqwidth())/1.5)
        y=int((self.master.winfo_screenheight()-self.master.winfo_reqheight())/2)
        self.master.geometry('{}x{}'.format(x,y))
        
        #Create the common frame
        self.bfiles=tk.Frame(master, width = mfw, height=mfh*len(cfields))
        self.fpath=tk.Label(self.bfiles,text=fieldsFT[0][0])
        self.fp=tk.StringVar(self.bfiles)
        self.fpath_val=tk.Entry(self.bfiles,textvariable=self.fp)
        self.browse_button = tk.Button(self.bfiles,text="Browse", fg="green",command=self.browseSt)
    
        self.fpath.grid(row=0,column=0)
        self.fpath_val.grid(row=0,column=1)
        self.browse_button.grid(row=0,column=2)
        
        self.ncores=tk.Label(self.bfiles,text=fieldsFT[1][0])
        self.nc=tk.StringVar(self.bfiles,value=fieldsFT[1][1])
        self.nc_val=tk.Entry(self.bfiles,textvariable=self.nc)
        self.ncores.grid(row=1,column=0)
        self.nc_val.grid(row=1,column=1)

        self.fext=tk.Label(self.bfiles,text=fieldsFT[2][0])
        self.fe=tk.StringVar(self.bfiles,value=fieldsFT[2][1])
        self.fext_val=tk.Entry(self.bfiles,textvariable=self.fe)
        self.fext.grid(row=2,column=0)
        self.fext_val.grid(row=2,column=1)


        self.bfiles.pack(side="top")
        
        
        #Create Notebooks
        self.nbfr=tk.Frame(master, width = mfw, height=mfh)
        self.nbfr.pack(side="top")
        self.n=ttk.Notebook(self.nbfr)
        self.FT_frame=tk.Frame(self.n, width = mfw, height=mfh-40)   
        self.WL_frame=tk.Frame(self.n, width = mfw, height=mfh-40)
        self.n.add(self.FT_frame, text='FT')
        self.n.add(self.WL_frame, text='WL')
        self.n.pack()
        
        
        # Create the main containers for the FT destriping notebook
        tk.Label(self.FT_frame,text="FT approach").grid(row=0)
        self.cen_frame_FT=tk.Frame(self.FT_frame, width = nbfw, height=nbfh*len(FTfields[1:]), pady=3)
        self.btm_frame_FT=tk.Frame(self.FT_frame, width = nbfw, height=nbfh, pady=3)
        
        # Layout the containers
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.cen_frame_FT.grid(row=1, sticky="ew")
        self.btm_frame_FT.grid(row=2, sticky="ew")

        self.entsFT=self.makeFrame(self.cen_frame_FT,fieldsFT)
        
        # Widgets of the bottom frame
        self.run_button = tk.Button(self.btm_frame_FT,text="Run", fg="Red",command=partial(self.RemCurt,'FT',self.entsFT))
        self.quit_button = tk.Button(self.btm_frame_FT, text='Quit', command=self.cancel_b)
        self.prev_button= tk.Button(self.btm_frame_FT, text='Preview', command=partial(self.preview_window_FFTfilt,self.entsFT))
        
        self.run_button.grid(row=0,column=0)
        self.quit_button.grid(row=0,column=2)
        self.prev_button.grid(row=0,column=3)

        root.bind('<Return>', (lambda event: self.fetch(self.entsFT))) 

        # Create the main containers for the FT destriping notebook
        tk.Label(self.WL_frame,text="WL approach").grid(row=0)
        self.cen_frame_WL=tk.Frame(self.WL_frame, width = nbfw, height=nbfh*len(wlFTfields[1:]), pady=3)
        self.wlfam_frame=tk.Frame(self.WL_frame, width = nbfw, height=nbfh, pady=3)
        self.btm_frame_WL=tk.Frame(self.WL_frame, width = nbfw, height=nbfh, pady=3)
        
        # Layout the containers
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.cen_frame_WL.grid(row=1, sticky="ew")
        self.wlfam_frame.grid(row=2, sticky="ew")
        self.btm_frame_WL.grid(row=3, sticky="ew")

        self.wlfamlab=tk.Label(self.wlfam_frame,text='Wavelet Family:')
        self.wlfamstr=tk.StringVar(self.wlfam_frame)
        self.wlfamstr.set('coif')
        self.wlfamopt = tk.OptionMenu(self.wlfam_frame, self.wlfamstr, 'coif', 'db', 'sym')
        self.wlfamlab.grid(row=0,column=0)
        self.wlfamopt.grid(row=0,column=1)
        
        self.entsWL=self.makeFrame(self.cen_frame_WL,fieldsWL)
        
        # Widgets of the bottom frame
        self.run_button = tk.Button(self.btm_frame_WL,text="Run", fg="Red",command=partial(self.RemCurt,'WL',self.entsWL))
        self.quit_button = tk.Button(self.btm_frame_WL, text='Quit', command=self.cancel_b)
        self.prev_button= tk.Button(self.btm_frame_WL, text='Preview', command=partial(self.preview_window_WLfilt,self.entsWL))
        
        self.run_button.grid(row=0,column=0)
        self.quit_button.grid(row=0,column=2)
        self.prev_button.grid(row=0,column=3)

        root.bind('<Tab>', (lambda event: self.fetch(self.entsWL))) 

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
            
    def makeTopFrame(self,parent,fieldz):
        entries=[]
        for i in range(1,2):
           lab = tk.Label(parent, width=25, text=fieldz[i][0], anchor='w')
           ent_txt=tk.StringVar(parent,value=fieldz[i][1])
           ent = tk.Entry(parent,textvariable=ent_txt)
           ent.config(justify=tk.RIGHT)
           lab.grid(row=i,column=0)
           ent.grid(row=i,column=1)
           entries.append((fieldz[i][0], ent))
        return entries

    def makeFrame(self,parent,fieldz):
        entries=[]
        entries.append((fieldz[0][0],self.fpath_val))
        entries.append((fieldz[1][0],self.nc_val))
        entries.append((fieldz[2][0],self.fext_val))        
        if fieldz[4][0]=='Level':
            entries.append(('wavfam',self.wlfamstr))
        for i in range(3,len(fieldz)):
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

        
    def RemCurt(self,ftype,entries):
        # Read the images to be aligned
        args=self.arggen(entries)
        if ftype=='FT':
            stfolder,nC,ext,sfn,wt,wd=args[0],int(args[1]),args[2],args[3],int(args[4]),int(args[5])
        if ftype=='WL':
            stfolder,nC,ext,wavfam,sfn,lev,sig,strw=args[0],int(args[1]),args[2],str(args[3]),args[4],int(args[5]),float(args[6]),int(args[7])

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
        if ftype=='FT':
            self.mp_process(nC,remove_stripe_ft,stack,fnames,stfolder,sfn,wt,wd,ext)
        if ftype=='WL':
            imwidth=stack[0].shape[1]
            if wavfam=='coif':
                wnumber=int((imwidth/strw)*0.09)
                if wnumber>17:wnumber=17
                if wnumber==0:wnumber=1
                wavname=wavfam+str(wnumber)
            if wavfam=='db':
                wnumber=int((imwidth/strw)*0.08)
                if wnumber>38:wnumber=30
                if wnumber==0:wnumber=1
                wavname=wavfam+str(wnumber)
            if wavfam=='sym':
                wnumber=int((imwidth/strw)*0.09)
                if wnumber>20:wnumber=20
                if wnumber==0:wnumber=1
                wavname=wavfam+str(wnumber)            
            self.mp_process(nC,remove_stripe_fw,stack,fnames,stfolder,sfn,lev,wavname,sig,ext)
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

    def mp_process(self,nprocs,func,*args):
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
    
        return resultdict

    def make_prev_win(self,counter,imgarray,ltext,txttitle,funct,*args):
        self.canv_speed = int(imgarray.shape[1]/4)
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
        self.prev_c_den[counter]=tk.Canvas(self.mainframes[counter],width=prevsize[1],height=prevsize[0])
        self.prev_c_orig[counter].pack(side='left')
        self.imgp[counter]=Image.fromarray(imgarray)
        self.figp[counter]=ImageTk.PhotoImage(self.imgp[counter])
        self.prev_c_orig[counter].create_image(0,0,image=self.figp[counter],anchor=tk.NW,tags='orig')

        self.prev_c_orig[counter].config(scrollregion=(0,0,imgarray.shape[1],imgarray.shape[0]))
        self.prev_c_den[counter].config(scrollregion=(0,0,imgarray.shape[1],imgarray.shape[0]))
        self.prev_c_den[counter].pack(side='left')

        self.imgden[self.wcounter]=Image.fromarray(funct(*args))
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

    def preview_window_FFTfilt(self,entriesFT):
        args=self.arggen(entriesFT)
        stfolder,nC,ext,sfn,wt,wd=args[0],int(args[1]),args[2],args[3],int(args[4]),int(args[5])
        im=self.load_image_from_folder(stfolder,ext,0)
        self.wcounter += 1
        text='FT Destriping Preview'
        texttolabel="Parameters: Window Thickness="+str(wt)+" - Distance from center="+str(wd)
        self.make_prev_win(self.wcounter,im,texttolabel,text,remove_stripe_ft,im,'',stfolder,sfn,wt,wd,ext,'Y')
        return

    def preview_window_WLfilt(self,entriesWL):
        args=self.arggen(entriesWL)
        stfolder,nC,ext,wavfam,sfn,lev,sig,strw=args[0],int(args[1]),args[2],str(args[3]),args[4],int(args[5]),float(args[6]),int(args[7])
        im=self.load_image_from_folder(stfolder,ext,0)
        imwidth=im.shape[1]
        if wavfam=='coif':
            wnumber=int((imwidth/strw)*0.09)
            if wnumber>17:wnumber=17
            if wnumber==0:wnumber=1
            wavname=wavfam+str(wnumber)
        if wavfam=='db':
            wnumber=int((imwidth/strw)*0.08)
            if wnumber>38:wnumber=30
            if wnumber==0:wnumber=1
            wavname=wavfam+str(wnumber)
        if wavfam=='sym':
            wnumber=int((imwidth/strw)*0.09)
            if wnumber>20:wnumber=20
            if wnumber==0:wnumber=1
            wavname=wavfam+str(wnumber)
        self.wcounter += 1
        text='WL Destriping Preview'
        texttolabel="Parameters: Level="+str(lev)+" - Damping sigma="+str(sig)+" Wavelet: "+wavname
        self.make_prev_win(self.wcounter,im,texttolabel,text,remove_stripe_fw,im,'',stfolder,sfn,lev,wavname,sig,ext,'Y')
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
    cfields=('Stack path','Number of Cores (out of '+str(os.cpu_count())+')','Files extension')
    cdeftexts=('/',str(os.cpu_count()-2),'.tif')
    FTfields=('Subdir name','Window Thickness','Lambda')
    FTdeftxt=('DestrFT',2,6)
    wlFTfields=('Subdir name','Level','Sigma','Stripes width (px)')
    wlFTdeftxt=('DestrWL',10,2,15)
    
    fieldsFT=[]
    fieldsWL=[]
    if len(cfields)==len(cdeftexts) and len(FTfields)==len(FTdeftxt):
        for i in range(len(cfields)):
            tmp=(cfields[i],cdeftexts[i])
            fieldsFT.append(tmp)
        for i in range(len(FTfields)):
            tmp=(FTfields[i],FTdeftxt[i])
            fieldsFT.append(tmp)
    if len(cfields)==len(cdeftexts) and len(wlFTfields)==len(wlFTdeftxt):
        for i in range(len(cfields)):
            tmp=(cfields[i],cdeftexts[i])
            fieldsWL.append(tmp)
        for i in range(len(wlFTfields)):
            tmp=(wlFTfields[i],wlFTdeftxt[i])
            fieldsWL.append(tmp)
    root=tk.Tk()
    app=App(root)
    app.mainloop()