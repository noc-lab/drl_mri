import os
import scipy.io
from skimage.io import imread,imsave
from PIL import Image
import torchvision
import numpy as np


import os



import numpy as np
import random


import pandas as pd
import matplotlib.pyplot as plt
import h5py
from PIL import Image
from scipy.ndimage import zoom
import skimage.transform as T

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import fastmri
from fastmri.data import transforms, mri_data, subsample

import copy



def add_kspace_noise(image,acc = 4,frac_c = 0.08,seed=3):
    
    #4 --- 0.08
    #8 --- 0.04
    
    #adv training comparison
    
    # image: H*W np.array [0-255], a grey-level image
    
    #print(image.max())
    #print(image.shape)
    
    kspace_i=transforms.to_tensor((image).astype(np.complex))
    kspace_i=fastmri.fft2c(kspace_i)

    mask_func = subsample.RandomMaskFunc(center_fractions=[frac_c],accelerations=[acc])
    img_k = kspace_i  # [H,W,2]


    # apply mask in k-space
    img_k_masked, mask_slice = transforms.apply_mask(img_k, mask_func,seed=seed)  # img_k_masked:[H,W,2], mask_slice:[1,W,1]
    mask_slice = mask_slice.squeeze(2).repeat(img_k.shape[0], 1)  # [H,W]

    img_masked = fastmri.ifft2c(img_k_masked)  # [H,W,2]
    img_masked = fastmri.complex_abs(img_masked)  # [H,W]
    
    img_masked[img_masked>255]=255
    #img_masked=(img_masked*(255/img_masked.max())).numpy().astype(np.uint8)
    
    '''print(img_masked.max())
    if img_masked.max()>255:
      bb'''
    
    img_masked=img_masked.numpy().astype(np.uint8)

    #print('img_masked.shape:', img_masked.shape)
    
    return img_masked



def add_KS(input_folder, acc, frac_c, seed):
  
  np.random.seed(seed)
  random.seed(seed)
  
  output_file=input_folder[:-4]+'_KS_'+str(frac_c)+'.npy'
  
  #filenames=sorted(os.listdir(input_folder))
  
  n=0
  final_data=[]
  
  
  lines=np.load(input_folder)
  
  
  
  for i in range(lines.shape[0]):
  #for f in filenames:
    #image=imread(input_folder+f)
    #image=image[:,:,0]
    #label=f[-5]
    
    image=lines[i,3:]
    PID=lines[i,0]
    slice_id=lines[i,1]
    label=lines[i,2]
    image=image.reshape(2,256,256)
    
    #print(image)
    #print(image.shape)
    #print(label)
    
    '''typ=f.split('_')[0]
    if typ=='train':
      typ=10
    else:
      typ=20 # use 10 to indicate training set and 20 the test set'''
    
    
    #print(image)
    #print(image.shape)
    #print(label)
    #print(typ)
    #aa
    
    
    if frac_c!=1:
      rand_stat=random.randint(1,10000)
      perturbed_image=copy.deepcopy(image)
      perturbed_image[0,:,:]=add_kspace_noise(image=image[0,:,:],frac_c = frac_c, acc=acc,seed=rand_stat)
      perturbed_image[1,:,:]=add_kspace_noise(image=image[1,:,:],frac_c = frac_c, acc=acc,seed=rand_stat)
    else:
      perturbed_image=image
    
    
    #print(perturbed_image)
    #print(perturbed_image.shape)
    #image=Image.fromarray(perturbed_image)
    #image.show()
    #aa
    
    perturbed_image=perturbed_image.reshape(2*256*256).tolist()
    #final_data.append([typ,label]+perturbed_image)
    final_data.append([PID,slice_id,label]+perturbed_image)
    
    #perturbed_image=np.stack([perturbed_image]*3, axis=2)
    #print(perturbed_image.shape)
    #imsave(output_folder+f,perturbed_image)
    
    n+=1
    print('now image ',n)
  
  final_data=np.array(final_data)
  final_data=final_data.astype(np.uint8)
  np.save(output_file, final_data)
  
  print(final_data)
  print(final_data.shape)
  






def add_GW(input_folder, sigma, seed):
  
  np.random.seed(seed)
  random.seed(seed)
  
  output_file=input_folder[:-4]+'_GW_'+str(sigma)+'.npy'
  
  #filenames=sorted(os.listdir(input_folder))
  
  n=0
  final_data=[]
  
  
  lines=np.load(input_folder)
  
  lines=lines.astype(float)
  lines[:,3:]=lines[:,3:]/255
  
  for i in range(lines.shape[0]):

    
    image=lines[i,3:]
    PID=lines[i,0]
    slice_id=lines[i,1]
    label=lines[i,2]
    
    image=image.reshape(2,256,256)
    
    
    v=np.random.normal(0, sigma, size=(2,256,256))
    
    if sigma!=0:
      perturbed_image=image+v
    else:
      perturbed_image=image
    
    
    perturbed_image=perturbed_image.reshape(2*256*256).tolist()
    #final_data.append([typ,label]+perturbed_image)
    final_data.append([PID,slice_id,label]+perturbed_image)
    

    
    n+=1
    print('now image ',n)
  

  
  final_data=np.array(final_data)
  
  final_data[:,3:]=np.clip(final_data[:,3:],0,1)
  final_data[:,3:]=final_data[:,3:]*255
  
  final_data=final_data.astype(np.uint8)
  np.save(output_file, final_data)
  
  print(final_data)
  print(final_data.shape)








#add_KS(input_folder='tumor/', acc = 4,frac_c = 0.08, seed=2)



add_KS(input_folder='stroke_sample_b1000_adc.npy', acc = 4,frac_c = 0.08, seed=8)
add_KS(input_folder='stroke_sample_b1000_adc.npy', acc = 8,frac_c = 0.04, seed=9)
add_KS(input_folder='stroke_sample_b1000_adc.npy', acc = 12,frac_c = 0.02, seed=10)
add_KS(input_folder='stroke_sample_b1000_adc.npy', acc = 6,frac_c = 0.06, seed=11)

add_KS(input_folder='stroke_sample_b1000_adc_augment.npy', acc = 4,frac_c = 0.08, seed=12)
add_KS(input_folder='stroke_sample_b1000_adc_augment.npy', acc = 8,frac_c = 0.04, seed=13)
add_KS(input_folder='stroke_sample_b1000_adc_augment.npy', acc = 12,frac_c = 0.02, seed=14)
add_KS(input_folder='stroke_sample_b1000_adc_augment.npy', acc = 6,frac_c = 0.06, seed=15)




'''
add_GW(input_folder='stroke_sample_b1000_adc.npy', sigma=0.02, seed=16)
add_GW(input_folder='stroke_sample_b1000_adc.npy', sigma=0.04, seed=17)
add_GW(input_folder='stroke_sample_b1000_adc.npy', sigma=0.06, seed=18)
add_GW(input_folder='stroke_sample_b1000_adc.npy', sigma=0.08, seed=19)
add_GW(input_folder='stroke_sample_b1000_adc.npy', sigma=0.1, seed=20)'''

'''
add_GW(input_folder='stroke_sample_b1000_adc_augment.npy', sigma=0.02, seed=21)
add_GW(input_folder='stroke_sample_b1000_adc_augment.npy', sigma=0.04, seed=22)
add_GW(input_folder='stroke_sample_b1000_adc_augment.npy', sigma=0.06, seed=23)
add_GW(input_folder='stroke_sample_b1000_adc_augment.npy', sigma=0.08, seed=24)
add_GW(input_folder='stroke_sample_b1000_adc_augment.npy', sigma=0.1, seed=25)'''



'''
add_GW(input_folder='stroke_sample_b1000_adc.npy', sigma=0.01, seed=26)
add_GW(input_folder='stroke_sample_b1000_adc.npy', sigma=0.03, seed=27)
add_GW(input_folder='stroke_sample_b1000_adc.npy', sigma=0.05, seed=28)

add_GW(input_folder='stroke_sample_b1000_adc_augment.npy', sigma=0.01, seed=29)
add_GW(input_folder='stroke_sample_b1000_adc_augment.npy', sigma=0.03, seed=30)
add_GW(input_folder='stroke_sample_b1000_adc_augment.npy', sigma=0.05, seed=31)'''








