






import os
import nibabel as nib
import csv
import copy
import pandas as pd


import random
import numpy as np
random.seed(2)

import csv


from PIL import Image
import torchvision
from skimage.io import imread,imsave




############  process images




def int16_to_int8(img16):

  img16=img16.astype(float)
  #ratio = np.amax(img16) / 256
  img8 = (img16/np.max(img16)*255).astype(np.uint8)

  return img8




def shape_transform(image,shape):
  
  image=Image.fromarray(image)
  
  #print(image.size)
  image=torchvision.transforms.Resize(shape)(image)
  image=np.array(image).astype(np.uint8)
  
  return image



meta_data=pd.read_csv('meta.csv')
print(meta_data)


all_data=[]




for i in meta_data.index:
  
  
  # ---------------- load mri and seg images and check the shapes


  #if meta_data.loc[i,'mri_seg_matched']==0 or meta_data.loc[i,'mriHem_segHem_matched']==0:
  if meta_data.loc[i,'mri_seg_matched']==0 or meta_data.loc[i,'mri_adc_matched']==0:
    print('mri and seg mismatch, this image is not ready to use yet')
    continue


  
  folder_path=meta_data.loc[i,'folder_path']
  folder="data_coregistered/"+folder_path[-5:]
  
  PID=int(folder_path[-5:-1])
  
  #print(PID)
  #aa
  
  '''mriHem_fn=meta_data.loc[i,'mriHem_fn']
  
  if meta_data.loc[i,'has_hem']==1: # only hem images have hem labels, but all images should have swi/ffe version to be used as our 2nd channel
    
    segHem_fn=meta_data.loc[i,'segHem_fn']
    Hem_match_label=meta_data.loc[i,'mriHem_segHem_matched']
  else:
    #mriHem_fn=None
    segHem_fn=None
    Hem_match_label=1'''
  
  mri_fn=meta_data.loc[i,'mri_fn']
  mriAdc_fn=meta_data.loc[i,'mriAdc_fn']
  seg_fn=meta_data.loc[i,'seg_fn']
  match_label=meta_data.loc[i,'mri_seg_matched']
  
  print('now processing image ', mri_fn)
  
  

  
  
  
  img_ori = nib.load(mri_fn)

  
  imgAdc_ori = nib.load(mriAdc_fn)
  seg_ori=nib.load(seg_fn)


  '''coregistered_mriHem_fn=folder+mriHem_fn.split('/')[-1]
  if coregistered_mriHem_fn[:-2]!='gz':
    coregistered_mriHem_fn=coregistered_mriHem_fn+'.gz' # the coregistered swi/ffe are .gz, but the original swi/ffe are not!
  
  if meta_data.loc[i,'has_hem']==1:
    coregistered_segHem_fn=folder+segHem_fn.split('/')[-1]'''
    
    #print(coregistered_segHem_fn)
    
  
  print(img_ori.shape)
  print(seg_ori.shape)
  
  #assert img_ori.shape==seg_ori.shape
  #assert len(img_ori.shape)==3
  
  img=img_ori.get_fdata()
  
  if len(img.shape)==4: # some images have 1 last channel dim 1, now we squeeze them
    img=img[:,:,:,0]
  
  imgAdc=imgAdc_ori.get_fdata()
  seg=seg_ori.get_fdata()
  

  '''coregistered_mriHem_ori=nib.load(coregistered_mriHem_fn)
  coregistered_mriHem=coregistered_mriHem_ori.get_fdata()

  if meta_data.loc[i,'has_hem']==1:
    coregistered_segHem_ori=nib.load(coregistered_segHem_fn)
    coregistered_segHem=coregistered_segHem_ori.get_fdata()
    assert img_ori.shape==coregistered_segHem.shape
  else:
    coregistered_segHem=None

  assert img_ori.shape==coregistered_mriHem_ori.shape'''
  
  # -------------------  reshape and find the label of each slice
  
  
  slice_num=img.shape[2]
  #drop_ratio=0 # drop the marginal slices
  
  #for k in range(int(slice_num*drop_ratio),int(slice_num*(1-drop_ratio))):
  for k in range(1,int(slice_num-1)):
    
    slice_id=k # also record the slice id in each mri image, for future locating
    
    
    slc_pre=img[:,:,k-1]
    slc=img[:,:,k]
    slc_after=img[:,:,k+1]
    
    slcAdc_pre=imgAdc[:,:,k-1]
    slcAdc=imgAdc[:,:,k]
    slcAdc_after=imgAdc[:,:,k+1]
    
    slc_seg=seg[:,:,k]
    
    '''slc_coregistered_mriHem=coregistered_mriHem[:,:,k]
    
    if meta_data.loc[i,'has_hem']==1:
      slc_coregistered_segHem=coregistered_segHem[:,:,k]'''
    
    # -----------------   convert 16 bit to 8 bit for each SLICE, rather than the whole mri image. 
    
    slc_pre=int16_to_int8(img16=slc_pre)
    slc=int16_to_int8(img16=slc)
    slc_after=int16_to_int8(img16=slc_after)
    
    slcAdc_pre=int16_to_int8(img16=slcAdc_pre)
    slcAdc=int16_to_int8(img16=slcAdc)
    slcAdc_after=int16_to_int8(img16=slcAdc_after)
    #slc_coregistered_mriHem=int16_to_int8(img16=slc_coregistered_mriHem)
    # -----------------
    
    #print(slc.shape)
    if slc.shape!=(256,256):
      print(slc.shape,' converting slice shape...')
      
      slc_pre=shape_transform(image=slc_pre, shape=(256,256))
      slc=shape_transform(image=slc, shape=(256,256)) # convert the slice shape to target. IXI is 256*256, now we first use this shape
      slc_after=shape_transform(image=slc_after, shape=(256,256))
      
      slcAdc_pre=shape_transform(image=slcAdc_pre, shape=(256,256))
      slcAdc=shape_transform(image=slcAdc, shape=(256,256))
      slcAdc_after=shape_transform(image=slcAdc_after, shape=(256,256))
      
      #slc_coregistered_mriHem=shape_transform(image=slc_coregistered_mriHem, shape=(256,256))
    #image=Image.fromarray(slc)
    #image.show()
    #aa
    
    #print(slc_seg.shape)
    #print(slc_seg)
    
    stroke_pixel_num=np.sum(slc_seg) # FOR NOW WE DON'T KNOW IF SEG DATA IS PROPER TO RESHAPE!!!!!!!!!!!! though checking the labels won't hurt
    
    '''if meta_data.loc[i,'has_hem']==1:
      hem_pixel_num=np.sum(slc_coregistered_segHem)
    
    
      if stroke_pixel_num==0 and hem_pixel_num==0:
        label=0
      if stroke_pixel_num>0 and hem_pixel_num==0:
        label=1
      if stroke_pixel_num==0 and hem_pixel_num>0:
        label=2
      if stroke_pixel_num>0 and hem_pixel_num>0:
        label=2
    
    else:'''
    if stroke_pixel_num==0: 
      label=0
    else:
      label=1
      
    
    
    print('slice '+str(k)+ ' stroke pixel number: '+ str(stroke_pixel_num) + ', stroke label = '+str(label))
    
    
    '''
    if stroke_pixel_num==0 and meta_data.loc[i,'has_hem']==1: # IMPORTANT!!! we don't want potential non-hem stroke images labeled as 0
      continue

    if stroke_pixel_num==0: 
      label=0
    else:
      if meta_data.loc[i,'has_hem']==1: # for hem, label as 2
        label=2
      else:
        label=1

    print('slice '+str(k)+ ' stroke pixel number: '+ str(stroke_pixel_num) + ', stroke label = '+str(label))
    '''

    #slc_combined=np.stack((slc, slc_coregistered_mriHem), axis=0)
    slc_combined=np.stack((slc, slcAdc), axis=0)
    #slc_combined=np.stack((slc_pre,slc,slc_after, slcAdc_pre,slcAdc,slcAdc_after), axis=0) # now we have 6 channels in total in v9.1, sorted like this
    
    #print(slc_combined.shape)
    #aa
    
    slc_combined=np.array([PID,slice_id, label]+slc_combined.reshape(2*256*256).tolist()).astype(np.uint8)
    #slc_combined=np.array([PID, label]+slc_combined.reshape(6*256*256).tolist()).astype(np.uint8) # the first 0 is a label placeholder, since in our data npy format the first position is label, but now we are doing unsupervised pretraining # in v9.1 we added the PID for patient splitting
    
    #print(slc.tolist())
    #print(slc)
    #print(slc.shape)
    
    all_data.append(slc_combined)
  


# ------------------ save data



random.seed(3)
random.shuffle(all_data)
all_data=np.array(all_data).astype(np.uint8)

print(all_data)
print(all_data.shape)
print(np.sum(all_data[:,0]))
#print(all_data[-1].tolist())
#print(np.max(all_data[-1])) # we transform the whole 3-d mri tensor to uint8, so each slice can have max pixel value much smaller than 255

np.save('stroke_sample_b1000_adc.npy',all_data)






























