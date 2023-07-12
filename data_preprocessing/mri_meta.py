



import os
import nibabel as nib
import csv
import copy



root_path='/net/engnas/Research/cilse_metascan_xinzhang/data/Clean_Data/'



all_folders=sorted(os.listdir(root_path))

print(all_folders)


attr_dic={'folder_path':None,
          'mri_fn':None,
          'mri_shape':None,

          'mriAdc_fn':None,
          'mriAdc_shape':None,

  
          'seg_fn':None,
          'seg_shape':None,
          
          'mriHem_fn':None,
          'mriHem_shape':None,
          
          'segHem_fn':None,
          'segHem_shape':None,      
          
          'has_hem':0,
          'mri_seg_matched':0,
          'mri_adc_matched':0,
          'mriHem_segHem_matched':0,
          
          }


meta_dic={}



for folder in all_folders:
  
  print(folder)
  
  meta_dic[folder]=copy.deepcopy(attr_dic)
  
  meta_dic[folder]['folder_path']=root_path+folder+'/'
  
  all_filenames=os.listdir(meta_dic[folder]['folder_path'])
  
  #print(all_filenames)
  
  for filename in all_filenames:
    if filename.lower().find('stroke')!=-1 and filename.lower().find('.nii')!=-1:
      meta_dic[folder]['seg_fn']=meta_dic[folder]['folder_path']+filename
    
    if filename.lower().find('b1000')!=-1:
      meta_dic[folder]['mri_fn']=meta_dic[folder]['folder_path']+filename


    if filename.lower().find('adc.nii')!=-1:
      meta_dic[folder]['mriAdc_fn']=meta_dic[folder]['folder_path']+filename


    if filename.lower().find('yes_stroke_hemorrhage')!=-1:
      meta_dic[folder]['has_hem']=1



 
  #if meta_dic[folder]['has_hem']==1: # now we coregister all images
    
  for filename in all_filenames:
    if filename.lower().find('hemorrhage.nii')!=-1:
      meta_dic[folder]['segHem_fn']=meta_dic[folder]['folder_path']+filename
    
    if filename.lower().find('swi.nii')!=-1 or filename.lower().find('t2_ffe')!=-1 or filename.lower().find('t2w_ffe')!=-1:
      if filename.lower().find('txt')==-1: # txt: when there is no swi/ffe, they added a 'there is no swi ffe file' -named txt file. we want to avoid load such txt!
        meta_dic[folder]['mriHem_fn']=meta_dic[folder]['folder_path']+filename
    


    
  print(meta_dic[folder]['mri_fn'])
  print(meta_dic[folder]['seg_fn'])
  
  if meta_dic[folder]['mri_fn']!=None:
    mri=nib.load(meta_dic[folder]['mri_fn']).get_fdata()
    
    if len(mri.shape)==4: # some images have 1 last channel dim 1, now we squeeze them
      mri=mri[:,:,:,0]
      
    meta_dic[folder]['mri_shape']=mri.shape


  if meta_dic[folder]['mriAdc_fn']!=None:
    mri=nib.load(meta_dic[folder]['mriAdc_fn']).get_fdata()
    meta_dic[folder]['mriAdc_shape']=mri.shape



  if meta_dic[folder]['seg_fn']!=None:
    seg=nib.load(meta_dic[folder]['seg_fn']).get_fdata()
    meta_dic[folder]['seg_shape']=seg.shape


  if meta_dic[folder]['mriHem_fn']!=None:
    mri=nib.load(meta_dic[folder]['mriHem_fn']).get_fdata()
    meta_dic[folder]['mriHem_shape']=mri.shape


  if meta_dic[folder]['segHem_fn']!=None:
    seg=nib.load(meta_dic[folder]['segHem_fn']).get_fdata()
    meta_dic[folder]['segHem_shape']=seg.shape



  if meta_dic[folder]['mri_shape']==meta_dic[folder]['seg_shape'] and meta_dic[folder]['mri_shape']!=None:
    meta_dic[folder]['mri_seg_matched']=1


  if meta_dic[folder]['mri_shape']==meta_dic[folder]['mriAdc_shape'] and meta_dic[folder]['mriAdc_shape']!=None:
    meta_dic[folder]['mri_adc_matched']=1


  if meta_dic[folder]['has_hem']==0 or meta_dic[folder]['mriHem_shape']==meta_dic[folder]['segHem_shape']:
    if meta_dic[folder]['mriHem_shape']!=None: # added, some images have no valid swi/ffe version
      meta_dic[folder]['mriHem_segHem_matched']=1





out = open('meta.csv', 'a', newline='',encoding='utf-8')
csv_write = csv.writer(out, dialect='excel')

head=list(attr_dic.keys())
csv_write.writerow(head)


for k in meta_dic:
  
  out_line=list(meta_dic[k].values())
  csv_write.writerow(out_line)


out.close()












