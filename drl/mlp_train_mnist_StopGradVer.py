import os
import sys

#os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch
import torch.nn as nn
import numpy as np
#from skimage.io import imread,imsave
import random
seed=3
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

import torchvision
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from typing import Type, Any, Callable, Union, List, Optional

import csv
import copy
import pandas as pd
from scipy.linalg import sqrtm
import pickle

from einops.layers.torch import Rearrange, Reduce

from transformers import ViTFeatureExtractor, ViTModel, ViTConfig

from transformers.models.vit.modeling_vit import ViTEncoder, ViTLayer, ViTEmbeddings
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, SequenceClassifierOutput

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.metrics import average_precision_score

pair = lambda x: x if isinstance(x, tuple) else (x, x)


# v2: use data augmentation for MAe training (flip and rotate), use input channel 1 in vit
# v3: add kspace perturbed images as mae input
# v3.1: compute loss on all patches, not just masked ones
# v4: same as v3.1, except that we use linear layer in patch_embed rather than conv, in order to use dro later

# formal_v1: a formal version to wrap up the current results



our_config={
        'eval_epochs':1,
        'save_epochs':1,#100,
        'training_epoch':100,
        'batchsize':128,
        'eval_batchsize':128,
        'wd':0.01,
        'lr':1e-5,
        'ckpt': None,
        
        'current_model':sys.argv[1],
        
        'is_pretraining':sys.argv[2]=='MAe',
        'is_MAe_visualize':sys.argv[2]=='MAe_visualize',
        'is_training':sys.argv[2]=='ERM',
        'is_predict':sys.argv[2]=='estimate',
        'is_gen_repr_csv':sys.argv[2]=='estimate',
        'is_DRO_training':sys.argv[2]=='DRO',
        'is_adv_training':sys.argv[2]=='AT',
        'is_UAP_generating':sys.argv[2].find('UAPgen')!=-1,#sys.argv[2]=='UAPgen',
        'is_ATT_map_visualization':sys.argv[2]=='ATT_map',
        
        'DRO_coef':float(sys.argv[3]),
        #'DRO_coef':2e-6,
        
        'DRO_lr':float(sys.argv[4]),
        
        'epochs_each_DRO_stage':int(sys.argv[5]),
        'performance_record_file':sys.argv[6],
        
        'DRO_seed':int(float(sys.argv[7])), # this seed is for DRO training minibatches, since we don't want all stages have the same orders
        'noisy_sample_used':int(float(sys.argv[8])),
        'DRO_target_layer':int(float(sys.argv[9])) if sys.argv[9] not in ['P','B'] else sys.argv[9],
        
        'friend_in_DRO':sys.argv[10], # 'AT' or 'PGD' or 'None'
        
        'channel_num':1,
        'layer_num':4,
        'head_num':4,
        
        'training_set_balanced':False,
        'training_set_augmented':True,
        
        #'test_set_balanced':False,
        'PID_split':False,
        
        #'data_folder':'/data2/brhao/mri_project/stroke_sample_process_formal_v1/',
        #'data_folder':'/data2/brhao/mri_project/stroke_sample_process_formal_v1.1/',
        'data_folder':'/data2/brhao/mri_project/stroke_sample_process_formal_v1.2/',
        
        
        }




# ----------------------------------- load and split the images



def load_and_split(filename):
  
  filename_augment=filename[:-4]+'_augment.npy'
  
  split_seed=3
  random.seed(split_seed)
  np.random.seed(split_seed)


  lines=np.load(filename)
  lines_augment=np.load(filename_augment)
  
  np.random.shuffle(lines)
  sample_num=lines.shape[0]

  if our_config['PID_split']==False:
  
    train=lines[:int(sample_num*0.8),:] 
    val=lines[int(sample_num*0.8):int(sample_num*0.9),:]
    test=lines[int(sample_num*0.9):,:]

    
    if our_config['training_set_augmented']==True:
    
      train_id_list=[(train[i,0],train[i,1]) for i in range(train.shape[0])] # for slice split, we record tuple (PID, slice_id) as unique id
      train_augment=np.array([lines_augment[i,:] for i in range(lines_augment.shape[0]) if (lines_augment[i,0],lines_augment[i,1]) in train_id_list]).astype(np.uint8)
      train=np.concatenate((train, train_augment), axis=0)
      np.random.shuffle(train)
    

  else:
    all_PID=np.unique(lines[:,0]) # the 1st column is PID
    all_PID=all_PID.tolist() # for patient split, we directly use PID as unique id
    
    np.random.shuffle(all_PID)

    
    train_PID=all_PID[:int(len(all_PID)*0.8)]
    val_PID=all_PID[int(len(all_PID)*0.8):int(len(all_PID)*0.9)]
    test_PID=all_PID[int(len(all_PID)*0.9):]
  
  
    train=np.array([lines[i,:] for i in range(lines.shape[0]) if lines[i,0] in train_PID])
    val=np.array([lines[i,:] for i in range(lines.shape[0]) if lines[i,0] in val_PID])
    test=np.array([lines[i,:] for i in range(lines.shape[0]) if lines[i,0] in test_PID])
    
    if our_config['training_set_augmented']==True:
      train_augment=np.array([lines_augment[i,:] for i in range(lines_augment.shape[0]) if lines_augment[i,0] in train_PID])
      train=np.concatenate((train, train_augment), axis=0)
      np.random.shuffle(train)
  
  
  
  # ----------------------  for balanced sets

  
  #if our_config['test_set_balanced']==True:
  
  test_pos=[test[i,:] for i in range(test.shape[0]) if test[i,2]==1]
  test_neg=[test[i,:] for i in range(test.shape[0]) if test[i,2]==0]
  test_neg_DownSampled=test_neg[:len(test_pos)]
  test_balance=test_pos+test_neg_DownSampled
  
  test_balance=np.array(test_balance)
  #np.random.shuffle(test) # no need to shuffle the test set

  
  if our_config['training_set_balanced']==True:
  
    train_pos=[train[i,:] for i in range(train.shape[0]) if train[i,2]==1]
    train_neg=[train[i,:] for i in range(train.shape[0]) if train[i,2]==0]
    train_neg_DownSampled=train_neg[:len(train_pos)]
    train=train_pos+train_neg_DownSampled
    
    train=np.array(train)
    np.random.shuffle(train)
    
    #train_balance=train


  # --------------------- cut the pid and slice id

  train=train[:,2:] # 1st and 2nd are PID and slice_id
  val=val[:,2:]
  test=test[:,2:]
  test_balance=test_balance[:,2:]

  #print(train.shape)
  #print(val.shape)
  #print(test.shape)
  

  return train, val, test, test_balance



# -------------------------------------- build the model



import collections.abc


class vision_model(nn.Module):
  
  #def __init__(self,config):
  def __init__(self):
  
    super().__init__()
    
    self.MAe_vit=MaskedAutoencoderViT(img_size=256, patch_size=16, in_chans=our_config['channel_num'],
               embed_dim=256, depth=our_config['layer_num'], num_heads=our_config['head_num'],
               decoder_embed_dim=256, decoder_depth=our_config['layer_num'], decoder_num_heads=our_config['head_num'],
               mlp_ratio=2., norm_layer=nn.LayerNorm, norm_pix_loss=False).cuda()
    

    self.B= nn.Linear(256, 2)


  def forward(
      self,
      features=None,
      labels=None,
      output_attentions=False,
      DRO_layer=None,
  ):


    hidden_states_before_linear=None
    patches_before_linear=None

    
    
    if DRO_layer in ['P','B']:
      DRO_layer=0 # temply set to 0. we will not use this at all anyway
    '''_out=self.vit(features,output_attentions=output_attentions,DRO_layer=DRO_layer)
    vit_outputs = _out.last_hidden_state
    cls_output=vit_outputs[:,0,:]
    
    hidden_states_before_linear=_out.hidden_states
    patches_before_linear=_out.pooler_output'''
    #print(cls_output.shape)
    
    if output_attentions==False:
      emb, mask, ids_restore, patches_before_linear, hidden_states_before_linear=self.MAe_vit.forward_encoder(x=features, mask_ratio=0, DRO_layer=DRO_layer) # for fine-tuning, use no masks anymore
    else:
      emb, mask, ids_restore, all_att_mats=self.MAe_vit.forward_encoder(x=features, mask_ratio=0, return_all_att=True)
    
    #print(patches_before_linear.shape)
    #bb
    cls_output=emb[:,0,:]
    
    
    repr_=cls_output
    
    logits = self.B(repr_)
    #print(repr_.shape)
    
    loss=None
    
    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        
    
    
    #return loss, logits, repr_, hidden_states_before_linear, patches_before_linear
    
    if output_attentions==False:
      return loss, logits, repr_, hidden_states_before_linear, patches_before_linear
    else:
      return loss, logits, repr_, hidden_states_before_linear, patches_before_linear, all_att_mats





# --------------------------------- model training/test

#for m in model.named_parameters():
#  print(m)



def pred_one_dataset_batch(model,dataset,batchsize=our_config['eval_batchsize'],output_repr=False,DRO_layer=None):


  model.eval()
  PRED=[]
  AUC=None
  ACC=None
  REPR=[]
  REPR_inViT=[]
  PATCHES=[]
  
  LOSS=[]
  SCORES=[]

  for r in range(int(len(dataset)/batchsize)+1): # +1: fix the previous bug
    
    #print(r)
    
    eval_index=[i for i in range(len(dataset))]
    ind_slice=eval_index[r*batchsize:(r+1)*batchsize]
    
    if ind_slice==[]:
      continue
    
    X=dataset[ind_slice,1:]
    y=dataset[ind_slice,0]

    #if X==[]: # to deal with the bug when the total sample number is a integer times of the batchsize
      #continue



    X=torch.Tensor(X).to('cuda')
    X=torch.reshape(X,(-1,2,256,256))
    #X=torch.cat([X,X,X],1) # duplicate the channel since seems like this MAe only supports 3 channels now... will explore
    
    X=X[:,0:1,:,:] # this is b1000! never forget to change other positions as well
    
    
    X=X/255
    y=torch.LongTensor(y).to('cuda')


    _, output, repr_, repr_inViT, raw_patches=model(features=X, labels=y, DRO_layer=DRO_layer)
    
    #print(output.shape)
    #pred=output.cpu().detach().numpy().tolist()
    pred=torch.argmax(output, dim=-1).cpu().numpy().tolist()
    #score=output[:,1].cpu().detach().numpy().tolist()
    score=torch.nn.functional.softmax(output,dim=-1)[:,1].cpu().detach().numpy().tolist() # correct AUC!!
    repr_inViT=repr_inViT.cpu().detach().numpy()#.tolist()
    raw_patches=raw_patches.cpu().detach().numpy()#.tolist()



    repr_=repr_.cpu().detach().numpy()#.tolist()
    PRED.extend(pred)
    REPR.extend(repr_)
    REPR_inViT.extend(repr_inViT)
    PATCHES.extend(raw_patches)
    SCORES.extend(score)

    
    loss=_.cpu().detach().numpy()
    #print(loss) # here loss is a number
    #print(loss.shape)
    
    LOSS.append(loss*len(ind_slice))
  LOSS=np.sum(LOSS)/len(dataset)
  

  GT=dataset[:,0].astype(np.uint8).tolist()
  
  
  #AUC=roc_auc_score(GT,PRED) # THIS IS VERY WRONG!!!!
  AUC=roc_auc_score(GT,SCORES)
  AUPRC=average_precision_score(GT,SCORES)
  
  
  cm=confusion_matrix(GT, PRED)
  #print(cm)
  
  '''
  print('false alarm rate: ', cm[0,1]/(cm[0,1]+cm[0,0]))
  
  precision, recall, w_f1, _ = precision_recall_fscore_support(GT, PRED, average='weighted')
  precision, recall, f1, _ = precision_recall_fscore_support(GT, PRED)
  print('weighted f1: ',w_f1)
  print('precision',precision)
  print('recall',recall)'''
  
  #print('AUPRC', AUPRC)
  
  #print(GT[:50])
  #print(PRED[:50])
  
  assert(len(PRED)==len(GT))
  
  ACC=np.mean([GT[i]==PRED[i] for i in range(len(GT))])
  #MAE=np.mean([abs(PRED[k]-GT[k]) for k in range(len(GT))])
  #RMSE=np.mean((np.array(GT)-np.array(PRED))**2,axis=0)**0.5


  '''
  # compute PPV like Dima's group
  prodict_prob_list=SCORES
  labels_list=GT
  top_ind=sorted(range(len(prodict_prob_list)), key=lambda k: prodict_prob_list[k], reverse=True)
  #top_ind=top_ind[:int(0.1*len(labels_list))] # set the top ratio
  top_ind=top_ind[:int(np.sum(labels_list))] # using the top (# gt positive sample) confident predictions to compute PPV
  top_pred_gt_labels=[labels_list[i] for i in top_ind]
  #print(top_pred_gt_labels)
  ppv=np.sum(top_pred_gt_labels)/len(top_pred_gt_labels)
  print('PPV: ', ppv)'''

  metrics={'AUC':AUC, 'ACC':ACC, 'AUPRC':AUPRC, 'cm':cm}


  if not output_repr:
    return PRED, metrics
  else:
    return PRED, metrics, REPR, REPR_inViT, PATCHES




from torch.autograd import Variable

def LinfPGDAttack(model,features,labels,epsilon=0.2,k=5,a=0.1):
  
  features_nat=copy.deepcopy(features)
  
  for i in range(k):
  
    features_=Variable(copy.deepcopy(features),requires_grad=True)
    labels_=copy.deepcopy(labels)
    
    #loss, output, repr_=model(features=features_, labels=labels_)
    loss, output, repr_, repr_inViT, raw_patches = model(features=features_, labels=labels_, DRO_layer=our_config['DRO_target_layer'])
    
    #print(features_.grad)
    loss.backward()
    gradient=features_.grad
    #print(features_.grad)
    #print(features_.grad.shape)
    
    features_tp=features+a*torch.sign(gradient)
    features_tp=torch.clamp(features_tp,min=features_nat-epsilon,max=features_nat+epsilon)
    #features_tp=torch.clamp(features_tp,min=-0.5,max=0.5)
    features_tp=torch.clamp(features_tp,min=0,max=1)
    
    features=features_tp
    
    features_.grad.zero_()
    
  return features




######################### function for adding kspace noise


import fastmri
from fastmri.data import transforms, mri_data, subsample



def add_kspace_noise(image,acc = 4,frac_c = 0.08,seed=3):
    
    #4 --- 0.08
    #8 --- 0.04
    
    #adv training comparison
    
    # image: H*W np.array [0-255], a grey-level image
    
    kspace_i=transforms.to_tensor((image).astype(np.complex))
    kspace_i=fastmri.fft2c(kspace_i)

    mask_func = subsample.RandomMaskFunc(center_fractions=[frac_c],accelerations=[acc])
    img_k = kspace_i  # [H,W,2]


    # apply mask in k-space
    img_k_masked, mask_slice = transforms.apply_mask(img_k, mask_func,seed=seed)  # img_k_masked:[H,W,2], mask_slice:[1,W,1]
    mask_slice = mask_slice.squeeze(2).repeat(img_k.shape[0], 1)  # [H,W]

    img_masked = fastmri.ifft2c(img_k_masked)  # [H,W,2]
    img_masked = fastmri.complex_abs(img_masked)  # [H,W]
    #img_masked=(img_masked*(255/img_masked.max())).numpy().astype(np.uint8)
    img_masked=img_masked.numpy().astype(np.uint8)
    #print(img_masked.max())
    #print('img_masked.shape:', img_masked.shape)
    
    return img_masked



################### MAe pretraining

import sys
sys.path.append(os.getcwd()+'/masked_autoencoder/')
from masked_autoencoder.models_mae import MaskedAutoencoderViT


################### MAe visualization


if our_config['is_MAe_visualize']==True:

  import matplotlib.pyplot as plt
  
  
  
  def show_image(image, title=''):
      # image is [H, W, 3]
      #assert image.shape[2] == 3
      print(image.shape)
      if image.shape[2] != 3:
        image=torch.cat([image,image,image],2) # or it seems the image display will be blue
      #plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
      plt.imshow(torch.clip(image* 255, 0, 255).int())
      plt.title(title, fontsize=16)
      plt.axis('off')
      return
  
  
  
  def run_one_image(img, model, target):
      
      x=img
      #x = torch.tensor(img)
  
      # make it a batch-like
      #x = x.unsqueeze(dim=0)
      #x = torch.einsum('nhwc->nchw', x)
  
      # run MAE
      loss, y, mask = model(x.float(), mask_ratio=0.75, targets=target)
      
      #emb, mask_, ids_restore_ = model.forward_encoder(x.float(), mask_ratio=0) # check the unmasked encoder output
      #print(emb.shape)
      
      y = model.unpatchify(y)
      y = torch.einsum('nchw->nhwc', y).detach().cpu()
  
      # visualize the mask
      mask = mask.detach()
      #mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
      mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *1)
      mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
      mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
      
      x = torch.einsum('nchw->nhwc', x)
      target = torch.einsum('nchw->nhwc', target)
  
      x=x.cpu()
      target=target.cpu()
      #print(x)
      #print(mask)
      
      # masked image
      im_masked = x * (1 - mask)
  
      # MAE reconstruction pasted with visible patches
      #im_paste = target * (1 - mask) + y * mask
      im_paste = x * (1 - mask) + y * mask
  
      # make the plt figure larger
      plt.rcParams['figure.figsize'] = [24, 24]
  
      plt.subplot(1, 5, 1)
      show_image(x[0], "original")
  
      plt.subplot(1, 5, 2)
      show_image(im_masked[0], "masked")
  


      plt.subplot(1, 5, 3)
      show_image(target[0], "target")


      plt.subplot(1, 5, 4)
      show_image(y[0], "reconstruction")


      plt.subplot(1, 5, 5)
      show_image(im_paste[0], "reconstruction + visible")
  
      plt.show()




  train,val,test=load_and_split(filename="/projectnb/noc-lab/boranhao/mri_project/ixi_t1t2_uint8.npy")
  #train,val,test=load_and_split(filename="/projectnb/noc-lab/boranhao/mri_project/tumor_KS_1.npy")
  
  
  print(train.shape)
  print(val.shape)
  print(test.shape)


  model=MaskedAutoencoderViT(img_size=256, patch_size=16, in_chans=our_config['channel_num'],
                 embed_dim=256, depth=our_config['layer_num'], num_heads=our_config['head_num'],
                 decoder_embed_dim=256, decoder_depth=our_config['layer_num'], decoder_num_heads=our_config['head_num'],
                 mlp_ratio=2., norm_layer=nn.LayerNorm, norm_pix_loss=False).cuda()
  
  model.load_state_dict(torch.load('MAe_model_ep500.pt'))
  model.eval()

  
  #X=train[999,1:]
  X=test[299,1:]

  X=torch.Tensor(X).to('cuda')
  X=torch.reshape(X,(-1,1,256,256))
  #X=torch.cat([X,X,X],1)
  
  
  X_pert=torch.Tensor(add_kspace_noise(X[0,0,:,:].cpu().numpy(),acc = 4,frac_c = 0.08,seed=random.randint(1,10000))).to('cuda')
  X_pert=torch.reshape(X_pert,(-1,1,256,256))
  
  X=X/255
  X_pert=X_pert/255

  #run_one_image(img=X, model=model, target=X)
  run_one_image(img=X_pert, model=model, target=X)







################### ERM training


if our_config['is_training']==True:

  #train,val,test=load_and_split(filename='mnist_GW_0.npy')
  #train,val,test=load_and_split(filename="/projectnb/noc-lab/boranhao/mri_project/tumor_KS_1.npy")
  #train,val,test=load_and_split(filename="/data2/brhao/mri_project/stroke_sample_process_2/stroke_sample_b1000_uint8.npy")
  #train,val,test=load_and_split(filename="/data2/brhao/mri_project/stroke_sample_process_3/stroke_sample_b1000_uint8.npy")
  train,val,test,test_balance=load_and_split(filename=our_config['data_folder']+"stroke_sample_b1000_adc.npy")
  #train_,val_,test=load_and_split(filename="/projectnb/noc-lab/boranhao/mri_project/stroke_sample_process/stroke_sample_b1000_uint8_KS_0.04.npy") # use this to simply test a noisy test set
  
  print(train.shape)
  print(val.shape)
  print(test.shape)
  print(test_balance.shape)


  
  model=vision_model().cuda()
  if our_config['current_model']!='None':
    print('MAe pretrained model loaded')
    model.MAe_vit.load_state_dict(torch.load(our_config['current_model']))
  
  
  #optimizer = torch.optim.AdamW(model.parameters(), lr=our_config['lr'], weight_decay=our_config['wd'])
  #optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0)
  optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
  #optimizer = torch.optim.AdamW(model.parameters(), lr=our_config['lr'], weight_decay=0) # we don't use weight decay in this version
  #optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=our_config['lr'], weight_decay=our_config['wd'])
  
  batchsize=our_config['batchsize']
  #batchsize=128
  all_index=[i for i in range(len(train))]
  random.seed(seed)
  
  
  for e in range(1,our_config['training_epoch']+1): 
      
      if our_config['is_training']==False:
          break
      
      # training for each epoch -----------------------------------
      
      model.train()
      
      random.shuffle(all_index)
      for r in range(int(len(train)/batchsize)): # no +1: in training, make sure the batchsize is stable
          
          ind_slice=all_index[r*batchsize:(r+1)*batchsize]
          #X=[train[i,1:] for i in ind_slice]
          X=train[ind_slice,1:]
          #y=[train[i,0] for i in ind_slice]
          y=train[ind_slice,0]

  

          X=torch.Tensor(X).to('cuda')
          X=torch.reshape(X,(-1,2,256,256))
          #print(X.shape)
          #X=torch.cat([X,X,X],1) # duplicate the channel since seems like this MAe only supports 3 channels now... will explore
          
          X=X[:,0:1,:,:] # this is b1000! never forget to change other positions as well
          
          X=X/255
          y=torch.LongTensor(y).to('cuda')

      
          optimizer.zero_grad()
          loss, output, repr_, repr_inViT, raw_patches = model(features=X, labels=y, DRO_layer=our_config['DRO_target_layer'])
          #print(output.shape)
          l_numerical = loss.item()
          
          loss.backward()
          optimizer.step()
  
      print(f"Epoch: {e}, Loss: {l_numerical}")
      
      
      if e%our_config['eval_epochs']==0:
          #continue
        
        PRED, metrics, REPR, REPR_inViT, PATCHES=pred_one_dataset_batch(model,dataset=val,output_repr=True,DRO_layer=our_config['DRO_target_layer'])
        print(metrics['cm'])
        print('ep'+str(e)+' val AUC: ',metrics['AUC'])
        print('ep'+str(e)+' val ACC: ',metrics['ACC'])
        print('ep'+str(e)+' val AUPRC: ',metrics['AUPRC'])
        

        out_line=sys.argv+[e,l_numerical,metrics['AUC'],metrics['ACC'],metrics['AUPRC']]
        

        PRED, metrics, REPR, REPR_inViT, PATCHES=pred_one_dataset_batch(model,dataset=test,output_repr=True,DRO_layer=our_config['DRO_target_layer'])
        print(metrics['cm'])
        print('ep'+str(e)+' test AUC: ',metrics['AUC'])
        print('ep'+str(e)+' test ACC: ',metrics['ACC'])
        print('ep'+str(e)+' test AUPRC: ',metrics['AUPRC'])
      
        
        out_line=out_line+[metrics['AUC'],metrics['ACC'],metrics['AUPRC']]



        # for balanced test set performance recording
        PRED, metrics, REPR, REPR_inViT, PATCHES=pred_one_dataset_batch(model,dataset=test_balance,output_repr=True,DRO_layer=our_config['DRO_target_layer'])

        out_line=out_line+[metrics['AUC'],metrics['ACC'],metrics['AUPRC']]



        out = open(our_config['performance_record_file'], 'a', newline='',encoding='utf-8')
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow(out_line)
        out.close()
        
        
      if e%our_config['save_epochs']==0:
        torch.save(model.state_dict(), 'ERM_model_ep'+str(e)+'.pt')




def write_repr_csv(filename,repr_vectors,labels,dim=2):
  
  out = open(filename, 'a', newline='',encoding='utf-8')
  csv_write = csv.writer(out, dialect='excel')

  assert(len(repr_vectors)==len(labels))
	
  for i in range(len(labels)):
    
    if dim==2:
      csv_write.writerow([labels[i]]+repr_vectors[i].tolist())
    elif dim==3:
      flattened_vectors=[]
      for vec in repr_vectors[i].tolist():
        flattened_vectors.extend(vec)
      csv_write.writerow([labels[i]]+flattened_vectors)
    else:
      print('not implemented')
      aa
	
  out.close()



############# filename list



filename_list=[our_config['data_folder']+"stroke_sample_b1000_adc.npy",
                our_config['data_folder']+"stroke_sample_b1000_adc_KS_0.08.npy",
                our_config['data_folder']+"stroke_sample_b1000_adc_KS_0.06.npy",
                our_config['data_folder']+"stroke_sample_b1000_adc_KS_0.04.npy",
                our_config['data_folder']+"stroke_sample_b1000_adc_KS_0.02.npy",
                
                ]



'''
filename_list=[our_config['data_folder']+"stroke_sample_b1000_adc.npy",
                our_config['data_folder']+"stroke_sample_b1000_adc_GW_0.02.npy",
                our_config['data_folder']+"stroke_sample_b1000_adc_GW_0.03.npy",
                our_config['data_folder']+"stroke_sample_b1000_adc_GW_0.04.npy",
                our_config['data_folder']+"stroke_sample_b1000_adc_GW_0.05.npy",
                our_config['data_folder']+"stroke_sample_b1000_adc_GW_0.06.npy",


                ]'''



'''
filename_list=[our_config['data_folder']+"stroke_sample_b1000_adc.npy",
                our_config['data_folder']+"stroke_sample_b1000_adc_KS_0.08_GW_0.02.npy",
                our_config['data_folder']+"stroke_sample_b1000_adc_KS_0.06_GW_0.03.npy",
                our_config['data_folder']+"stroke_sample_b1000_adc_KS_0.04_GW_0.04.npy",
                our_config['data_folder']+"stroke_sample_b1000_adc_KS_0.02_GW_0.05.npy",


                ]'''




#filename_for_repr="/data2/brhao/mri_project/stroke_sample_process_4/stroke_sample_b1000_uint8_KS_0.08.npy"
#filename_for_repr="/data2/brhao/mri_project/stroke_sample_process_4/stroke_sample_b1000_uint8_GW_0.06.npy"

filename_for_repr=our_config['data_folder']+"stroke_sample_b1000_adc_KS_0.08.npy"
#filename_for_repr=our_config['data_folder']+"stroke_sample_b1000_adc_GW_0.02.npy"
#filename_for_repr=our_config['data_folder']+"stroke_sample_b1000_adc_KS_0.08_GW_0.02.npy"




if our_config['is_predict']==True:


  model=vision_model().cuda()
  model.load_state_dict(torch.load(our_config['current_model']))



  model.eval()


  #if our_config['is_gen_repr_csv']==False: # for now if the purpose is to estimate W, don't waste time to predict test set
  
  for filename in filename_list:
    
    print('Now testing the test set in '+filename)
    noisy_train,noisy_val,noisy_test,noisy_test_balance=load_and_split(filename=filename)
    PRED, metrics, REPR, REPR_inViT, PATCHES=pred_one_dataset_batch(model,dataset=noisy_test,output_repr=True, DRO_layer=our_config['DRO_target_layer'])
    print(metrics['cm'])
    print('test AUC: ',metrics['AUC'])
    print('test ACC: ',metrics['ACC'])
    print('test AUPRC: ',metrics['AUPRC'])
    
    
    
    
    print('***********************************')
    print('Now testing the BALANCED test set in '+filename)
    print('***********************************')
    
    PRED, metrics, REPR, REPR_inViT, PATCHES=pred_one_dataset_batch(model,dataset=noisy_test_balance,output_repr=True, DRO_layer=our_config['DRO_target_layer'])
    print(metrics['cm'])
    print('test AUC: ',metrics['AUC'])
    print('test ACC: ',metrics['ACC'])
    print('test AUPRC: ',metrics['AUPRC'])

  
  
  
  
  

  
  print('Now testing the val set in '+our_config['data_folder']+"stroke_sample_b1000_adc.npy")
  train,val,test,test_balance=load_and_split(filename=our_config['data_folder']+"stroke_sample_b1000_adc.npy")
  #PRED, AUC, ACC, REPR=pred_one_dataset_batch(model,dataset=val,output_repr=True)
  PRED, metrics, REPR, REPR_inViT, PATCHES=pred_one_dataset_batch(model,dataset=train[:our_config['noisy_sample_used'],:],output_repr=True, DRO_layer=our_config['DRO_target_layer']) # use training set in v2
  print(metrics['cm'])
  print('val AUC: ',metrics['AUC'])
  print('val ACC: ',metrics['ACC'])
  print('val AUPRC: ',metrics['AUPRC'])
  


  if our_config['is_gen_repr_csv']==True:
    if our_config['DRO_target_layer']=='P':
      pickle.dump({'reprs':np.array(PATCHES), 'labels':train[:our_config['noisy_sample_used'],0].tolist()},open('currentrepr_mnist_clean.pkl','wb'))
    elif our_config['DRO_target_layer']=='B':
      pickle.dump({'reprs':np.array(REPR), 'labels':train[:our_config['noisy_sample_used'],0].tolist()},open('currentrepr_mnist_clean.pkl','wb'))
    else:
      pickle.dump({'reprs':np.array(REPR_inViT), 'labels':train[:our_config['noisy_sample_used'],0].tolist()},open('currentrepr_mnist_clean.pkl','wb'))



  
  print('Now testing the val set in '+filename_for_repr)
  noisy_train,noisy_val,noisy_test,noisy_test_balance=load_and_split(filename=filename_for_repr)  
  #PRED, AUC, ACC, REPR_noisy=pred_one_dataset_batch(model,dataset=noisy_val,output_repr=True)
  PRED, metrics, REPR_noisy, REPR_inViT_noisy, PATCHES_noisy=pred_one_dataset_batch(model,dataset=noisy_train[:our_config['noisy_sample_used'],:],output_repr=True, DRO_layer=our_config['DRO_target_layer']) # use training set in v2
  print(metrics['cm'])
  print('val AUC: ',metrics['AUC'])
  print('val ACC: ',metrics['ACC'])
  print('val AUPRC: ',metrics['AUPRC'])




  if our_config['is_gen_repr_csv']==True:
    if our_config['DRO_target_layer']=='P':
      pickle.dump({'reprs':np.array(PATCHES_noisy), 'labels':noisy_train[:our_config['noisy_sample_used'],0].tolist()},open('currentrepr_mnist.pkl','wb'))
    elif our_config['DRO_target_layer']=='B':
      pickle.dump({'reprs':np.array(REPR_noisy), 'labels':noisy_train[:our_config['noisy_sample_used'],0].tolist()},open('currentrepr_mnist.pkl','wb'))
    else:
      pickle.dump({'reprs':np.array(REPR_inViT_noisy), 'labels':noisy_train[:our_config['noisy_sample_used'],0].tolist()},open('currentrepr_mnist.pkl','wb'))





  if our_config['is_gen_repr_csv']==True:
    if our_config['DRO_target_layer']=='P':
      pickle.dump({'reprs':np.array(PATCHES_noisy)-np.array(PATCHES), 'labels':train[:our_config['noisy_sample_used'],0].tolist()},open('currentreprDiff_mnist.pkl','wb'))
    elif our_config['DRO_target_layer']=='B':
      pickle.dump({'reprs':np.array(REPR_noisy)-np.array(REPR), 'labels':train[:our_config['noisy_sample_used'],0].tolist()},open('currentreprDiff_mnist.pkl','wb'))
    else:
      pickle.dump({'reprs':np.array(REPR_inViT_noisy)-np.array(REPR_inViT), 'labels':train[:our_config['noisy_sample_used'],0].tolist()},open('currentreprDiff_mnist.pkl','wb'))








# ---------------- DRO training


class DRO(nn.Module):
  
  #def __init__(self,config):
  def __init__(self):
  
    super().__init__()
    
    self.MAe_vit=MaskedAutoencoderViT(img_size=256, patch_size=16, in_chans=our_config['channel_num'],
               embed_dim=256, depth=our_config['layer_num'], num_heads=our_config['head_num'],
               decoder_embed_dim=256, decoder_depth=our_config['layer_num'], decoder_num_heads=our_config['head_num'],
               mlp_ratio=2., norm_layer=nn.LayerNorm, norm_pix_loss=False).cuda()
    

    #self.B= nn.Linear(config.hidden_size, 4)
    self.B= nn.Linear(256, 2)


  def forward(
      self,
      features=None,
      labels=None,
      output_attentions=False,
      DRO_layer=None,
      W_minus_half=None,
      DRO_coef=None,
  ):
  

    '''
    _out=self.vit(features,output_attentions=output_attentions,DRO_layer=DRO_layer)
    vit_outputs = _out.last_hidden_state
    cls_output=vit_outputs[:,0,:]
    
    hidden_states_before_linear=_out.hidden_states
    patches_before_linear=_out.pooler_output'''
    #print(cls_output.shape)
    
    
    emb, mask, ids_restore, patches_before_linear, hidden_states_before_linear=self.MAe_vit.forward_encoder(x=features, mask_ratio=0) # for fine-tuning, use no masks anymore
    
    #repr_=cls_output
    
    cls_output=emb[:,0,:]
    repr_=cls_output
    
    output = self.B(repr_)

    #print(repr_.shape)
    loss=None
    
    if labels is not None:

      loss_fct = nn.CrossEntropyLoss()
      #loss_fct = nn.MSELoss()


      if our_config['DRO_target_layer']=='P':
        #for W in self.vit.embeddings.patch_embeddings.our_projection.named_parameters():
        for W in self.MAe_vit.patch_embed.our_proj.named_parameters():  # use the P layer in MAe
          if "weight" in W[0]:
            U,S,Vh=torch.linalg.svd(torch.matmul(W_minus_half,W[1].T), full_matrices=False)
            #U,S,Vh=torch.linalg.svd(W[1], full_matrices=False)      # now in mri, it seems that the previous metric learning is not a good idea. maybe it's due to the correlation between mri slices?
            r = torch.max(S)

      elif our_config['DRO_target_layer']=='B':
        for W in self.B.named_parameters():
          if "weight" in W[0]:
            U,S,Vh=torch.linalg.svd(torch.matmul(W_minus_half,W[1].T), full_matrices=False)     
            r = torch.max(S)
      
      else:
        '''for W in self.vit.encoder.layer[our_config['DRO_target_layer']].output.dense.named_parameters():
          if "weight" in W[0]:
            U,S,Vh=torch.linalg.svd(torch.matmul(W_minus_half,W[1].T), full_matrices=False)       
            r = torch.max(S)'''

        '''for W in self.vit.encoder.layer[our_config['DRO_target_layer']].attention.attention.query.named_parameters():
          if "weight" in W[0]:
            U,S,Vh=torch.linalg.svd(torch.matmul(W_minus_half,W[1].T), full_matrices=False)       
            r1 = torch.max(S)

        for W in self.vit.encoder.layer[our_config['DRO_target_layer']].attention.attention.key.named_parameters():
          if "weight" in W[0]:
            U,S,Vh=torch.linalg.svd(torch.matmul(W_minus_half,W[1].T), full_matrices=False)       
            r2 = torch.max(S)

        for W in self.vit.encoder.layer[our_config['DRO_target_layer']].attention.attention.value.named_parameters():
          if "weight" in W[0]:
            U,S,Vh=torch.linalg.svd(torch.matmul(W_minus_half,W[1].T), full_matrices=False)       
            r3 = torch.max(S)

   
        r=r1+r2+r3'''

        
        '''for W in self.MAe_vit.blocks[our_config['DRO_target_layer']].attn.qkv.named_parameters():
          if "weight" in W[0]:
            U,S,Vh=torch.linalg.svd(torch.matmul(W_minus_half,W[1].T), full_matrices=False)       
            r = torch.max(S)'''


        for W in self.MAe_vit.blocks[our_config['DRO_target_layer']].attn.proj.named_parameters():
          if "weight" in W[0]:
            U,S,Vh=torch.linalg.svd(torch.matmul(W_minus_half,W[1].T), full_matrices=False)       
            r = torch.max(S)

        
        
      loss = loss_fct(output.view(-1, 2), labels.view(-1))+DRO_coef*r


    return loss, output


import scipy.io


if our_config['is_DRO_training']==True:


  train,val,test,test_balance=load_and_split(filename=our_config['data_folder']+"stroke_sample_b1000_adc.npy")
  
  print(train.shape)
  print(val.shape)
  print(test.shape)
  print(test_balance.shape)


  if our_config['friend_in_DRO']=='AT':
    noisy_train,noisy_val,noisy_test,noisy_test_balance=load_and_split(filename=filename_for_repr)
    #train=train+noisy_val[:our_config['noisy_sample_used']] # use clean training set and perturbed val set to do AT+DRO
    #train=train+noisy_train[:our_config['noisy_sample_used']] # use clean training set and perturbed val set to do AT+DRO # v2
    train=np.concatenate((train,noisy_train[:our_config['noisy_sample_used'],:]), axis=0)
  

  W=scipy.io.loadmat('current_W.mat')

  W=W['W']
  
  W_inv=np.linalg.inv(W)
  W_minus_half=sqrtm(W_inv)
  W_half=sqrtm(W)
  #W_minus_half=W_inv
  #W_half=W



  W=W.astype(np.float32)
  W_inv=W_inv.astype(np.float32)
  W_minus_half=W_minus_half.astype(np.float32)
  W_half=W_half.astype(np.float32)
  
  W_minus_half=torch.Tensor(W_minus_half).cuda()




  #model=vision_model(config).cuda()
  #model.load_state_dict(torch.load(our_config['current_model']))
  model=vision_model().cuda()
  model.load_state_dict(torch.load(our_config['current_model']))
  model.eval()
  

  #DRO_trainer=DRO(config).cuda()
  DRO_trainer=DRO().cuda()
  DRO_trainer.load_state_dict(model.state_dict(),strict=True) # directly load all layers
  DRO_trainer.train()


  
  for param in DRO_trainer.parameters(): # when we train the P layer, it seems that unfreeze the downstream layer will help. we also had this observation before
    param.requires_grad = False

  
  if our_config['DRO_target_layer']=='P':
    #for param in DRO_trainer.vit.embeddings.patch_embeddings.our_projection.parameters():
    for param in DRO_trainer.MAe_vit.patch_embed.our_proj.parameters():
      param.requires_grad = True
  elif our_config['DRO_target_layer']=='B':
    for param in DRO_trainer.B.parameters():
      param.requires_grad = True
  else:
    #for param in DRO_trainer.vit.encoder.layer[our_config['DRO_target_layer']].output.dense.parameters():
      #param.requires_grad = True
    #for param in DRO_trainer.vit.encoder.layer[our_config['DRO_target_layer']].attention.attention.query.parameters():
      #param.requires_grad = True
    #for param in DRO_trainer.vit.encoder.layer[our_config['DRO_target_layer']].attention.attention.key.parameters():
      #param.requires_grad = True
    #for param in DRO_trainer.vit.encoder.layer[our_config['DRO_target_layer']].attention.attention.value.parameters():
      #param.requires_grad = True
    
    #for param in DRO_trainer.MAe_vit.blocks[our_config['DRO_target_layer']].attn.qkv.parameters():
      #param.requires_grad = True

    for param in DRO_trainer.MAe_vit.blocks[our_config['DRO_target_layer']].attn.proj.parameters():
      param.requires_grad = True


  '''
  for param in DRO_trainer.MAe_vit.patch_embed.our_proj.parameters():
    param.requires_grad = True
  for param in DRO_trainer.B.parameters():
    param.requires_grad = True
  for param in DRO_trainer.MAe_vit.blocks[our_config['DRO_target_layer']].attn.proj.parameters():
    param.requires_grad = True'''


  optimizer = torch.optim.Adam(DRO_trainer.parameters(), lr=our_config['DRO_lr'])
  #optimizer = torch.optim.AdamW(DRO_trainer.parameters(), lr=our_config['DRO_lr'], weight_decay=0) # don't use weight decay in dro
  
  batchsize=our_config['batchsize']
  all_index=[i for i in range(len(train))]
  random.seed(our_config['DRO_seed'])
  
  
  
  for e in range(1,our_config['epochs_each_DRO_stage']+1): 
      
  

      random.shuffle(all_index)
      
      #print(DRO_trainer.fc3.state_dict())
      
      
      for r in range(int(len(train)/batchsize)): # no +1: in training, make sure the batchsize is stable

          ind_slice=all_index[r*batchsize:(r+1)*batchsize]

          X=train[ind_slice,1:]
          y=train[ind_slice,0]


          X=torch.Tensor(X).to('cuda')
          X=torch.reshape(X,(-1,2,256,256))
          #X=torch.cat([X,X,X],1) # duplicate the channel since seems like this MAe only supports 3 channels now... will explore
          
          X=X[:,0:1,:,:] # this is b1000! never forget to change other positions as well
          
          X=X/255
          y=torch.LongTensor(y).to('cuda')


          optimizer.zero_grad()
          
          if our_config['friend_in_DRO']=='PGD':
            X=LinfPGDAttack(model=model,features=X,labels=y,epsilon=0.001,k=3,a=0.0004)
          
          loss, output = DRO_trainer(features=X, labels=y, W_minus_half=W_minus_half, DRO_coef=our_config['DRO_coef']) # use PGD attacked images in each iteration
          #loss, output, repr_ = model(features=X, labels=y)
          #print(output.shape)
          l_numerical = loss.item()
          
          loss.backward()
          optimizer.step()
          
          model.load_state_dict(DRO_trainer.state_dict(),strict=True) # directly load all layers, now for every iteration to perform PGD
          
      print(f"Epoch: {e}, Loss: {l_numerical}")
      


  
  '''
  e=0  # use these to quickly output the ERM model performance!!!!!
  l_numerical=0   # use these to quickly output the ERM model performance!!!!!'''



  model.load_state_dict(DRO_trainer.state_dict(),strict=True) # directly load all layers
  #print(model.cnn.fc2.state_dict()['weight'])
  
  
  # ------------------ test DRO model
  


  all_AUC=[]
  all_ACC=[]
  all_AUPRC=[]

  all_balance_AUC=[]
  all_balance_ACC=[]
  all_balance_AUPRC=[]


  out = open(our_config['performance_record_file'], 'a', newline='',encoding='utf-8')
  csv_write = csv.writer(out, dialect='excel')


  for filename in filename_list:
    
    print('Now testing the test set in '+filename)
    noisy_train,noisy_val,noisy_test,noisy_test_balance=load_and_split(filename=filename)
    
    PRED, metrics, REPR, REPR_inViT, PATCHES=pred_one_dataset_batch(model,dataset=noisy_test,output_repr=True,DRO_layer=our_config['DRO_target_layer'])
    print(metrics['cm'])
    print('ep'+str(e)+' test AUC: ',metrics['AUC'])
    print('ep'+str(e)+' test ACC: ',metrics['ACC'])
    print('ep'+str(e)+' test AUPRC: ',metrics['AUPRC'])
    
    
    all_AUC.append(metrics['AUC'])
    all_ACC.append(metrics['ACC'])
    all_AUPRC.append(metrics['AUPRC'])

    
    PRED, metrics, REPR, REPR_inViT, PATCHES=pred_one_dataset_batch(model,dataset=noisy_test_balance,output_repr=True,DRO_layer=our_config['DRO_target_layer'])
    
    all_balance_AUC.append(metrics['AUC'])
    all_balance_ACC.append(metrics['ACC'])
    all_balance_AUPRC.append(metrics['AUPRC'])

 
  csv_write.writerow(sys.argv+[e,l_numerical]+all_AUC+all_ACC+all_AUPRC+['']+all_balance_AUC+all_balance_ACC+all_balance_AUPRC)
  
  
  out.close()
  
  #torch.save(model.state_dict(), 'current_mnist_mlp8L_ep'+str(e)+'.pt')
  torch.save(model.state_dict(), 'current_mnist_vit.pt')
  



















