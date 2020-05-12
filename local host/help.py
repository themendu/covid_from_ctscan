# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:07:14 2020

@author: HP
"""

from __future__ import division, print_function
# coding=utf-8
import os

import torch
import torchvision

from PIL import Image




import numpy as np
import cv2
#***********************************************************************
model = torch.load('best_model.pt',map_location=torch.device('cpu'))
#***********************************************************************
from torchvision import transforms
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])])


soft=torch.nn.Softmax(dim=0)



def model_predict_covid(img_path, model):
      lower_black = np.array([0,0,0], dtype = "uint16")
      upper_black = np.array([200,200,200], dtype = "uint16")
      image = cv2.imread(img_path)
      image3 = cv2.blur(image, (60,60))
      image_blurred= cv2.medianBlur(image3,3)
      black_mask = cv2.inRange(image_blurred, lower_black, upper_black)
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      dest_and = cv2.bitwise_and(gray,black_mask, mask = None) 
      dest_and= cv2.cvtColor(dest_and,cv2.COLOR_GRAY2RGB)
      (height, width,_ )= dest_and.shape
      width_cutoff = width // 2
      s1 = dest_and[:, :width_cutoff]
      s2 = dest_and[:, width_cutoff:]
      
      image1=Image.fromarray(s1)     
      image1 = TRANSFORM_IMG(image1).float()
      image1=image1.unsqueeze_(0)
      outputs = model(image1)
      d=outputs.detach().numpy()
      e=torch.tensor([d[0][0],d[0][1]])
      c1=soft(e)
      
      image1=Image.fromarray(s2)     
      image1 = TRANSFORM_IMG(image1).float()
      image1=image1.unsqueeze_(0)
      outputs = model(image1)
      d=outputs.detach().numpy()
      e=torch.tensor([d[0][0],d[0][1]])
      c2=soft(e)
      
      c=np.concatenate((c1,c2),axis=None)
      d=np.argmax(c)
      if d==0 or d==2: return float(100*(1-c[d])),'neg'
      else:  return  float(100*(c[d])),'pos'



def model_predict_noncovid(img_path, model):
     lower_black = np.array([0,0,0], dtype = "uint16")
     upper_black = np.array([170,170,170], dtype = "uint16")
     image = cv2.imread(img_path)
     image3 = cv2.blur(image, (40,40))
     image_blurred= cv2.medianBlur(image3,3)
     black_mask = cv2.inRange(image_blurred, lower_black, upper_black)
     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     dest_and = cv2.bitwise_and(gray,black_mask, mask = None) 
     dest_and= cv2.cvtColor(dest_and,cv2.COLOR_GRAY2RGB)
     (height, width,_ )= dest_and.shape
     width_cutoff = width // 2
     s1 = dest_and[:, :width_cutoff]
     s2 = dest_and[:, width_cutoff:]
 
     image1=Image.fromarray(s1)     
     image1 = TRANSFORM_IMG(image1).float()
     image1=image1.unsqueeze_(0)
     outputs = model(image1)
     d=outputs.detach().numpy()
     e=torch.tensor([d[0][0],d[0][1]])
     c1=soft(e)
      
     image1=Image.fromarray(s2)     
     image1 = TRANSFORM_IMG(image1).float()
     image1=image1.unsqueeze_(0)
     outputs = model(image1)
     d=outputs.detach().numpy()
     e=torch.tensor([d[0][0],d[0][1]])
     c2=soft(e)

     c=np.concatenate((c1,c2),axis=None)
     d=np.argmax(c)
     if d==0 or d==2: return float(100*(1-c[d])),'neg'
     else:  return  float(100*(c[d])),'pos'
    



#**********************************************************
file_path='abc.png'
#**********************************************************

        # Make prediction
a,m=model_predict_covid(file_path, model)
b,n=model_predict_noncovid(file_path,model)
if m=='neg' and n=='neg': 
    if(a>=b): print(str(a)+"% chance of covid") 
    else:  print(str(b)+"% chance of covid")
elif (m=='neg' and n=='pos'):    print(str(b)+"% chance of covid")
elif (m=='pos' and n=='neg'):    print(str(a)+"% chance of covid")
else:   
     if(a>=b): print(str(a)+"% chance of covid") 
     else:  print(str(b)+"% chance of covid")   


