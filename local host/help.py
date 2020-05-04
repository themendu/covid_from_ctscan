# -*- coding: utf-8 -*-
"""
Created on Fri May  1 20:41:01 2020

@author: HP
"""


import torch
import torchvision
from PIL import Image
device = torch.device("cpu")
from torchvision import transforms
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])])
from torchvision import transforms
import cv2
import numpy as np
import  PIL
to_pil = transforms.ToPILImage()

lower_black = np.array([0,0,0], dtype = "uint16")
upper_black = np.array([200,200,200], dtype = "uint16")

soft=torch.nn.Softmax(dim=0)

model = torch.load('best_model.pt',map_location=torch.device('cpu'))


def model_predict_covid(img_path, model):
      image = cv2.imread(img_path)
      kernel = np.ones((5,5),np.float32)/25
      image3 = cv2.blur(image, (60,60))
      image_blurred= cv2.medianBlur(image3,3)
      black_mask = cv2.inRange(image_blurred, lower_black, upper_black)
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      dest_and = cv2.bitwise_and(gray,black_mask, mask = None) 
      dest_and= cv2.cvtColor(dest_and,cv2.COLOR_GRAY2RGB)
      image=Image.fromarray(dest_and)
      image = TRANSFORM_IMG(image)
      image=image.unsqueeze_(0)      
      outputs = model(image)
      d=outputs.detach().numpy()
      e=torch.tensor([d[0][0],d[0][1]])
      c=soft(e)
      return c.numpy()
  
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
     image=Image.fromarray(dest_and)
     image = TRANSFORM_IMG(image).float()
     image=image.unsqueeze_(0)
     outputs = model(image) 
     d=outputs.detach().numpy()
     e=torch.tensor([d[0][0],d[0][1]])
     c=soft(e)
    
     return c.numpy()


a=model_predict_covid('perfect.jpg', model)
b=model_predict_noncovid('perfect.jpg',model)
c=np.concatenate((a,b),axis=None)
d=np.argmax(c)
if d==0 or d==2:  print(str(100*(1-c[d]))+"% chance of covid-19")
else:     print(str(100*c[d])+"% chance of covid-19")

