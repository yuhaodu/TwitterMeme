#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 10:26:42 2018

@author: du

1. Normalize How to? 1). The whole dataset or training 2). After rescale or before rescale.


This is a file to implement a classifier for meme


# change version from benchmark to model
1. change model initilization
2. change train_log dir
3. change model_save dir
"""

import torch 
import torch.nn as nn 
from torchvision import models
import torch.nn.functional as F
import os
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils,datasets
from skimage import io, transform
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from PIL import ImageFile
from collections import OrderedDict
from torch import optim
import classifier_utils as cu
import time
from sklearn.metrics import recall_score, precision_score, accuracy_score
from torch.optim.lr_scheduler import MultiStepLR
ImageFile.LOAD_TRUNCATED_IMAGES = True

data_root = '/projects/academic/kjoseph/meme_classifier/train_3'

validation_root = '/projects/academic/kjoseph/meme_classifier/validation_3' 

test_root = '/projects/academic/kjoseph/meme_classifier/test_2'

log_dir = '/projects/academic/kjoseph/meme_classifier/model_train_log.txt' 

checkpoint_dir = '/projects/academic/kjoseph/meme_classifier/checkpoint/'

mis_dir = '/projects/academic/kjoseph/meme_classifier/mis_cla.pkl'


## define class for preprocessing


    

class TwitterDataset(Dataset):
    "twitter image dataset"
    def __init__(self, dir_, transform=None):
        """
        Args:
            dir : the directory for data
            transform(callable, optinal) : Optional transform to be applied on a sample
        """
        self.image_dir = glob.glob(dir_ + '/' + '*.png')
        self.image_dir.extend(glob.glob(dir_ + '/' + '*.jpg'))
        self.transform = transform
    
    def __len__(self):
        return len(self.image_dir)
    
    def __getitem__(self, idx):
        
        img_name = self.image_dir[idx]
        image = io.imread(str(img_name))[:,:,:3]
        
        if self.transform:
            image = self.transform(image)
        

        return image 

class TwitterDataset(Dataset):
    """
    twitter image dataset
    Load the image and its corresponding caption and original text.
    And this is for training validation and testing.
    """
    def __init__(self, dir_, transform=None):
        """
        Args:
            dir : the directory for data
            transform(callable, optinal) : Optional transform to be applied on a sample
        """
        meme_dir = dir_ + '/' + 'meme'
        nmeme_dir = dir_ + '/' + 'not_meme'
        self.image_dir = glob.glob(meme_dir + '/' + '*.png')
        self.image_dir.extend(glob.glob(meme_dir + '/' + '*.jpg'))
        self.image_dir.extend(glob.glob(nmeme_dir + '/' + '*.jpg'))
        self.image_dir.extend(glob.glob(nmeme_dir + '/' + '*.png'))
        self.transform = transform

    def __len__(self):
        return len(self.image_dir) 

    def __getitem__(self, idx):
        
        """
        Input the index of image and return its label and there is more stuff then add it here
        """

        img_name = self.image_dir[idx]
        #image = io.imread(str(img_name))[:,:,:3]
        label = int(img_name.split('.')[-2].split('_')[-1])
        name = img_name.split('/')[-1]
        image = Image.open(img_name) 
        if self.transform:
            image = self.transform(image)
        image = image[:3,:,:]
        #image = Image.open(img_name)
        return (image,label,name)


# calculate the mean and covariance for dataset for preprocessing0

"""
composed_ini = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
meme_dataset = datasets.ImageFolder(root=data_root,transform=composed_ini)
vali_dataset = datasets.ImageFolder(root=validation_root,transform= composed_ini)
channel_1 = [] 
channel_2 = []
channel_3 = []


for img in meme_dataset:
    img = np.asarray(img[0]) # Using Image folder, return(image(Image.open('mm.png')), label(int))
    channel_1.extend(img[0,:,:])
    channel_2.extend(img[1,:,:])
    channel_3.extend(img[2,:,:])

for img in vali_dataset:
    img = np.asarray(img[0])
    channel_1.extend(img[0,:,:])
    channel_2.extend(img[1,:,:])
    channel_3.extend(img[2,:,:])
mean_1 = np.mean(channel_1)
std_1 = np.std(channel_1)
mean_2 = np.mean(channel_2)
std_2 = np.std(channel_2)
mean_3 = np.mean(channel_3)
std_3 = np.std(channel_3)
print(mean_1)
print(mean_2)
print(mean_3)
print(std_1)
print(std_2)
print(std_3)
"""        
# mean_1 = 0.5750266162298565
# mean_2 = 0.5502954201862815
# mean_3 = 0.535786363271125
# std_1 = 0.35602713878803227
# std_2 = 0.353793437107924
# std_3 = 0.35727357548538692
#transforms.RandomHorizontalFlip(),
        
#Rescale((224,224))
mean_1 = 0.579662
mean_2 = 0.5555058
mean_3 = 0.5413896
std_1 = 0.3494197
std_2 = 0.3469673
std_3 = 0.35115704

#composed = transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[mean_1 , mean_2, mean_3],std=[std_1, std_2, std_3])])

composed = transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[mean_1 , mean_2, mean_3],std=[std_1, std_2, std_3])])
 
composed_vali = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[mean_1, mean_2, mean_3],std=[std_1, std_2, std_3])])
#composed_vali = transforms.Compose([transforms.Resize((256,256)),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[mean_1, mean_2, mean_3],std=[std_1, std_2, std_3])])

 
#

meme_dataset = datasets.ImageFolder(root=data_root,transform=composed)



#dataset_loader = DataLoader(meme_dataset,batch_size=cu.batch_size, shuffle=True,num_workers=4)


## build the model
    


resnet = models.resnet50(pretrained = True) # load pretrained teh resnet50 model

backbone = nn.Sequential(*list(resnet.children())[:-1])

class Meme_baseline(nn.Module):
    
    def __init__(self,batch_size):
        super(Meme_baseline,self).__init__()
        self.batch_size =batch_size
        self.conv1 = nn.Conv2d(3,32,3,stride=1,padding =1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32,64,3,stride =1, padding =1)
        self.maxpool2 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(p=0.6)
        self.fc1 = nn.Linear(64*56*56, 1000, bias = True)
        self.relu1 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(1000,2,bias = True)
        self.layer_init()
    
    def forward(self,x):
        
        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.maxpool2(out)
        out = self.drop1(out.view(out.shape[0],-1))
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.drop2(out)
        out = self.fc2(out)
        out = F.log_softmax(input = out,dim =1)
        return(out)
    
    def layer_init(self):
        self.conv1.weight.data.normal_(0.0,0.01)
        self.conv2.weight.data.normal_(0.0,0.01)
        self.fc1.weight.data.normal_(0.0,0.01)
        self.fc2.weight.data.normal_(0.0,0.01)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        
#try1 = Meme_baseline(4)
          

class Meme_classifier(nn.Module):
    
    def __init__(self):
        self.middle_layer = 1000
        super(Meme_classifier, self).__init__()
        self.backbone = backbone
        for param in self.backbone:
            param.requires_grad = False
        self.drop1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(2048,self.middle_layer,bias = True)
        self.relu1 = nn.ReLU() 
        #self.drop3 = nn.Dropout(p= 0.5)
        #self.fc3 = nn.Linear(self.middle_layer, 500, bias = True)
        #self.relu3 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(self.middle_layer,2,bias = True)
        #self.fc1 = nn.Linear(2048,2,bias = True)
        
    
    def forward(self, x):
        
        out = self.backbone(x)
        out = out.view([-1,2048])
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.relu1(out)
        #out = self.drop3(out)
        #out = self.fc3(out)
        #out = self.relu3(out)
        out = self.drop2(out)
        out = self.fc2(out)
        ##out = out.view([-1])
        out = F.log_softmax(input = out, dim =1)
        return(out)
        
    
# train 

def train(model,dataloader, loss, gpu, optimizer, dataset):
    """
    This is the function for training the model
    Input: model: training model
           loader: dataloader
           criterion: critertion  loss
           gpu: using gpu or not (binary)
           dataset: dataset  training
 
    """
    model.train()
    current_loss = 0
    index = 1
    for (train_,label) in dataloader:
        index += 1
        optimizer.zero_grad()
        
        if gpu:
            
            train_, label = train_.cuda(), label.cuda() # put data into GPU
            model = model.cuda() # put data into GPU
 
            output = model.forward(train_)
            #print('output type is: ' + str(output.type()))
            #label = label.float()
            #print('label type is: ' + str(label.type()))
            _, preds = torch.max(output,1)
            loss_ = loss(output,label)
            loss_.cuda() # send loss to GPU
        else:
            output = model.forward(train_)
            # _, preds = torch.max(output,1)
            loss_ = loss(output,preds)
        loss_.backward()
        optimizer.step()
        current_loss += loss_.item() * train_.shape[0]
    
    epoch_loss = current_loss / len(dataset)
    
    return model,epoch_loss

def validation(model,dataloader, loss, gpu, optimizer, dataset):
    """
    This is the function for evaluating the model
    Input: model: evaluation model
           loader: dataloader
           criterion: critertion for loss
           gpu: using gpu or not (binary)
           dataset: dataset for evaluation
           
    Output: precision, accuracy, recall
    
    """
    
    model.eval()
    current_loss = 0
    target = []
    output = []
    for (validation_,label) in dataloader:
                
        if gpu:
            
            validation_, label = validation_.cuda(), label.cuda() # put data into GPU
            model = model.cuda() # put data into GPU
            output_ = model.forward(validation_)
            _, preds = torch.max(output_,1)
            #label = label.float()
            #preds = (output_ > 0.5).float() # This is the prediction of output is output is bigger than 0.5, then it predicts the image as not meme 
            loss_ = loss(output_,label)
        else:
            output_ = model.forward(validation_)
            _, preds = torch.max(output_,1)
            loss_ = loss(output_,label)
        
        target.extend(label.cpu().numpy()) # add label to vector
        output.extend(preds.cpu().numpy()) # add prediction to output
        current_loss += loss_.item() * validation_.shape[0]
    target = (np.array(target) - 1) * (-1) # change meme's label to 1 and non meme to 0
    output = (np.array(output) -1) * (-1) # change meme's label to 1 and non meme to 0 
    recall =  recall_score(target,output)
    precision = precision_score(target,output)
    accuracy = accuracy_score(target,output)
    epoch_loss = current_loss / len(dataset)
    
    return epoch_loss,recall,precision,accuracy

#composed = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.575, 0.550, 0.536],std=[0.356, 0.353, 0.357])])

meme_dataset = datasets.ImageFolder(root=data_root,transform=composed)  # meme_dataset
validation_dataset = datasets.ImageFolder(root=validation_root, transform = composed_vali)  # validation dataset
## main_function:
    
model = Meme_classifier()# This is our model
#model = Meme_baseline(cu.batch_size) # This is bench mark model
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
loss = nn.NLLLoss()
#loss = nn.BCEWithLogitsLoss()
dataloader = DataLoader(meme_dataset,batch_size=cu.batch_size, shuffle=True,num_workers=4)
validation_dataloader = DataLoader(validation_dataset, batch_size = cu.batch_size, shuffle = True, num_workers = 4)
scheduler = MultiStepLR(optimizer,milestones = cu.milestones,gamma = 0.1)
cu.write_log(log_dir,'-------------------------')
#cu.write_log(log_dir,'middle_layer is: ' + str(model.middle_layer))
cu.write_log(log_dir,'-------------------------')

vali_loss_p = 1000000
index = 0 

# load the model from pretrained model

load_model = False
if load_model:

   print('Load model:-------------')
   model_dir = '/projects/academic/kjoseph/meme_classifier/checkpoint/model.pth'
   checkpoint = torch.load(model_dir)
   model.load_state_dict(checkpoint['model_state'])
   optimizer.load_state_dict(checkpoint['optimizer_state'])
   index = checkpoint['epochs']
   for state in optimizer.state.values():
       for k,v in state.items():
           if torch.is_tensor(v):
              state[k] = v.cuda()     # send all optimizer's value to gpu

for i in range(cu.num_epoch):
    i = index + i
    # train
    # load model
    scheduler.step()
    start = time.time()
    gpu = True # use GPU or not
    model,epoch_loss = train(model,dataloader,loss,gpu,optimizer,meme_dataset)
    log = "-------------The training epoch {} loss is: ".format(i+1) + str(epoch_loss)
    cu.write_log(log_dir,log)
    end = time.time()
    log_2 = 'Using {} sec for epoch {}'.format(start-end, i+ 1)
    cu.write_log(log_dir,log_2)
    
    ## validation
    
    vali_loss,recall,precision,accuracy = validation(model,validation_dataloader,loss,gpu,optimizer,validation_dataset)
    log_3 = '----------------Recall for epoch {} is : {}'.format(i+1,recall)
    log_4 = '----------------Precision for epoch {} is : {}'.format(i+1, precision)
    log_5 = '----------------Accuracy for epoch {} is {}'.format(i+1, accuracy)
    log_6 = '----------------Validation loss for epoch {} is {}'.format(i+1, vali_loss)
    
    cu.write_log(log_dir, log_3)
    cu.write_log(log_dir, log_4)
    cu.write_log(log_dir, log_5)
    cu.write_log(log_dir, log_6)
    if vali_loss < vali_loss_p: 
       vali_loss_p = vali_loss # give loss new value
       checkpoint = {'model_state': model.state_dict(),'criterion_state': loss.state_dict(),
              'optimizer_state': optimizer.state_dict(),'scheduler_state': scheduler.state_dict(),
              'epochs': i+1}
       torch.save(checkpoint, checkpoint_dir + 'model' + '.pth')
    
    
    
    
    
    

    
            

        
            
            
        
        









