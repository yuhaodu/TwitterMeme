"""
This is file for benchmark

Use svm or logistic regression for meme classifier 

"""




import torch 
import torch.nn as nn 
from torchvision import models
import os
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFile
import bcolz
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm
from shutil import copyfile
import TwitterDataset as td
import classifier_utils as cu 
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import numpy as np
from sklearn.model_selection import KFold
import csv
mean_1 = 0.579662
mean_2 = 0.5555058
mean_3 = 0.5413896
std_1 = 0.3494197
std_2 = 0.3469673
std_3 = 0.35115704
ImageFile.LOAD_TRUNCATED_IMAGES = True

data_root = '/projects/academic/kjoseph/meme_classifier/train_3'

validation_root = '/projects/academic/kjoseph/meme_classifier/validation_3'

test_root = '/projects/academic/kjoseph/meme_classifier/test_2'

glove_dir = '/projects/academic/kjoseph/download_result/glove.6B/'
vectors = bcolz.open(glove_dir + '6B.50.dat')[:]
words = pickle.load(open(glove_dir + '6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(glove_dir + '6B.50_idx.pkl', 'rb'))
matrix = torch.FloatTensor(vectors)
resnet = models.resnet50(pretrained = True) # load pretrained the resnet50 model
backbone = nn.Sequential(*list(resnet.children())[:-1])

dic_1 = pickle.load(open('/projects/academic/kjoseph/download_result/newdict_meme.pkl','rb'))

dic_2 = pickle.load(open('/projects/academic/kjoseph/download_result/newdict_nmeme.pkl','rb')) 
dic_1.update(dic_2) 
dic_3 = pickle.load(open('/projects/academic/kjoseph/download_result/newdict_tmeme.pkl','rb'))
composed = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[mean_1, mean_2, mean_3],std=[std_1, std_2, std_3])])


train_dataset = td.TwitterDataset(data_root,word2idx,dic_1,cu.max_len,transform = composed)

validation_dataset = td.TwitterDataset(validation_root,word2idx,dic_1,cu.max_len,transform = composed)

test_dataset = td.TwitterDataset(test_root,word2idx,dic_3,cu.max_len,transform = composed)

train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle = True, num_workers = 4)

validation_dataloader = DataLoader(validation_dataset, batch_size = 64, shuffle = True, num_workers = 4)

test_dataloader = DataLoader(test_dataset, batch_size = 64, shuffle = True, num_workers = 4)

def f_l_extract(dataloader,backbone):

    """
    this is a function to extract the label and datafeature from the dataset using cnn.  
    return the feature matrix and label vector
    input: 
       dataloader: pytorch dataloader
       backbone: cnn for feature extractor
    """
    backbone = backbone.cuda()
    feature = [] # the feature for whole dataset
    label = [] # the label for whole dataset
    for (data_,label_,name_, text_, leng_) in dataloader:
         data_ = data_.cuda()
         output = backbone(data_).view(data_.shape[0],-1)
         feature.extend(output.detach().cpu().numpy())
         label.extend(label_.numpy())
    return feature,label

def f_l_extract(dataloader,backbone):
    """
    this is a function to extract the label and datafeature from the dataset using cnn and bag of the words  
    return the feature matrix and label vector
    input: 
       dataloader: pytorch dataloader
       backbone: cnn for feature extractor
       matrix: matrix for initilizing the word embedding layer
    """
    embedding = nn.Embedding.from_pretrained(matrix).cuda()
    backbone = backbone.cuda()
    feature = [] # the feature for whole dataset
    label = [] # the label for whole dataset
    name = []
    for (data_,label_,name_, text_, leng_) in dataloader:
         data_ = data_.cuda()
         output = backbone(data_).view(data_.shape[0],-1)
         text_ = text_.long().cuda()
         output2 = embedding(text_).sum(1)
         #leng_ = (leng_.view(leng_.shape[0],1).repeat(1,output.shape[1]) + 1).float().cuda()
         #output = output / leng_
         #print('output 2 shape: ' + str(output2.shape))
         output = torch.cat([output,output2],dim = 1)
         feature.extend(output.detach().cpu().numpy())
         label.extend(label_.numpy())
         name.extend(name_)
    return feature,label,name


def extract_index(target,output):
    target = (np.array(target) - 1) * (-1) # change meme's label to 1 and non meme to 0
    output = (np.array(output) -1) * (-1) # change meme's label to 1 and non meme to 0  
    fal_p = []
    fal_n = [] 
    len_ = len(target)
    for i in range(len_):
        if target[i] == 1 and output[i] == 0:
           fal_n.append(i)
        if target[i] == 0 and output[i] == 1:
           fal_p.append(i)

    return fal_p,fal_n
        
        



def cal_stat(target,output):
    target = (np.array(target) - 1) * (-1) # change meme's label to 1 and non meme to 0
    output = (np.array(output) -1) * (-1) # change meme's label to 1 and non meme to 0   
    recall =  recall_score(target,output)
    precision = precision_score(target,output)
    accuracy = accuracy_score(target,output)
    return recall,precision,accuracy

train_feature,train_label,train_name = f_l_extract(train_dataloader,backbone)
validation_feature,validation_label,validation_name = f_l_extract(validation_dataloader,backbone)
test_feature,test_label,test_name = f_l_extract(test_dataloader,backbone)


## Logistic regression

print(len(train_feature))
print(len(train_feature[0]))
print(len(train_label))
print('---------')
print(len(validation_feature))
print(len(validation_feature[0]))
ss = StandardScaler() # define a scaler for data which is used for standardlization
ss.fit(train_feature) 
train_feature = ss.transform(train_feature)  # get normalized data    

pca = PCA(n_components = 512) # define pca and pca dimension
pca.fit(train_feature) 
#print('the variance explained is:' + str(np.sum(pca.explained_variance_ratio_)))
train_feature = pca.transform(train_feature)

"""
kf = KFold(n_splits = 10)
kf.get_n_splits(train_feature)

recall_l = []
precision_l = []
accuracy_l = []
train_label = np.array(train_label)
for train_index, test_index in kf.split(train_feature):
    logisticregr = LogisticRegression(penalty = 'l1', solver = 'saga', max_iter = 300 )
    fold_train = train_feature[train_index]
    fold_trlabel = train_label[train_index]
    fold_test = train_feature[test_index]
    fold_telabel = train_label[test_index]
    logisticregr.fit(fold_train, fold_trlabel)
    fold_p = logisticregr.predict(fold_test)
    recall,precision,accuracy = cal_stat(fold_telabel, fold_p)
    recall_l.append(recall)
    precision_l.append(precision)
    accuracy_l.append(accuracy)

print('recall is: ' + str(recall_l))
print('precision is :' + str(precision_l))
print('accuracy is : ' + str(accuracy_l))
"""
logisticregr = LogisticRegression(penalty = 'l1', solver = 'saga', max_iter = 300 )
logisticregr.fit(train_feature, train_label)

validation_feature = ss.transform(validation_feature)
validation_feature = pca.transform(validation_feature)

validation_p = logisticregr.predict(validation_feature)

recall,precision,accuracy = cal_stat(validation_label, validation_p)
print('recall is:' + str(recall))
print('precision is: ' +str(precision))
print('accuracy is :' + str(accuracy))

test_feature = ss.transform(test_feature)
test_feature = pca.transform(test_feature)

test_p = logisticregr.predict(test_feature)
test_pro = logisticregr.predict_proba(test_feature)
recall,precision,accuracy = cal_stat(test_label, test_p)
print('test recall is:' + str(recall))
print('test precision is: ' +str(precision))
print('test accuracy is :' + str(accuracy))

fal_p,fal_n = extract_index(test_label, test_p)

fal_p.sort(key = lambda x : test_pro[x][1]) # sort the index list accord to the probability and the wrongest to be the first
fal_n.sort(key = lambda x : test_pro[x][0]) # sort the index list according to the probability and the make wrongest to the first

dict_ = pickle.load(open('/projects/academic/kjoseph/meme_classifier/testname_text_url.pkl','rb'))

test_name = np.array(test_name)

field_name = ['name','full_text','url' , 'score' ]
with open('/projects/academic/kjoseph/meme_classifier/fal_p.csv','w',  newline = '') as csvfile:
     writer = csv.DictWriter(csvfile,field_name)
     for i in fal_p:
         writer.writerow({'name':  test_name[i], 'full_text' : dict_[test_name[i]][0], 'url' : dict_[test_name[i]][1], 'score': test_pro[i][1] }) 

with open('/projects/academic/kjoseph/meme_classifier/fal_n.csv','w',  newline = '') as csvfile:
     writer = csv.DictWriter(csvfile,field_name)
     for i in fal_n:
         writer.writerow({'name':  test_name[i], 'full_text' : dict_[test_name[i]][0], 'url' : dict_[test_name[i]][1], 'score' : test_pro[i][0] })


## SVM
"""
ss = StandardScaler()
pca = PCA(n_components = 512)
ss.fit(train_feature)
train_feature = ss.transform(train_feature)
pca.fit(train_feature) 
train_feature = pca.transform(train_feature)
validation_feature = ss.transform(validation_feature)
validation_feature = pca.transform(validation_feature)
test_feature = ss.transform(test_feature)
test_feature = pca.transform(test_feature)


clf = svm.SVC(gamma= 'scale')
clf.fit(train_feature,train_label)
validation_p = clf.predict(validation_feature)
test_p = clf.predict(test_feature)
recall,precision,accuracy = cal_stat(validation_label, validation_p)
print('recall is:' + str(recall))
print('precision is: ' +str(precision))
print('accuracy is :' + str(accuracy))
recall,precision,accuracy = cal_stat(test_label, test_p)
print('test recall is:' + str(recall))
print('test precision is: ' +str(precision))
print('test accuracy is :' + str(accuracy))
"""
