#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is used to fetch the url for meme

@author: du
"""

import glob
import pandas as pd
import requests
import io 
import skimage.io
from PIL import Image
import pickle
import sys,traceback
# get a new csv file




dir_meme = '/projects/academic/kjoseph/meme_classifier/train_3/meme'

dir_meme_2  = '/projects/academic/kjoseph/meme_classifier/validation_3/meme'

list_1 = glob.glob(dir_meme + '/*.png')
list_2 = glob.glob(dir_meme_2 + '/*.png')

list_1.extend(list_2)
def write_tolog(dir_,log):
    f1 = open(dir_,'a')
    f1.write(log)
    f1.write('\n')
    f1.close()

f1 = '/projects/academic/kjoseph/meme_classifier/log_fetch.txt'
list_1 = [i for i in list_1 if len(i.split('/')[-1].split('_')) == 2]
list_1.sort(key = lambda x : int(x.split('/')[-1].split('_')[0]))
list_2 = [ i.split('/')[-1]  for i in list_1 ]

df = pd.read_csv('/projects/academic/kjoseph/download/all_meme.csv')

df = df[df.user_id_str != 27995305]


len_ = len(list_1)
len_2 = len(df)
num = 0
dic = {}
for i in range(len_):
    print(i)
    
    img_1 = skimage.io.imread(list_1[i])
    
    for j in range(num,len_2):
        
        try:
            response = requests.get(df.media_url[j],timeout =5)
            if response.status != 200:
               continue
            img_2 = skimage.io.imread(io.BytesIO(response.content))
            if (img_2 == img_1).all():
                dic[list_2[i]] = j
                num += 1 
                log = 'Find {} sucessful from csv column {}'.format(list_2[i],j) 
                write_tolog(f1,log)
                break
            else:
                log = 'Continue'
                write_tolog(f1,log)
                continue
        except:
            traceback.print_exc(file = sys.stdout)
            continue

pickle.dump(dic,open('/projects/academic/kjoseph/meme_classifier/original_meme.pkl','wb'))
print('Then number of key: ' + str(len([*dic.keys()])))
print(len([*dic.keys()]))        
   
