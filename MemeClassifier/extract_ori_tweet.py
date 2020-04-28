import glob
import json
"""
list_l = glob.glob('/data/yuhao/web_sci/meme_classifier/train_3/meme/*png')
list_1l = glob.glob('/data/yuhao/web_sci/meme_classifier/train_3/not_meme/*png')
list_2l = glob.glob('/data/yuhao/web_sci/meme_classifier/train_3/not_caption/*png')
list_3l = glob.glob('/data/yuhao/web_sci/meme_classifier/validation_3/meme/*png')
list_4l = glob.glob('/data/yuhao/web_sci/meme_classifier/validation_3/not_meme/*png') 

list_ = []
list_.extend(list_l)
list_.extend(list_1l)
list_.extend(list_2l)
list_.extend(list_3l)
list_.extend(list_4l)
print(list_[:10])


list_.sort(key = lambda x: int(x.split('/')[-1].split('_')[-2]))
print(list_[:10])
idx_max = len(list_)
print('maximum ids is:{}'.format(idx_max))
f1 = open('/data/yuhao/web_sci/meme_classifier/data/name_twitter.json','a')
with open('/data/yuhao/web_sci/meme_classifier/data/all_twitter.json','r') as json_file:
    idx = 0
    total = 0
    content = json_file.readlines()
    for l_index, line in enumerate(content):
        print('line index is: {}'.format(l_index))
        line = json.loads(line)
        num = len(line['extended_entities']['media'])
        total_end = total + num 
        for l_dir in list_[idx:]:
            name = l_dir.split('/')[-1]
            index = int(name.split('_')[-2])
            if index in range(total,total_end):
                dic = {}
                dic[name] = line
                idx += 1
                json.dump(dic,f1)
                f1.write('\n')
                print('match : {}'.format(name))
            elif index < total:
                print('wrong')
                break
            else:
                break
        total = total_end
        if idx >= idx_max:
            print('success')
            break
f1.close()

"""

import time
import os
import glob
import gzip
import json
import re
import zipfile
import pandas as pd
import pickle


dir_csv = '/data/yuhao/web_sci/meme_classifier/data/all_meme_2.csv'

f1 = open('/data/yuhao/web_sci/meme_classifier/data/name_twitter_meme1.json','a')
df = pd.read_csv(dir_csv)
print(df.columns)

def extract_tweet(file,index):
    """
    Function to extract tweet from gz file using file name and index
    """
    with gzip.open('/data/yuhao/download/twitter_data/' + file, 'rb') as gzfile:
        lines = gzfile.readlines()
        for l_index,line in enumerate(lines):
            if l_index == index:
                return json.loads(line)

def extract_tweet2(name):
    """
    Accroding to image name to extract the old tweet
    name = file_index~num_user_id....
    """
    file_index = name.split('_')[0]
    index = int(name.split('_')[1].split('~')[0])
    dir = '/data/yuhao/download/twitter_data/part-' + file_index + '-2dd5ecae-57fd-458b-8a18-86594b7614e8-c000.json.gz'
    
    with gzip.open(dir, 'rb') as gzfile:
        lines = gzfile.readlines()
        for l_index,line in enumerate(lines):
            if l_index == index:
                return json.loads(line)

list_dir = '/data/yuhao/web_sci/meme_classifier/train_3/meme1/*png'
#list_dir1 = '/data/yuhao/web_sci/meme_classifier/validation_3/meme_new/*png'

list_d = glob.glob(list_dir)
#list_d2 = glob.glob(list_dir1)
#list_d.extend(list_d2)
print(len(list_d))

start_time = time.time()

for i in list_d:
    name = i.split('/')[-1]
    twitter = extract_tweet2(name)
    json.dump({name:twitter},f1)
    f1.write('\n')
f1.close()
print('using time{}'.format(time.time()- start_time))
