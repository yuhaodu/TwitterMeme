import csv
import pandas as pd
import requests
import os
import zipfile
import json

"""
Change the column name in two function
at line 24 and line 42 
"""
os.mkdir('/data/yuhao/web_sci/meme_classifier/test')
os.mkdir('/data/yuhao/web_sci/meme_classifier/test/meme')

df = pd.read_csv('/data/yuhao/web_sci/meme_classifier/data/test.csv')
df.reset_index()

def saveimage_from_name(name,dir_):
    """
    Input: name : image name
           dir_ : directory for image
    Aim: extract image from zip file and save image into disk
    """
    label = df.loc[df['name'] == name, 'label'].iloc[0]
    print(label)
    if  'n' in label :
        label = '1'
    else:
        label = '0'
    file_num = name.split('_')[0]
    l_line = name.split('_')[1].split('~')[0]
    m_line = name.split('_')[1].split('~')[1]
    dir ='/data/yuhao/download/image_data/caption_image/{}cap.zip'.format(file_num)
    zip_file = zipfile.ZipFile(dir)
    image_byte = zip_file.read(name)
    name = name.split('.')[0] + '_{}.png'.format(label)
    f = open('{}/{}'.format(dir_,name),'wb')
    f.write(image_byte)

def save_dict(name):
    file_num = name.split('_')[0]
    label = df.loc[df['name'] == name, 'label'].iloc[0]
    if  'n' in label:
        label = '1'
    else:
        label = '0'
    dir = '/data/yuhao/download/image_data/caption/{}.json'.format(file_num)
    with open(dir,'r') as json_file:
        lines = json_file.readlines()
        for index, line in enumerate(lines):
            dic_l = json.loads(line)
            try:
                value = dic_l[name]
                break
            except:
                continue
    name = name.split('.')[0] + '_{}.png'.format(label)
    return {name:value}

dic_save = {}

for i in df['name']:
    try:
        print(i)
        saveimage_from_name(i,'/data/yuhao/web_sci/meme_classifier/test/meme')
        dic_update = save_dict(i)
        dic_save.update(dic_update)
    except:
        continue

json.dump(dic_save,open('/data/yuhao/web_sci/meme_classifier/test/dic1.json','w'))

