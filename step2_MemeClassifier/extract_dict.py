# first change the name of image to name + _0.png and then process it


from pytesseract import image_to_string
import re
import glob
import gzip
import json
import pickle
import os 
import requests
import zipfile 
import sys,traceback
import io
from PIL import Image

def extract_full_text(image_dir):
    image_name = image_dir.split('/')[-1]
    file = image_name.split('_')[0]
    index = int(image_name.split('_')[1].split('~')[0])
    text = image_to_string(image_dir)
    text = re.sub('\n',' ', text)
    file_list = glob.glob('/data/yuhao/download/twitter_data/part-'+ str(file) + '*.json.gz')
    file_ = file_list[0]
    with gzip.open(file_, 'rb') as gzfile:
         lines = gzfile.readlines()
         for i,line in enumerate(lines):
             if i == index:
                 tweet = json.loads(line)['full_text']
    return (text,tweet)

#dic = pickle.load(open('/data/yuhao/web_sci/meme_classifier/test_3.pkl','rb'))
#print(len([*dic.keys()]))
#dir_l = glob.glob('/data/yuhao/web_sci/meme_classifier/update/*png')
#dic_new = {}
#for k,v in dic.items():
#    try:
#        int(k.split('_')[-1].split('.')[0])
#        dic_new[k] = v
#    except:
#        dic_new[k.split('.')[0] + '_0.png'] = v

#for i in dir_l:
#    image = i.split('/')[-1]
#    dic[image] = extract_full_text(i)
#print(len([*dic.keys()]))

#pickle.dump(dic,open('/data/yuhao/web_sci/meme_classifier/test_4.pkl','wb'))


#def change_name_meme(dir):
#    """
#    change meme image's name into name_0.png
#    """
#    list_l = glob.glob(dir + '*png')
#    for i in list_l:
#        try:
#            n = int(i.split('/')[-1].split('_')[-1].split('.')[0])
#        except:
#            os.rename(i,i.split('.')[0] + '_0.png')


#change_name_meme('/data/yuhao/web_sci/meme_classifier/update/')
twitter_name = 'memes'
f1 = zipfile.ZipFile('/data/yuhao/web_sci/meme_classifier/update/{}.zip'.format(twitter_name),'a')
with gzip.open('/data/yuhao/web_sci/meme_classifier/twitter_gz_file/{}.json.gz'.format(twitter_name),'rb') as gzfile:
    dic = {}
    content = gzfile.readlines()
    for l_index,line in enumerate(content):
        twitter = json.loads(line)
        print('The index is:' + str(l_index))
        try:
            media = twitter['entities']['media']
            user_id = twitter['user']['id_str']
            for m in media:
                twitter_id = m['id_str']

                if m['type'] == 'photo':
                    response = requests.get(m['media_url'],timeout = 5)
                    if response.status_code == 200:
                       image_name = twitter_name  + '_' + str(twitter_id) + '_0.png'
                       text = image_to_string(Image.open(io.BytesIO(response.content)))
                       text = re.sub('\n', ' ', text)
                       if len(text.split(' ')) <2:
                           continue
                       f1.writestr(image_name,response.content)
                       full_text = twitter['full_text']
                       dic[image_name] = (text,full_text)
                       print('text is:' + text )
                       print('full_text is:' + full_text)
        except:
            traceback.print_exc(file = sys.stdout)
            pass
pickle.dump(dic,open('/data/yuhao/web_sci/meme_classifier/{}.pkl'.format(twitter_name),'wb'))
f1.close()
