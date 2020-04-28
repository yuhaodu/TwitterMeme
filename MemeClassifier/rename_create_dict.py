import glob
from pytesseract import image_to_string
import json 
import os
import re
dir = '/data/yuhao/web_sci/meme_classifier/data/google_tweet_meme'
dic = {}
for j,_,i in os.walk(dir):
    for index,image in enumerate(i):
        image_caption = re.sub('\n',' ',image_to_string(j + '/' + image))
        if len(image_caption.split(' ')) >= 2:
            name = 'google_tweetmeme_{}_0.png'.format(index)
            os.rename(j + '/' + image,j+'/' + name)
            dic[name] = (image_caption,'')

print(image_caption)
json.dump(dic,open('/data/yuhao/web_sci/meme_classifier/data/google_tweet_meme.json','w'))
