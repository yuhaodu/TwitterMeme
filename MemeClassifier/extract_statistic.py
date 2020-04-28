"""
This is file is used to extract tweet statistic from tweet name
"""
import json 
dir_1 = '/data/yuhao/web_sci/meme_classifier/data/name_twitter.json'
dir_2 = '/data/yuhao/web_sci/meme_classifier/data/name_twitter_meme1.json'
dir_3 = '/data/yuhao/web_sci/meme_classifier/data/name_twitter_new.json'

def extract_statistic(dir):
    """
    Input:
        dir: dir of csv file
    output:
        {name: folllower, friends, retweet,.. }

    """
    dic_all = {}
    with open(dir,'r') as file:
        lines = file.readlines()
        for line in lines:
            dic = json.loads(line)
            keys = [*dic.keys()][0]
            value = dic[keys]
            dic_new = {}
            dic_new['rt_c'] = value['retweet_count']
            dic_new['f_c'] = value['favorite_count']
            dic_new['friend_c'] = value['user_friends_count']
            dic_new['followe_c']= value['user_followers_count']
            dic_all[keys] = dic_new
    return dic_all            
dic_1 = extract_statistic(dir_1)
dic_2 = extract_statistic(dir_2)
dic_3 = extract_statistic(dir_3)
dic_1.update(dic_2)
dic_1.update(dic_3)

json.dump(dic_1,open('/data/yuhao/web_sci/meme_classifier/data/name_statistic.json','w'))
