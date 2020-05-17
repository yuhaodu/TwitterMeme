"""
These are different kinds of help function
"""

import pickle
import glob
import json 
def extract_all_pkl(dir_):
    """
    extract all dictionary within one folder and return one dictionary
    input:
        dir_: dir_ for all pkl file
    """

    file_list = glob.glob(dir_ + '*.pkl')
    dic = {}
    for i in file_list:
        dic_update = pickle.load(open(i,'rb'))
        dic.update(dic_update)

    file_list = glob.glob(dir_ + '*.json')
    for i in file_list:
        dic_update = json.load(open(i,'rb'))
        dic.update(dic_update)
    return dic
