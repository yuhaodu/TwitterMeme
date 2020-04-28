import numpy as np
import matplotlib.pyplot as plt
import glob
import json

file_list = glob.glob('/data/yuhao/download/image_data/meme_stat/*json')

print(file_list[0])
not_meme_n = []
meme_n = []
for i in file_list:
    dict_up = json.load(open(i,'r'))
    nm_n = [len(v[0]) for k,v in dict_up.items()]
    m_n = [len(v[1]) for k,v in dict_up.items()]
#    print('nm_n length is {}'.format(len(nm_n)))
#    print('m_n length is {}'.format(len(m_n)))
#    print(nm_n)
    not_meme_n.extend(nm_n)
    meme_n.extend(m_n)

fig, ax = plt.subplots()
n,bins,patches = plt.hist(x=meme_n, bins = range(0,1000+50,50))
ax.set_xticks(bins)
plt.grid(axis='y', alpha = 0.75)
plt.xlabel('meme')
plt.ylabel('count')
plt.xlim(0,1000)
plt.ylim(0,10000)
plt.savefig('/data/yuhao/web_sci/meme_count_histogram.jpg')
plt.close()

def get_p_spammer():
    for i in file_list:
        dict_up = json.load(open(i,'r'))
        for k,v in dict_up.items():
            if len(v[1]) > 600:
                print(i)
                print(k)
                return

#get_p_spammer()

