"""

This is a file to rename all of file in meme and not_meme

"""

import glob
import os

dir_1 = '/projects/academic/kjoseph/meme_classifier/validation_3/meme'
dir_2 = '/projects/academic/kjoseph/meme_classifier/validation_3/not_meme'

list_= glob.glob(dir_2+'/*g')
for i in list_:
    os.rename(i, i.split('.')[-2] + str('_1.png') )
