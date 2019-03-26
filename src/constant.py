import os
import matplotlib.pyplot as plt
import re

def get_colour(category_name):
    cmap = plt.get_cmap('tab10')
    base_cat = re.sub('[\+\-]+', '', category_name)
    base_cats = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']
    return cmap(base_cats.index(base_cat))

TEMP_DATA_DIR = 'data'
DATA_DIR = 'local-data'
SGNS_DIR = 'E:/sgns'

DICTIONARY_PATH = os.path.join(DATA_DIR, 'dictionary.txt')
MFD_DF = os.path.join(DATA_DIR, 'mfd_dic.pkl')

CATEGORY = 'category'
WORD = 'word'
VECTOR = 'vector'
YEAR = 'year'
CONCEPT = 'concept'
SCORE = 'score'

ALL_YEARS = range(1800, 1991, 10)