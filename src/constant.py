import os
import matplotlib.pyplot as plt
import re

def get_colour(category_name):
    cmap = plt.get_cmap('tab10')
    base_cat = re.sub('[\+\-]+', '', category_name)
    base_cats = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']
    return cmap(base_cats.index(base_cat))

def get_linestyle(category_name):
    linestyles = [(0,()), (0,(1,1)), (0,(5,1)), (0,(5,5)), (0,(3,5,1,5,1,5))]
    base_cat = re.sub('[\+\-]+', '', category_name)
    base_cats = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']
    return linestyles[base_cats.index(base_cat)]

TEMP_DATA_DIR = 'C:/Users/Jing/Documents/GitHub/morality-emergence-and-change/data'
DATA_DIR = 'C:/Users/Jing/Documents/GitHub/morality-emergence-and-change/local-data'
# SGNS_DIR = 'D:/sgns'
SGNS_COHA_DIR = 'C:/Users/Jing/Documents/GitHub/morality-emergence-and-change/data/coha-word_sgns/sgns'
SGNS_DIR = 'C:/Users/Jing/Documents/GitHub/morality-emergence-and-change/data/sgns'
SGNS_NYT_DIR = 'C:/Users/Jing/Documents/GitHub/morality-emergence-and-change/data/nyt'
SGNS_FICTION_DIR = 'D:/kim'

DICTIONARY_PATH = os.path.join(DATA_DIR, 'dictionary.txt')
MFD_DF = os.path.join(TEMP_DATA_DIR, 'mfd_dic.pkl')
MFD_DF_NEUTRAL = os.path.join(TEMP_DATA_DIR, 'mfd_dic_neutral.pkl')
MFD_DF_BINARY = os.path.join(TEMP_DATA_DIR, 'mfd_dic_binary.pkl')

MFD_DF_NEUTRAL = os.path.join(DATA_DIR, 'mfd_dic_neutral.pkl')

CATEGORY = 'category'
WORD = 'word'
VECTOR = 'vector'
YEAR = 'year'
CONCEPT = 'concept'
SCORE = 'score'

ALL_YEARS = range(1800, 1991, 10)
