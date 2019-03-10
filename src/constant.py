import os

DATA_DIR = 'C:/Users/Jing/Documents/Work/Winter2019/MoralSentiments/morality-emergence-and-change/data'
SGNS_DIR = 'D:/WordEmbeddings'
DICTIONARY_PATH = os.path.join(DATA_DIR, 'dictionary.txt')

MFD_DF = os.path.join(DATA_DIR, 'mfd_dic.pkl')

CATEGORY = 'category'
WORD = 'word'
VECTOR = 'vector'
YEAR = 'year'

ALL_YEARS = range(1800, 1991, 10)