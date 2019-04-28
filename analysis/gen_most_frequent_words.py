import embeddings
import math
from os.path import join
from itertools import groupby
import constant
import pickle
import pandas as pd
import nltk


def extract_most_freq_words(df, k=5000):
    for year in [1800, 1850, 1900, 1950]:
        df[df[constant.YEAR] == year]\
            .sort_values(by='frequency', ascending=False)\
            .head(k)\
            .to_csv(join(constant.TEMP_DATA_DIR, '{}_most_frequent.csv'.format(year)), index=False)

english_vocab = set(nltk.corpus.wordnet.all_lemma_names())
stopwords = set(nltk.corpus.stopwords.words('english'))

# freq_dict = {'ldk': 2, 'apple': {1920: 2.3, 1940: 3.2, 1800: 10}, 'banana': {1900: 3, 1850: 5, 1990: 2}}

df = []
for year in constant.ALL_YEARS:
    f = open(join(constant.SGNS_DIR, 'pos', '{}-pos_counts.pkl'.format(year)), 'rb')
    freq_dict = pickle.load(f, encoding='latin1')
    for word in freq_dict:
        if 'NOUN' in freq_dict[word] and word in english_vocab and word not in stopwords:
            df.append({constant.WORD: word,
                       constant.YEAR: math.floor(year / 50) * 50,
                       'frequency': freq_dict[word]['NOUN']})
df = pd.DataFrame(df)
df = df.groupby(by=[constant.WORD, constant.YEAR], as_index=False).sum()
extract_most_freq_words(df)