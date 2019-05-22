import embeddings
import math
from os.path import join
from itertools import groupby
import constant
import pickle
import pandas as pd
import nltk

"""
Pull the most frequent k words according to historical data statistics
"""


def extract_most_freq_words(df, k=5000):
    df\
    .sort_values(by='frequency', ascending=False)\
    .head(k)\
    .to_csv(join(constant.TEMP_DATA_DIR, 'most_frequent.csv'), index=False)


def run():
    df = []
    total_counts = {}
    for year in constant.ALL_YEARS:
        f = open(join(constant.SGNS_DIR, 'pos', '{}-pos_counts.pkl'.format(year)), 'rb')
        freq_dict = pickle.load(f, encoding='latin1')
        total_count = 0
        for word in freq_dict:
            for dict in emb_dict_all.values():
                if word not in dict:
                    continue
            if 'NOUN' in freq_dict[word] and word in english_vocab and word not in stopwords and word.isalpha():
                count = freq_dict[word]['NOUN']
                df.append({constant.WORD: word,
                           constant.YEAR: math.floor(year / 50) * 50,
                           'count': count})
                total_count += count
        total_counts[year] = total_count
    df = pd.DataFrame(df) \
        .groupby(constant.WORD) \
        .filter(lambda x: len(x) == 20)
    assert not df.empty
    df['frequency'] = df['count'] / df[constant.YEAR].map(total_counts)
    df = df \
        .groupby(by=[constant.WORD], as_index=False) \
        .sum() \
        .drop(columns=[constant.YEAR])
    extract_most_freq_words(df, k=10000)


english_vocab = set(nltk.corpus.wordnet.all_lemma_names(pos='n', lang='eng'))
stopwords = set(nltk.corpus.stopwords.words('english'))
emb_dict_all,_ = embeddings.choose_emb_dict('NGRAM')

run()
