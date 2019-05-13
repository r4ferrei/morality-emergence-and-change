import embeddings
import math
from os.path import join
from itertools import groupby
import constant
import pickle
import pandas as pd
import nltk


def extract_most_freq_words(df, k=5000):
    # for year in [1800, 1850, 1900, 1950]:
    df\
    .sort_values(by='frequency', ascending=False)\
    .head(k)\
    .to_csv(join(constant.TEMP_DATA_DIR, 'most_frequent.csv'.format(year)), index=False)

english_vocab = set(nltk.corpus.wordnet.all_lemma_names(pos='n', lang='eng'))
stopwords = set(nltk.corpus.stopwords.words('english'))

# freq_dict = {'ldk': 2, 'apple': {1920: 2.3, 1940: 3.2, 1800: 10}, 'banana': {1900: 3, 1850: 5, 1990: 2}}

df = []
total_counts = {}
for year in constant.ALL_YEARS:
    f = open(join(constant.SGNS_DIR, 'pos', '{}-pos_counts.pkl'.format(year)), 'rb')
    freq_dict = pickle.load(f, encoding='latin1')
    total_count = 0
    for word in freq_dict:
        if 'NOUN' in freq_dict[word] and word in english_vocab and word not in stopwords and word.isalpha():
            count = freq_dict[word]['NOUN']
            df.append({constant.WORD: word,
                       constant.YEAR: math.floor(year / 50) * 50,
                       'count': count})
            total_count += count
    total_counts[year] = total_count
df = pd.DataFrame(df)\
    .groupby(constant.WORD)\
    .filter(lambda x: len(x) == 20)
assert not df.empty
df['frequency'] = df['count']/df[constant.YEAR].map(total_counts)
df = df\
    .groupby(by=[constant.WORD], as_index=False)\
    .sum()\
    .drop(columns=[constant.YEAR])
print(df.head())
extract_most_freq_words(df, k=10000)