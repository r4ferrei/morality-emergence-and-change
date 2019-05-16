import embeddings
import os
import pandas as pd
import numpy as np
import constant
import pickle
from os.path import join
import scipy.stats
from statsmodels.regression.linear_model import OLS, add_constant
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import pingouin as pg
import models

plt.ion()
sns.set_context('paper')

freq_dir = os.environ.get('FREQ_DIR', constant.SGNS_DIR)

# Load moral words.
name = "slope_BINARY_retrievals.csv"
df = pd.read_csv(join(constant.TEMP_DATA_DIR, name))
wordlist = df[constant.WORD].values
del df

f = open(join(freq_dir, 'freqs.pkl'), 'rb')
freq_dict = pickle.load(f, encoding='latin1')

# Build time series of frequencies an compute correlations to per-decade log-odds.
word_cor_pval = {}
shuffled_cors = []

embs, vocab = embeddings.load_all()

binary_dict = models.choose_mfd_df('BINARY', embs, True)
clf = models.CentroidModel()
clf.fit(binary_dict)

neutral_mfd = models.choose_mfd_df('NULL', embs, True)
clf_neutral = models.CentroidModel()
clf_neutral.fit(neutral_mfd)

years = sorted(embs.keys())

for word in wordlist:
    if word not in freq_dict:
        continue
    if word not in vocab:
        continue

    relevance = clf_neutral.predict([embs[1800][word], embs[1990][word]])
    if relevance[0] != '1' or relevance[1] != '1':
        continue

    freqs = [freq_dict[word][year] for year in years]
    log_freqs = np.log(freqs)
    assert(len(log_freqs) == 20)

    preds = clf.predict_proba([embs[year][word] for year in years])
    preds = [p['+'] for p in preds]
    log_odds = np.log(preds) - np.log(np.subtract(1, preds))

    word_cor_pval[word] = scipy.stats.pearsonr(log_freqs, log_odds)

    control = []
    for _ in range(10):
        np.random.shuffle(log_odds)
        control.append(scipy.stats.pearsonr(log_freqs, log_odds)[0])
    shuffled_cors.append(np.mean(control))

del wordlist

cors = np.array(sorted([cp[0] for cp in word_cor_pval.values()], reverse=True))
shuffled_cors = sorted(shuffled_cors, reverse=True)

sns.pointplot(x=list(range(len(cors))), y=cors)
sns.pointplot(x=list(range(len(shuffled_cors))), y=shuffled_cors, color='grey')
plt.xticks([])
plt.hlines(y=0, xmin=plt.xlim()[0], xmax=plt.xlim()[1])
plt.title('Correlations of per-decade log frequency and polarity per word')
plt.xlabel('word')
plt.ylabel('Pearson r')
plt.savefig('polarity_correlations.pdf')
