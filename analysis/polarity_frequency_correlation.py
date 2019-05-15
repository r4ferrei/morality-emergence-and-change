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

# Load moral words starting from 1800.
name = "slope_BINARY_retrievals.csv"
df = pd.read_csv(join(constant.TEMP_DATA_DIR, name))
df = df[df['start_year'] == 1800]
wordlist = df[constant.WORD].values
del df

f = open(join(freq_dir, 'freqs.pkl'), 'rb')
freq_dict = pickle.load(f, encoding='latin1')

# Build time series of frequencies an compute correlations to per-decade log-odds.
word_cor_pval = {}

embs, vocab = embeddings.load_all()

binary_dict = models.choose_mfd_df('BINARY', embs, True)
clf = models.CentroidModel()
clf.fit(binary_dict)

years = sorted(embs.keys())

for word in wordlist:
    if word not in freq_dict:
        continue
    freqs = [freq_dict[word][year] for year in years]
    log_freqs = np.log(freqs)
    assert(len(log_freqs) == 20)

    preds = clf.predict_proba([embs[year][word] for year in years])
    preds = [p['+'] for p in preds]
    log_odds = np.log(preds) - np.log(np.subtract(1, preds))

    word_cor_pval[word] = scipy.stats.pearsonr(log_freqs, log_odds)

del wordlist

cors = np.array(sorted([cp[0] for cp in word_cor_pval.values()], reverse=True))

sns.pointplot(x=list(range(len(cors))), y=cors)
plt.xticks([])
plt.hlines(y=0, xmin=plt.xlim()[0], xmax=plt.xlim()[1])
plt.title('Correlations of per-decade log frequency and polarity per word')
plt.xlabel('word')
plt.ylabel('Pearson r')
plt.savefig('polarity_correlations.pdf')
