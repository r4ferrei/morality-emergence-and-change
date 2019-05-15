import os
import pandas as pd
import numpy as np
import constant
import pickle
from os.path import join
from statsmodels.regression.linear_model import OLS, add_constant
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--type", help="'binary' or 'null'")
parser.add_argument("--abs", action='store_true', help="absolute slope")
parser.add_argument("--log", action='store_true', help="log frequency")
args = parser.parse_args()

TYPE = args.type
ABS_SLOPE = args.abs
LOG_FREQ = args.log

plt.ion()
sns.set_context('paper')

RESTRICT_1800 = True

freq_dir = os.environ.get('FREQ_DIR', constant.SGNS_DIR)

# These responses are in leters instead of symbols due to R-style formulas.
# Adjust CSVs if re-generated.
if TYPE == 'binary':
    test_type = 'BINARY'
    RESPONSE = 'plus'
elif TYPE == 'null':
    test_type = 'NULL'
    RESPONSE = 'one'
else:
    assert(False)

df = pd.read_csv(join(constant.TEMP_DATA_DIR,
                      '{}_{}_retrievals.csv'.format('slope', test_type)))
wordlist = df[constant.WORD].values
f = open(join(freq_dir, 'freqs.pkl'), 'rb')
freq_dict = pickle.load(f, encoding='latin1')
conc_dict = pd.read_csv(join(constant.DATA_DIR, 'concretewords.csv'), index_col=0)
valence_dict = pd.read_csv(join(constant.DATA_DIR, 'valencewords.csv'), index_col=0)
freqlist = []
valencelist = []
concretelist = []
lengthlist = []

for word in wordlist:
    frequency = sum(freq_dict[word].values()) if word in freq_dict else None
    concrete = conc_dict.loc[word,'Conc.M'] if word in conc_dict.index else None
    valence = valence_dict.loc[word,'v_rating'] if word in valence_dict.index else None
    length = len(word)
    freqlist.append(frequency)
    valencelist.append(valence)
    concretelist.append(concrete)
    lengthlist.append(length)

df['valence'] = valencelist
df['concrete'] = concretelist
df['frequency'] = freqlist
df['logfrequency'] = np.log(freqlist)
df['length'] = lengthlist

if ABS_SLOPE:
    df['abs'] = df[RESPONSE].abs()
    del df[RESPONSE]

if RESTRICT_1800:
    df = df[df['start_year'] == 1800]


df = df.dropna()

#md = smf.mixedlm("{} ~ valence + concrete + logfrequency".format(RESPONSE),
#        df,
#        groups=df['word'])
#results = md.fit()
#print(results.summary())
#import sys; sys.exit(0)

formula = "{} ~ {} + length + concrete".format(
        'abs' if ABS_SLOPE else RESPONSE,
        'logfrequency' if LOG_FREQ else 'frequency')

cf = smf.ols(formula=formula, data=df)
results = cf.fit()
params = results.params

print(results.summary())


# Shuffle analysis.


shuffled = pd.read_csv(join(constant.TEMP_DATA_DIR,
                      '{}_{}_retrievals_shuffled.csv'.format('slope', test_type)))


response = 'abs' if ABS_SLOPE else RESPONSE

shuffled_coefs = []

for step in shuffled['step'].unique():
    curr = shuffled[shuffled['step'] == step]
    curr = curr[curr['word'].isin(df['word'].values)]

    if ABS_SLOPE:
        curr['abs'] = curr[RESPONSE].abs()

    curr = curr[['word', response]].merge(
            df[[
                'word',
                'valence',
                'concrete',
                'frequency',
                'logfrequency',
                'length']],
            on='word')

    cf = smf.ols(formula=formula, data=curr)
    res = cf.fit()
    shuffled_coefs.append(res.params)

shuffled_coefs = pd.DataFrame(shuffled_coefs)

low  = shuffled_coefs.quantile(.025)
high = shuffled_coefs.quantile(.975)

shuffle_result = pd.DataFrame.from_dict({
    'shuffle_95_low'   : low,
    'shuffle_95_high'  : high,
    'true_estimate'    : params
    },
    orient='index')

print("Shuffle test:")

print(shuffle_result)

# Plot

fig, ax = plt.subplots()

sns.barplot(data=shuffle_result.drop(columns=['Intercept']), ci='sd',
        orient='v', ax=ax, zorder=0, capsize=.15)
sns.scatterplot(data=params.drop(labels=['Intercept']), zorder=10,
        color='black', s=50)

title = ""
figname = ""

if TYPE == 'binary':
    if ABS_SLOPE:
        title = "absolute polarity slope regression coefficients"
        figname = "binary_absolute_shuffle_regression.pdf"
    else:
        title = "polarity slope regression coefficients"
        figname = "binary_shuffle_regression.pdf"
elif TYPE == 'null':
    title = "moral relevance slope regression coefficients"
    figname = "null_shuffle_regression.pdf"
else:
    assert(False)

ax.set_title(title)

plt.savefig(figname)
