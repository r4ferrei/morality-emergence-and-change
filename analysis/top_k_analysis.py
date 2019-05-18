import os
import embeddings
import models
import pandas as pd
import numpy as np
import constant
import pickle
from os.path import join
from statsmodels.regression.linear_model import OLS, add_constant
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import argparse
import pingouin as pg

parser = argparse.ArgumentParser()
parser.add_argument("--type", help="'binary' or 'null'")
parser.add_argument("--abs", action='store_true', help="absolute slope")
parser.add_argument("--log", action='store_true', help="log frequency")
args = parser.parse_args()

TYPE = args.type
ABS_SLOPE = args.abs
LOG_FREQ = args.log

embs, emb_vocab = embeddings.load_all()
mfd_neutral = models.choose_mfd_df('NULL', embs, True)
clf = models.CentroidModel()
clf.fit(mfd_neutral)


freq_dir = os.environ.get('FREQ_DIR', constant.SGNS_DIR)

# These responses are in leters instead of symbols due to R-style formulas.
# Adjust CSVs if re-generated.
if TYPE == 'binary':
    test_type = 'BINARY'
    RESPONSE = 'plus'
    RESTRICT_MORAL_FLANKS = True
    REQUIRE_MORAL_CHANGE = False

    name = "{}_{}_retrievals.csv"
elif TYPE == 'null':
    test_type = 'NULL'
    RESPONSE = 'one'
    RESTRICT_MORAL_FLANKS = False
    REQUIRE_MORAL_CHANGE = True

    name = "{}_{}_retrievals_include_irrelevant.csv"
else:
    assert(False)

df = pd.read_csv(join(constant.TEMP_DATA_DIR,
                      name.format('slope', test_type)))
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
    # Enforce that word embeddings exist throughout the period.
    if word not in emb_vocab:
        freqlist.append(None)
        valencelist.append(None)
        concretelist.append(None)
        lengthlist.append(None)
        continue

    # Possibly enforce that word starts and ends in moral domain.
    if RESTRICT_MORAL_FLANKS:
        word_embs = [embs[1800][word], embs[1990][word]]
        preds = clf.predict(word_embs)
        if preds[0] != '1' or preds[1] != '1':
            freqlist.append(None)
            valencelist.append(None)
            concretelist.append(None)
            lengthlist.append(None)
            continue

    # Possibly enforce that word starts and ends different relevance sides.
    if REQUIRE_MORAL_CHANGE:
        word_embs = [embs[1800][word], embs[1990][word]]
        preds = clf.predict(word_embs)
        if preds[0] == preds[1]:
            freqlist.append(None)
            valencelist.append(None)
            concretelist.append(None)
            lengthlist.append(None)
            continue

    frequency = sum(freq_dict[word].values()) if word in freq_dict else None
    concrete = conc_dict.loc[word,'Conc.M'] if word in conc_dict.index else None
    valence = valence_dict.loc[word,'v_rating'] if word in valence_dict.index else None
    length = len(word)
    freqlist.append(frequency)
    valencelist.append(valence)
    concretelist.append(concrete)
    lengthlist.append(length)

#df['valence'] = valencelist
df['concrete'] = concretelist
df['frequency'] = freqlist
df['length'] = lengthlist

if ABS_SLOPE:
    df['abs'] = df[RESPONSE].abs()
    del df[RESPONSE]

response = 'abs' if ABS_SLOPE else RESPONSE

df = df[['word', 'concrete', 'frequency', 'length', response]]
df = df.dropna()

df['logfrequency'] = np.log(df['frequency'])

#md = smf.mixedlm("{} ~ valence + concrete + logfrequency".format(RESPONSE),
#        df,
#        groups=df['word'])
#results = md.fit()
#print(results.summary())
#import sys; sys.exit(0)

formula = "{} ~ {} + length + concrete".format(
        response,
        'logfrequency' if LOG_FREQ else 'frequency')

cf = smf.ols(formula=formula, data=df)
results = cf.fit()
params = results.params

print(results.summary())


# Partial correlation analysis.

print("Partial correlations of significant factors and response:")

factor_ps = results.pvalues.drop(labels=['Intercept'])
signif_fac = factor_ps[factor_ps < .05].keys()

for i in range(len(signif_fac)):
    x = signif_fac[i]
    covar = signif_fac.drop(labels=[x]).tolist()
    y = response

    if not covar:
        continue

    pcor_res = pg.partial_corr(data=df, x=x, y=y, covar=covar)
    print("{} controlling for {}:".format(x, covar))
    print(pcor_res)

# Shuffle analysis.


shuffled = pd.read_csv(join(constant.TEMP_DATA_DIR,
                      '{}_{}_retrievals_shuffled.csv'.format('slope', test_type)))



shuffled_coefs = []

for step in shuffled['step'].unique():
    curr = shuffled[shuffled['step'] == step]
    curr = curr[curr['word'].isin(df['word'].values)]

    if ABS_SLOPE:
        curr['abs'] = curr[RESPONSE].abs()

    curr = curr[['word', response]].merge(
            df[[
                'word',
                #'valence',
                'concrete',
                #'frequency',
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

plt.ion()

matplotlib.rc('font', size=5)
matplotlib.rc('text', usetex=True)
fig, ax = plt.subplots(figsize=(3.03, 2.5))

shuffled_coefs_plot = shuffled_coefs.rename(columns={
    'logfrequency' : 'Frequency',
    'length'       : 'Length',
    'concrete'     : 'Concreteness',
    })

sns.barplot(data=shuffled_coefs_plot.drop(columns=['Intercept']), ci="sd",
        orient='v', ax=ax, zorder=0, capsize=.15)
sns.scatterplot(data=params.drop(labels=['Intercept']), zorder=10,
        color='black', s=50)

ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

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

title = "" # for paper

ax.set_title(title)
ax.set_ylabel("Coefficient")
plt.hlines(y=0, xmin=plt.xlim()[0], xmax=plt.xlim()[1], color='grey')
plt.tight_layout()

plt.savefig(figname, dpi=1000, bbox_inches='tight')
