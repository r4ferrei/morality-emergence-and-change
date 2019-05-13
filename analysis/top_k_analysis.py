import pandas as pd
import constant
import pickle
from os.path import join
from statsmodels.regression.linear_model import OLS, add_constant

test_type = 'BINARY'
df = pd.read_csv(join(constant.TEMP_DATA_DIR,
                      '{}_{}_retrievals.csv'.format('slope', test_type)))
wordlist = df[constant.WORD].values
f = open(join(constant.SGNS_DIR, 'freqs.pkl'), 'rb')
freq_dict = pickle.load(f, encoding='latin1')
conc_dict = pd.read_csv(join(constant.DATA_DIR, 'concretewords.csv'), index_col=0)
valence_dict = pd.read_csv(join(constant.DATA_DIR, 'valencewords.csv'), index_col=0)
freqlist = []
valencelist = []
concretelist = []

for word in wordlist:
    frequency = sum(freq_dict[word].values()) if word in freq_dict else None
    concrete = conc_dict.loc[word,'Conc.M'] if word in conc_dict.index else None
    valence = valence_dict.loc[word,'v_rating'] if word in valence_dict.index else None
    freqlist.append(frequency)
    valencelist.append(valence)
    concretelist.append(concrete)

df['valence'] = valencelist
df['concrete'] = concretelist
df['frequency'] = freqlist

df = df.dropna()
X = df[['valence', 'concrete', 'frequency']]
X = add_constant(X)
y = df['+']
cf = OLS(y, X)
results = cf.fit()
print(results.summary())