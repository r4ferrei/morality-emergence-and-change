import pickle
import constant
import embeddings
import os
import pandas as pd
import models
from models import CentroidModel,TwoTierCentroidModel
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

def load_test_df(emb_dict_all=None, reload=False):
    if reload:
        test_df = []
        for year in years:
            yr_emb_dict = emb_dict_all[year]
            for i,word in enumerate(test_words):
                if word in yr_emb_dict:
                    test_df.append({constant.YEAR:year,constant.WORD:word,constant.VECTOR:yr_emb_dict[word]})
        pickle.dump(pd.DataFrame(test_df),open(os.path.join(constant.TEMP_DATA_DIR, 'words.pkl'), 'wb'))
    return pickle.load(open(os.path.join(constant.TEMP_DATA_DIR, 'words.pkl'), 'rb'))

def bootstrap(model, word_df, mfd_dict, n=1000):
    categories = mfd_dict[constant.CATEGORY].unique()
    cons_predictions = []
    model.fit(mfd_dict)
    mean_predictions = model.predict_proba(word_df[constant.VECTOR])
    for i in range(n):
        resample_mfd_dict = mfd_dict.sample(n)
        model.fit(resample_mfd_dict)
        all_predictions = c.predict_proba(word_df[constant.VECTOR])
        cons_predictions.append(all_predictions)
    lower_bound = []
    upper_bound = []
    for i,year in word_df[constant.YEAR].values:
        year_predictions = [x[i] for x in cons_predictions]
        lower_bound.append({cat: min([x[cat] for x in year_predictions]) for cat in categories})
        upper_bound.append({cat: max([x[cat] for x in year_predictions]) for cat in categories})
    return mean_predictions, lower_bound, upper_bound

# Params
binary_fine_grained = np.array(['BINARY', 'FINEGRAINED'])[[False, True]]
plot_separate = True # Only has effect in the binary case
mfd_year = 1990
load = False
years = constant.ALL_YEARS
test_words = ['slavery', 'feminism', 'racism', 'abortion', 'automation', 'computer', 'diversity',
              'education', 'electric', 'engineer', 'environment', 'feminism', 'immigration',
              'information', 'machine', 'mechanic', 'nazism', 'phone', 'privacy', 'racism',
              'religion', 'robotics', 'sexism', 'technology']
nyt_corpus = ['NYT', 'NGRAM'][1]
all_models = [CentroidModel()]

# Loading data
emb_dict_all = None
if load:
    emb_dict_all,_ = embeddings.load_all(dir=constant.SGNS_DIR)
test_df = load_test_df(emb_dict_all=emb_dict_all, reload=load)
c = CentroidModel()
reduced_mfd_dict_list = []
mfd_dict = models.load_mfd_df(emb_dict_all, load)
mfd_dict_binary = models.load_mfd_df_binary(emb_dict=emb_dict_all, reload=False)

# Generate each plot
hsv = plt.get_cmap('nipy_spectral')
reduced_mfd_dict = mfd_dict[mfd_dict[constant.YEAR] == mfd_year]
reduced_mfd_dict_binary = mfd_dict_binary[mfd_dict_binary[constant.YEAR] == mfd_year]
for c in all_models:
    if 'FINEGRAINED' in binary_fine_grained:
        c.fit_bootstrap(reduced_mfd_dict)
    else:
        c.fit_bootstrap(reduced_mfd_dict_binary)
    for i,word in enumerate(test_words):
        if 'FINEGRAINED' in binary_fine_grained:
            word_df = test_df[test_df[constant.WORD] == word]
            mean, lower_bound, upper_bound = c.predict_proba_bootstrap(word_df[constant.VECTOR].values)
            c.fit(reduced_mfd_dict)
            all_predictions = c.predict_proba(word_df[constant.VECTOR])
            for cat in mfd_dict[constant.CATEGORY].unique():
                cat_prediction = [x[cat] for x in all_predictions]
                if '-' in cat:
                    plt.plot(word_df[constant.YEAR].values, cat_prediction, label=cat, ls='--', color=constant.get_colour(cat))
                else:
                    plt.plot(word_df[constant.YEAR].values, cat_prediction, label=cat, color=constant.get_colour(cat))
            plt.legend()
            plt.ylim(0.08,0.13)
            plt.hlines(0.5, min(word_df[constant.YEAR].values), max(word_df[constant.YEAR].values), colors='grey')
            plt.savefig(os.path.join(constant.TEMP_DATA_DIR,'images','category',word+'.png'))
            plt.clf()
        else:
            c.fit(reduced_mfd_dict_binary)
            word_df = test_df[test_df[constant.WORD] == word]
            all_predictions = c.predict_proba(word_df[constant.VECTOR])
            cat_prediction = [x['+'] for x in all_predictions]
            plt.title('{} {} Plot'.format(word, c.name))
            plt.ylabel('Category Score')
            plt.xlabel('Years')
            if plot_separate:
                plt.plot(word_df[constant.YEAR].values, cat_prediction, label='Moral Polarity', linewidth=2.0)
                plt.legend()
                plt.ylim(0.42,0.58)
                plt.hlines(0.5, min(word_df[constant.YEAR].values), max(word_df[constant.YEAR].values), colors='grey')
                plt.title('{} {}'.format(c.name, word))
                plt.savefig(os.path.join(constant.TEMP_DATA_DIR,'images','binaryseparate',word+'.png'))
                plt.clf()
            else:
                plt.plot(word_df[constant.YEAR].values, cat_prediction, label=word, linewidth=2.0, color=hsv(float(i)/(len(test_words)-1)))
                plt.legend()
    if not plot_separate:
        plt.hlines(0.5, min(test_df[constant.YEAR].values), max(test_df[constant.YEAR].values), colors='grey')
        plt.show()
        plt.savefig(os.path.join(constant.TEMP_DATA_DIR,'images','all_words.png'))

    