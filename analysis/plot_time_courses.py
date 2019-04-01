import pickle
import constant
import embeddings
import os
import pandas as pd
import models
from models import CentroidModel
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
binary_fine_grained = np.array(['BINARY', 'FINEGRAINED'])[[True, False]]
plot_separate = True # Only has effect in the binary case
mfd_year = 1990
load = True
years = constant.ALL_YEARS
test_words = ['abortion', 'gay', 'donation', 'charity', 'war', 'genocide', 'feminism', 'slavery', 'racism', 'education',
    'immigration', 'machine', 'computer', 'robot', 'automation', 'sexism', 'fat', 'skinny']
# test_words = ['slavery', 'feminism', 'racism', 'abortion', 'automation', 'computer', 'diversity',
#               'education', 'electric', 'engineer', 'environment', 'feminism', 'immigration',
#               'information', 'machine', 'mechanic', 'nazism', 'phone', 'privacy', 'racism',
#               'religion', 'robotics', 'sexism', 'technology']
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
plt.figure(figsize=(10, 6))
for c in all_models:
    if 'FINEGRAINED' in binary_fine_grained:
        c.fit_bootstrap(reduced_mfd_dict)
    else:
        c.fit_bootstrap(reduced_mfd_dict_binary)
    for i,word in enumerate(test_words):

        word_df = test_df[test_df[constant.WORD] == word]
        years = word_df[constant.YEAR].values
        mean_line, lower_bound, upper_bound = c.predict_proba_bootstrap(word_df[constant.VECTOR].values)

        if 'FINEGRAINED' in binary_fine_grained:
            f = plt.figure(figsize=(15,10))
            ax1 = f.add_axes([0.1, 0.5, 0.8, 0.4], title = word,
                   xticklabels=[], ylim=[0.1,0.13], ylabel='Positive Probability')
            ax2 = f.add_axes([0.1, 0.1, 0.8, 0.4], ylabel = 'Negative Probability',
                   ylim=[0.13,0.1], xlabel='Years')
            for cat in mfd_dict[constant.CATEGORY].unique():
                color = constant.get_colour(cat)
                cat_prediction,cat_prediction_l,cat_prediction_u\
                    = [x[cat] for x in mean_line],[x[cat] for x in lower_bound],[x[cat] for x in upper_bound]
                if '-' in cat:
                    # cat_prediction_l,cat_prediction_u = cat_prediction_l, \
                    #                                     cat_prediction_u
                    ax2.plot(years, cat_prediction, label=cat[:-1], ls='--', color=color)
                    ax = ax2
                else:
                    ax1.plot(years, cat_prediction, label=cat, color=color)
                    ax = ax1
                ax.fill_between(years,cat_prediction_l,cat_prediction_u,color=color,alpha=0.2)
            ax1.legend(loc='upper right')
                # ax.hlines(0.1, min(word_df[constant.YEAR].values), max(word_df[constant.YEAR].values), colors='grey')
            plt.savefig(os.path.join(constant.TEMP_DATA_DIR,'images','category',word+'.png'))
            plt.clf()
        else:
            cat_prediction,cat_prediction_l,cat_prediction_u\
                    = [x['+'] for x in mean_line],[x['+'] for x in lower_bound],[x['+'] for x in upper_bound]
            plt.title('{} {} Plot'.format(word, c.name))
            plt.ylabel('Category Score')
            plt.xlabel('Years')
            if plot_separate:
                plt.plot(word_df[constant.YEAR].values, cat_prediction, label='Moral Polarity', linewidth=2.0)
                plt.fill_between(word_df[constant.YEAR].values, cat_prediction_l, cat_prediction_u, alpha=0.2)
                plt.legend()
                plt.ylim(0.485,0.515)
                if word_df.shape[0] > 0:
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

    