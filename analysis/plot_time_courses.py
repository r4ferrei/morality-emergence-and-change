import pickle
import constant
import embeddings
import os
import pandas as pd
import models
from models import CentroidModel
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

def load_test_df(test_words, years, emb_dict_all=None, reload=False):
    if reload:
        test_df = []
        for year in emb_dict_all.keys():
            yr_emb_dict = emb_dict_all[year]
            for i,word in enumerate(test_words):
                embedding = embeddings.get_sent_embed(yr_emb_dict, word)
                if embedding is not None:
                    test_df.append({constant.YEAR:year,constant.WORD:word,constant.VECTOR:embedding})
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
        all_predictions = model.predict_proba(word_df[constant.VECTOR])
        cons_predictions.append(all_predictions)
    lower_bound = []
    upper_bound = []
    for i,year in word_df[constant.YEAR].values:
        year_predictions = [x[i] for x in cons_predictions]
        lower_bound.append({cat: min([x[cat] for x in year_predictions]) for cat in categories})
        upper_bound.append({cat: max([x[cat] for x in year_predictions]) for cat in categories})
    return mean_predictions, lower_bound, upper_bound

def log_odds(list_prob):
    neg_prob = np.subtract(1, list_prob)
    return np.log(np.divide(list_prob,neg_prob))

def set_model(model_lambda, mfd_dict, mfd_dict_binary, mfd_dict_null):
    all_models = {}
    mfd_dict.drop(columns=[constant.VECTOR]).to_csv('something.csv')
    for year in mfd_dict[constant.YEAR].unique():
        c = model_lambda()
        if 'FINEGRAINED' == binary_fine_grained:
            reduced_mfd_dict = mfd_dict[mfd_dict[constant.YEAR] == year]
            c.fit_bootstrap(reduced_mfd_dict)
        elif 'BINARY' == binary_fine_grained:
            reduced_mfd_dict_binary = mfd_dict_binary[mfd_dict_binary[constant.YEAR] == year]
            c.fit_bootstrap(reduced_mfd_dict_binary)
        else:
            reduced_mfd_dict_null = mfd_dict_null[mfd_dict_null[constant.YEAR] == year]
            c.fit_bootstrap(reduced_mfd_dict_null)
        all_models[year] = c
    return all_models

def bootstrap_all_models(model_list, word_df):
    mean_line_agg, lower_bound_agg, upper_bound_agg = [], [], []
    X =  word_df[constant.VECTOR].values
    years = word_df[constant.YEAR].values.tolist()
    for i,year in enumerate(years):
        c = model_list[year]
        mean_line, lower_bound, upper_bound = c.predict_proba_bootstrap([X[i]])
        mean_line_agg.append(mean_line[0])
        lower_bound_agg.append(lower_bound[0])
        upper_bound_agg.append(upper_bound[0])
    return mean_line_agg, lower_bound_agg, upper_bound_agg 


def set_plot(test_words, binary_fine_grained, plot_separate, mfd_year, load, years, nyt_corpus, all_models, plot_extra=None):
    # Loading data
    emb_dict_all = None
    if load:
        if nyt_corpus == 'NGRAM':
            emb_dict_all,_ = embeddings.load_all(dir=constant.SGNS_DIR, years=constant.ALL_YEARS)
        elif nyt_corpus == 'FICTION':
            emb_dict_all,_ = embeddings.load_all_fiction(dir='D:/WordEmbeddings/kim')
        else:
            emb_dict_all,_ = embeddings.load_all_nyt(dir=constant.SGNS_NYT_DIR)
    test_df = load_test_df(test_words, years, emb_dict_all=emb_dict_all, reload=load)
    mfd_dict = models.load_mfd_df(emb_dict_all, load)
    mfd_dict_binary = models.load_mfd_df_binary(emb_dict=emb_dict_all, reload=load)
    mfd_dict_null = models.load_mfd_df_neutral(emb_dict=emb_dict_all, reload=load)
    # Generate each plot
    hsv = plt.get_cmap('nipy_spectral')
    plt.figure(figsize=(10, 6))

    all_fitted_models = [set_model(c, mfd_dict, mfd_dict_binary, mfd_dict_null) for c in all_models]
    for model_list in all_fitted_models:
        for i,word in enumerate(test_words):
            word_df = test_df[test_df[constant.WORD] == word]
            years = word_df[constant.YEAR].values
            mean_line, lower_bound, upper_bound = bootstrap_all_models(model_list, word_df)
            if plot_extra is not None:
                plot_extra(word)
            if 'FINEGRAINED' == binary_fine_grained:
                f = plt.figure(figsize=(15,10))
                ax1 = f.add_axes([0.1, 0.5, 0.8, 0.4], title = word,
                    xticklabels=[], ylim=[0.1,0.12], ylabel='Positive Probability')
                ax2 = f.add_axes([0.1, 0.1, 0.8, 0.4], ylabel = 'Negative Probability',
                    ylim=[0.12,0.1], xlabel='Years')
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
                if binary_fine_grained == 'BINARY':
                    cat, dir, color, ylabel, lbound, ubound = '+', 'binaryseparate', 'black', 'Moral Sentiment Score', -0.2, 0.2
                else:
                    cat, dir, color, ylabel, lbound, ubound = '1', 'null', 'violet', 'Moral Relevance Score', -0.3, 0.3
                cat_prediction,cat_prediction_l,cat_prediction_u\
                        = [x[cat] for x in mean_line],[x[cat] for x in lower_bound],[x[cat] for x in upper_bound]
                cat_prediction, cat_prediction_l, cat_prediction_u = log_odds(cat_prediction), \
                                                                    log_odds(cat_prediction_l), log_odds(cat_prediction_u)
                plt.title('{} Plot'.format(word))
                plt.ylabel(ylabel)
                plt.xlabel('Years')
                if plot_separate and not word_df.empty:
                    name = model_list[years[0]].name
                    plt.plot(word_df[constant.YEAR].values, cat_prediction, linewidth=2.0, color=color)
                    plt.fill_between(word_df[constant.YEAR].values, cat_prediction_l, cat_prediction_u, alpha=0.2, color=color)
                    # plt.ylim(lbound,ubound)
                    plt.axhline(y=0, color='grey')
                    plt.legend()
                    plt.title('{} {}'.format(name, word))
                    plt.savefig(os.path.join(constant.TEMP_DATA_DIR,'images',dir,word+'.png'))
                    print(os.path.join(constant.TEMP_DATA_DIR,'images',dir,word+'.png'))
                    plt.clf()
                elif not word_df.empty:
                    plt.plot(word_df[constant.YEAR].values, cat_prediction, label=word, linewidth=2.0, color=hsv(float(i)/(len(test_words)-1)))
                    plt.legend()
                else:
                    plt.clf()
        if not plot_separate:
            plt.hlines(0.5, min(test_df[constant.YEAR].values), max(test_df[constant.YEAR].values), colors='grey')
            plt.show()
            plt.savefig(os.path.join(constant.TEMP_DATA_DIR,'images','all_words.png'))


# Params
binary_fine_grained = ['BINARY', 'FINEGRAINED', 'NULL'][0]
plot_separate = True # Only has effect in the binary case
mfd_year = 1990
load = True
years = constant.ALL_YEARS
nyt_corpus = ['NYT', 'NGRAM', 'FICTION'][0]


    