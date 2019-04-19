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

def set_model(model_lambda, btstrap, binary_fine_grained, mfd_dict, mfd_dict_binary, mfd_dict_null):
    all_models = {}
    mfd_dict.drop(columns=[constant.VECTOR]).to_csv('something.csv')
    for year in mfd_dict[constant.YEAR].unique():
        max_year = max(mfd_dict[constant.YEAR].unique())
        c = model_lambda()
        if 'FINEGRAINED' == binary_fine_grained:
            reduced_mfd_dict = mfd_dict[mfd_dict[constant.YEAR] == max_year]
        elif 'BINARY' == binary_fine_grained:
            reduced_mfd_dict = mfd_dict_binary[mfd_dict_binary[constant.YEAR] == max_year]
        else:
            reduced_mfd_dict = mfd_dict_null[mfd_dict_null[constant.YEAR] == max_year]
        if btstrap:
            c.fit_bootstrap(reduced_mfd_dict)
        else:
            c.fit(reduced_mfd_dict)
        all_models[year] = c
    return all_models

def plot_binary(mean_line, head_word, word_df, model_list, years, btstrap, binary_fine_grained, ylim1, ylim2,
                lower_bound=None, upper_bound=None):
    if binary_fine_grained == 'BINARY':
        cat, dir, color, ylabel, lbound_label, ubound_label = '+', 'binaryseparate', None, \
                                                  'Moral Sentiment Score', 'Negative', 'Positive'
    else:
        cat, dir, color, ylabel, lbound_label, ubound_label = '1', 'null', None, 'Moral Relevance Score', 'Irrelevant', 'Relevant'
    cat_prediction, cat_prediction_l, cat_prediction_u \
        = [x[cat] for x in mean_line], \
          [x[cat] for x in lower_bound] if btstrap else None, \
          [x[cat] for x in upper_bound] if btstrap else None
    cat_prediction, cat_prediction_l, cat_prediction_u = log_odds(cat_prediction), \
                                                         log_odds(cat_prediction_l) if btstrap else None,\
                                                         log_odds(cat_prediction_u) if btstrap else None
    plt.ylim(ylim1, ylim2)
    plt.text(min(word_df[constant.YEAR].values), ylim2, ubound_label, horizontalalignment='center', bbox=dict(boxstyle='square', ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    plt.text(min(word_df[constant.YEAR].values), ylim1, lbound_label, horizontalalignment='center', bbox=dict(boxstyle='square', ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    plt.title('{} Plot'.format(head_word))
    plt.ylabel(ylabel)
    plt.xlabel('Years')
    if not word_df.empty:
        name = model_list[years[0]].name
        word = word_df[constant.WORD].values.tolist()[0]
        if head_word == word:
            color, alpha, linewidth = 'black', 1, 4.0
        else:
            color, alpha, linewidth = None, 0.4, 2.0
        plt.plot(word_df[constant.YEAR].values,cat_prediction,color=color,linewidth=linewidth,label=word,alpha=alpha)
        if btstrap:
            plt.fill_between(word_df[constant.YEAR].values, cat_prediction_l, cat_prediction_u, alpha=0.2, color=color)
        plt.axhline(y=0, color='grey')
        plt.legend()
        plt.title('{} {}'.format(name, head_word))
        plt.savefig(os.path.join(constant.TEMP_DATA_DIR, 'images', dir, head_word + '.png'))
        print(os.path.join(constant.TEMP_DATA_DIR, 'images', dir, head_word + '.png'))
    else:
        plt.clf()

def plot_category(word, mfd_dict, mean_line, years, ylim1, ylim2, lower_bound, upper_bound):
    f = plt.figure(figsize=(15, 10))
    ax1 = f.add_axes([0.1, 0.5, 0.8, 0.4], title=word,
                     xticklabels=[], ylim=[0.1, 0.12], ylabel='Positive Probability')
    ax2 = f.add_axes([0.1, 0.1, 0.8, 0.4], ylabel='Negative Probability',
                     ylim=[0.12, 0.1], xlabel='Years')
    for cat in mfd_dict[constant.CATEGORY].unique():
        color = constant.get_colour(cat)
        cat_prediction, cat_prediction_l, cat_prediction_u \
            = [x[cat] for x in mean_line], [x[cat] for x in lower_bound] if lower_bound is not None else None, \
              [x[cat] for x in upper_bound] if upper_bound is not None else None
        if '-' in cat:
            ax2.plot(years, cat_prediction, label=cat[:-1], ls='--', color=color)
            ax = ax2
        else:
            ax1.plot(years, cat_prediction, label=cat, color=color)
            ax = ax1
        if lower_bound is not None:
            ax.fill_between(years, cat_prediction_l, cat_prediction_u, color=color, alpha=0.2)
    ax1.legend(loc='upper right')
    plt.savefig(os.path.join(constant.TEMP_DATA_DIR, 'images', 'category', word + '.png'))
    plt.clf()

def set_ylim(nyt_corpus, binary_fine_grained):
    if nyt_corpus == 'NGRAM':
        if binary_fine_grained == 'BINARY':
            return -0.12,0.12
        elif binary_fine_grained == 'NULL':
            return -0.08,0.08
    elif nyt_corpus == 'NYT':
        if binary_fine_grained == 'BINARY':
            return -8,8
        return -6,6
    elif nyt_corpus == 'FICTION':
        if binary_fine_grained == 'BINARY' or binary_fine_grained == 'NULL':
            return -0.4,0.4
    return None, None


def set_plot(binary_fine_grained, btstrap, load, all_models, emb_dict_all, load_test_df, nyt_corpus, plot_extra=None):
    # Loading data
    test_df = load_test_df(emb_dict_all=emb_dict_all, reload=load)
    mfd_dict = models.load_mfd_df(emb_dict_all, load)
    mfd_dict_binary = models.load_mfd_df_binary(emb_dict=emb_dict_all, reload=load)
    mfd_dict_null = models.load_mfd_df_neutral(emb_dict=emb_dict_all, reload=load)
    plt.figure(figsize=(10, 6))
    ylim1, ylim2 = set_ylim(nyt_corpus, binary_fine_grained)
    all_fitted_models = [set_model(c, btstrap, binary_fine_grained, mfd_dict, mfd_dict_binary, mfd_dict_null) for c in all_models]
    for model_list in all_fitted_models:
        head_word = None
        for i,word in enumerate(test_df[constant.WORD].unique()):
            word_df = test_df[test_df[constant.WORD] == word]
            word_df = word_df.sort_values(by=[constant.YEAR])
            years = word_df[constant.YEAR].values
            if constant.CONCEPT in word_df and word_df[constant.CONCEPT].values.tolist()[0] != word and 'FINEGRAINED' == binary_fine_grained:
                continue
            mean_line, lower_bound, upper_bound = models.models_predictions(model_list, word_df, btstrap)
            if plot_extra is not None:
                plot_extra(word, model_list)
            if constant.CONCEPT not in test_df:
                head_word = word
                plt.clf()
            if constant.CONCEPT in word_df:
                head_word_i = word_df[constant.CONCEPT].values.tolist()[0]
                if head_word is None or head_word_i != head_word:
                    plt.clf()
                head_word = head_word_i
            if head_word == word and 'FINEGRAINED' == binary_fine_grained:
                plot_category(word, mfd_dict, mean_line, years, ylim1, ylim2, lower_bound, upper_bound)
            elif 'BINARY' == binary_fine_grained or 'NULL' == binary_fine_grained:
                plot_binary(mean_line, head_word, word_df, model_list, years, btstrap, binary_fine_grained,ylim1,ylim2,
                            lower_bound, upper_bound)


# # Params
# binary_fine_grained = ['BINARY', 'FINEGRAINED', 'NULL'][0]
# plot_separate = True # Only has effect in the binary case
# mfd_year = 1990
# load = True
# years = constant.ALL_YEARS
# nyt_corpus = ['NYT', 'NGRAM', 'FICTION'][0]
    