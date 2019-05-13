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
from scipy import stats

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

def log_odds(list_prob, neg_prob=None):
    if neg_prob is None:
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


def plot_binary(word, mean_line, head_word, years, btstrap, binary_fine_grained, ylim1, ylim2, xlim1,
                lower_bound=None, upper_bound=None):
    if binary_fine_grained == 'BINARY':
        cat, dir, color, ylabel, lbound_label, ubound_label = '+', 'binaryseparate', 'salmon', \
                                                  'Moral polarity log odds ratio', 'Negative', 'Positive'
    else:
        cat, dir, color, ylabel, lbound_label, ubound_label = '1', 'null', 'dodgerblue', 'Moral relevance log odds ratio', \
                                                              'Irrelevant', 'Relevant'
    cat_prediction, cat_prediction_l, cat_prediction_u \
        = [x[cat] for x in mean_line], \
          [x[cat] for x in lower_bound] if btstrap else None, \
          [x[cat] for x in upper_bound] if btstrap else None
    cat_prediction, cat_prediction_l, cat_prediction_u = log_odds(cat_prediction), \
                                                         log_odds(cat_prediction_l) if btstrap else None,\
                                                         log_odds(cat_prediction_u) if btstrap else None
    plt.ylim(ylim1, ylim2)
    slope,intercept,_,p_val,_ = stats.linregress(years, cat_prediction)
    plt.text(xlim1, ylim2, ubound_label, horizontalalignment='left',
             bbox=dict(boxstyle='square', ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    plt.text(xlim1, ylim1, lbound_label, horizontalalignment='left',
             bbox=dict(boxstyle='square', ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    plt.ylabel(ylabel)
    plt.xlabel('Time')
    plt.title('{} - {}'.format(' '.join(ylabel.split()[:-1]), word))
    if head_word == word:
        color, alpha, linewidth = color, 1, 4.0
    else:
        color, alpha, linewidth = color, 0.4, 2.0
    plt.plot(years,cat_prediction,color=color,linewidth=linewidth,label='Slope: {0:.3}, P Value: {1:.3}'.format(slope, p_val),alpha=alpha)
    if btstrap:
        plt.fill_between(years, cat_prediction_l, cat_prediction_u, alpha=0.2, color=color)
    plt.axhline(y=0, color='grey')
    plt.legend()
    plt.savefig(os.path.join(constant.TEMP_DATA_DIR, 'images', dir, head_word + '.png'),
                bbox_inches='tight', format='png')
    plt.savefig(os.path.join(constant.TEMP_DATA_DIR, 'images', dir, head_word + '.pdf'),
                bbox_inches='tight',format='pdf')
    plt.clf()
    print(os.path.join(constant.TEMP_DATA_DIR, 'images', dir, head_word + '.pdf'))

def plot_category_hinton(word, categories, preds, years, max_weight, min_weight):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    for y, cat in enumerate(categories):
        ax.text(-2, y, cat, horizontalalignment='right', verticalalignment='center')
        for x,pred_dict in enumerate(preds):
            pred = pred_dict[cat]
            color = constant.get_colour(cat)
            slope = 0.5/(max_weight-min_weight)
            b = 0.5-(slope*max_weight)
            size = pred*slope+b
            if size == 0:
                size = 0.01
            if y == 0 and x%2 == 0:
                ax.text(x - size / 2, -2, years[x], horizontalalignment='center')
            circle = plt.Circle([x - size / 2, y - size / 2], size, facecolor=color, edgecolor=color)
            ax.add_patch(circle)
    ax.autoscale_view()
    ax.set_xticklabels(categories)
    plt.title(word)
    plt.savefig(os.path.join(constant.TEMP_DATA_DIR, 'images', 'category', word + '.pdf'), format='pdf')
    plt.savefig(os.path.join(constant.TEMP_DATA_DIR, 'images', 'category', word + '.png'), format='png')
    plt.clf()

def plot_category(f, word, mfd_dict, mean_line, years, lower_bound, upper_bound):
    min_year, max_year = min(mfd_dict[constant.YEAR].unique()), max(mfd_dict[constant.YEAR].unique())
    ax1 = f.add_axes([0.1, 0.5, 0.8, 0.4], title=word, xlim=[min_year, max_year],
                     xticklabels=[], ylim=[0.09, 0.11], ylabel='Positive Probability')
    ax2 = f.add_axes([0.1, 0.1, 0.8, 0.4], ylabel='Negative Probability',
                     ylim=[0.11, 0.09], xlim=[min_year, max_year], xlabel='Time')
    for cat in mfd_dict[constant.CATEGORY].unique():
        color = constant.get_colour(cat)
        cat_prediction, cat_prediction_l, cat_prediction_u \
            = [x[cat] for x in mean_line], [x[cat] for x in lower_bound] if lower_bound is not None else None, \
              [x[cat] for x in upper_bound] if upper_bound is not None else None
        if '-' in cat:
            ax2.plot(years, cat_prediction, label=cat[:-1], ls=constant.get_linestyle(cat), color=color)
            ax = ax2
        else:
            ax1.plot(years, cat_prediction, label=cat[:-1], ls=constant.get_linestyle(cat), color=color)
            ax = ax1
        if lower_bound is not None:
            ax.fill_between(years, cat_prediction_l, cat_prediction_u, color=color, alpha=0.2)
    ax1.legend(loc='upper right')
    plt.axhline(y=0, color='grey')
    plt.savefig(os.path.join(constant.TEMP_DATA_DIR, 'images', 'category', word + '.pdf'), format='pdf')
    plt.savefig(os.path.join(constant.TEMP_DATA_DIR, 'images', 'category', word + '.png'), format='png')
    plt.clf()

def plot_category_log_odds(word, mfd_dict, mean_line, years, lower_bound, upper_bound):
    min_year, max_year = min(mfd_dict[constant.YEAR].unique()), max(mfd_dict[constant.YEAR].unique())
    five_cats = set([x for x in mfd_dict[constant.CATEGORY].unique()])
    for cat in five_cats:
        color = constant.get_colour(cat)
        cat_prediction_pos, cat_prediction_l_pos, cat_prediction_u_pos \
            = [x[cat+'+'] for x in mean_line], [x[cat+'+'] for x in lower_bound] if lower_bound is not None else None, \
              [x[cat+'+'] for x in upper_bound] if upper_bound is not None else None
        cat_prediction_neg, cat_prediction_l_neg, cat_prediction_u_neg \
            = [x[cat + '-'] for x in mean_line], [x[cat + '-'] for x in lower_bound] if lower_bound is not None else None, \
              [x[cat + '-'] for x in upper_bound] if upper_bound is not None else None
        cat_prediction, cat_prediction_l, cat_prediction_u = cat_prediction, \
                                                             log_odds(cat_prediction_l_pos, cat_prediction_u_neg), \
                                                             log_odds(cat_prediction_u_pos, cat_prediction_u_neg)
        plt.plot(years, cat_prediction, label=cat, color=color)
        if lower_bound is not None:
            plt.fill_between(years, cat_prediction_l, cat_prediction_u, color=color, alpha=0.2)
    plt.legend(loc='upper right')
    plt.ylim(-0.1,0.1)
    plt.xlim(min_year, max_year)
    plt.axhline(y=0,color='grey')
    plt.savefig(os.path.join(constant.TEMP_DATA_DIR, 'images', 'category', word + '.eps'), format='eps')
    plt.clf()

def set_ylim(nyt_corpus, binary_fine_grained):
    if nyt_corpus == 'NGRAM':
        if binary_fine_grained == 'BINARY':
            return -0.08,0.08
        elif binary_fine_grained == 'NULL':
            return -0.08,0.08
        else:
            return 0.09,0.11
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
    f = plt.figure(figsize=(15, 6))
    ylim1, ylim2 = set_ylim(nyt_corpus, binary_fine_grained)
    all_fitted_models = [set_model(c, btstrap, binary_fine_grained, mfd_dict, mfd_dict_binary, mfd_dict_null) for c in all_models]
    plt.rcParams.update({'font.size': 15})
    all_years = sorted(test_df[constant.YEAR].unique())
    for model_list in all_fitted_models:
        results = {}
        max_weight = -1
        min_weight = 1
        for i,word in enumerate(test_df[constant.WORD].unique()):
            word_df = test_df[test_df[constant.WORD] == word].sort_values(by=[constant.YEAR])
            years = word_df[constant.YEAR].values
            mean_line, lower_bound, upper_bound = models.models_predictions(model_list, word_df, btstrap)
            results[word] = [years, mean_line, lower_bound, upper_bound]
            # max_weight = max(max_weight, max(max(x.values()) for x in mean_line))
            # min_weight = min(min_weight, min(min(x.values()) for x in mean_line))
        for word in results:
            years, mean_line, lower_bound, upper_bound = results[word]
            if 'FINEGRAINED' == binary_fine_grained:
                max_weight = max(max(x.values()) for x in mean_line)
                min_weight = min(min(x.values()) for x in mean_line)
                categories = sorted(list(mean_line[0].keys()))
                plot_category_hinton(word, categories, mean_line, years, max_weight, min_weight)
                # plot_category(f, word, mfd_dict, mean_line, years, lower_bound, upper_bound)
            elif 'BINARY' == binary_fine_grained or 'NULL' == binary_fine_grained:
                plt.xlim(min(all_years), max(all_years))
                plot_binary(word, mean_line, word, years, btstrap, binary_fine_grained,ylim1,ylim2,min(all_years),
                            lower_bound, upper_bound)


# # Params
# binary_fine_grained = ['BINARY', 'FINEGRAINED', 'NULL'][0]
# plot_separate = True # Only has effect in the binary case
# mfd_year = 1990
# load = True
# years = constant.ALL_YEARS
# nyt_corpus = ['NYT', 'NGRAM', 'FICTION'][0]
    