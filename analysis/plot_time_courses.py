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


def plot_binary(mean_line, head_word, word_df, model_list, btstrap, binary_fine_grained, ylim1, ylim2, xlim1, plot_extra,
                lower_bound=None, upper_bound=None):
    if binary_fine_grained == 'BINARY':
        cat, dir, color, ylabel, lbound_label, ubound_label = '+', 'binaryseparate', 'salmon', \
                                                  'Moral polarity log odds ratio', 'Negative', 'Positive'
    else:
        cat, dir, color, ylabel, lbound_label, ubound_label = '1', 'null', 'dodgerblue', 'Moral relevance log odds ratio', \
                                                              'Irrelevant', 'Relevant'
    years = word_df[constant.YEAR].values
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
    if not word_df.empty:
        name = model_list[years[0]].name
        word = word_df[constant.WORD].values.tolist()[0]
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
        plt.savefig(os.path.join(constant.TEMP_DATA_DIR, 'images', dir, head_word + '.png'),bbox_inches='tight')
        print(os.path.join(constant.TEMP_DATA_DIR, 'images', dir, head_word + '.png'))
    else:
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
            ax2.plot(years, cat_prediction, label=cat[:-1], ls='--', color=color)
            ax = ax2
        else:
            ax1.plot(years, cat_prediction, label=cat[:-1], color=color)
            ax = ax1
        if lower_bound is not None:
            ax.fill_between(years, cat_prediction_l, cat_prediction_u, color=color, alpha=0.2)
    ax1.legend(loc='upper right')
    plt.axhline(y=0, color='grey')
    plt.savefig(os.path.join(constant.TEMP_DATA_DIR, 'images', 'category', word + '.png'))
    plt.clf()

def plot_category_log_odds(word, mfd_dict, mean_line, years, lower_bound, upper_bound):
    min_year, max_year = min(mfd_dict[constant.YEAR].unique()), max(mfd_dict[constant.YEAR].unique())
    five_cats = set([x[:-1] for x in mfd_dict[constant.CATEGORY].unique()])
    for cat in five_cats:
        color = constant.get_colour(cat)
        cat_prediction_pos, cat_prediction_l_pos, cat_prediction_u_pos \
            = [x[cat+'+'] for x in mean_line], [x[cat+'+'] for x in lower_bound] if lower_bound is not None else None, \
              [x[cat+'+'] for x in upper_bound] if upper_bound is not None else None
        cat_prediction_neg, cat_prediction_l_neg, cat_prediction_u_neg \
            = [x[cat + '-'] for x in mean_line], [x[cat + '-'] for x in lower_bound] if lower_bound is not None else None, \
              [x[cat + '-'] for x in upper_bound] if upper_bound is not None else None
        cat_prediction, cat_prediction_l, cat_prediction_u = log_odds(cat_prediction_pos, cat_prediction_neg), \
                                                             log_odds(cat_prediction_l_pos, cat_prediction_u_neg), \
                                                             log_odds(cat_prediction_u_pos, cat_prediction_u_neg)
        plt.plot(years, cat_prediction, label=cat, color=color)
        if lower_bound is not None:
            plt.fill_between(years, cat_prediction_l, cat_prediction_u, color=color, alpha=0.2)
    plt.legend(loc='upper right')
    plt.ylim(-0.1,0.1)
    plt.xlim(min_year, max_year)
    plt.axhline(y=0,color='grey')
    plt.savefig(os.path.join(constant.TEMP_DATA_DIR, 'images', 'category', word + '.png'))
    plt.clf()


def plot_category_heat(word, mean_line, all_years, ylim1, ylim2):
    all_categories = mean_line[0].keys()
    pos_categories = sorted(x for x in all_categories if '+' in x)
    neg_categories = sorted(x for x in all_categories if '-' in x)
    for i,categories in enumerate([pos_categories, neg_categories]):
        plt.subplot(2, 1, i + 1)
        if i == 0:
            plt.ylabel('Positive categories')
            plt.title('Moral category probability - {}'.format(word))
            polarity, cmap = 'positive', 'afmhot'
        else:
            plt.ylabel('Negative categories')
            plt.xlabel('Time')
            polarity, cmap = 'negative', 'bone'
        m = []
        for cat in categories:
            cat_prediction = [x[cat] for x in mean_line]
            m.append([0]*(len(all_years)-len(cat_prediction))+cat_prediction)
        m = np.array(m)
        plt.imshow(m, cmap=cmap, vmin=ylim1, vmax=ylim2)
        plt.yticks(np.arange(len(categories)), categories)
        year_ticks = np.arange(len(all_years), step=4)
        plt.xticks(year_ticks, [all_years[i] for i in year_ticks])
        plt.colorbar()
    plt.savefig(os.path.join(constant.TEMP_DATA_DIR, 'images', 'category', word + '.png'))
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
        head_word = None
        for i,word in enumerate(test_df[constant.WORD].unique()):
            word_df = test_df[test_df[constant.WORD] == word]
            years = word_df[constant.YEAR].values
            word_df = word_df.sort_values(by=[constant.YEAR])
            if constant.CONCEPT in word_df and word_df[constant.CONCEPT].values.tolist()[0] != word and 'FINEGRAINED' == binary_fine_grained:
                continue
            mean_line, lower_bound, upper_bound = models.models_predictions(model_list, word_df, btstrap)
            if constant.CONCEPT not in test_df:
                head_word = word
                plt.clf()
            if constant.CONCEPT in word_df:
                head_word_i = word_df[constant.CONCEPT].values.tolist()[0]
                if head_word is None or head_word_i != head_word:
                    plt.clf()
                head_word = head_word_i
            if head_word == word and 'FINEGRAINED' == binary_fine_grained:
                plot_category_log_odds(word, mfd_dict, mean_line, years, lower_bound, upper_bound)
                # plot_category_heat(word, mean_line, all_years, ylim1, ylim2)
            elif 'BINARY' == binary_fine_grained or 'NULL' == binary_fine_grained:
                plt.xlim(min(all_years), max(all_years))
                plot_binary(mean_line, head_word, word_df, model_list, btstrap, binary_fine_grained,ylim1,ylim2,min(all_years),
                            plot_extra, lower_bound, upper_bound)


# # Params
# binary_fine_grained = ['BINARY', 'FINEGRAINED', 'NULL'][0]
# plot_separate = True # Only has effect in the binary case
# mfd_year = 1990
# load = True
# years = constant.ALL_YEARS
# nyt_corpus = ['NYT', 'NGRAM', 'FICTION'][0]
    