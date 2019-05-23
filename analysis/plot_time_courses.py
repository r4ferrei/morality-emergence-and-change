import constant
import models
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import embeddings
import argparse
from scipy import stats

"""
Visualization code to reproduce plots displaying the different moral sentiment tiers
"""

def log_odds(list_prob, neg_prob=None):
    if neg_prob is None:
        neg_prob = np.subtract(1, list_prob)
    return np.log(np.divide(list_prob, neg_prob))


def set_model(model_lambda, btstrap, binary_fine_grained, mfd_dict, mfd_dict_binary, mfd_dict_null):
    """
    Fit all relevant models

    :param model_lambda: Function (accepting no parameters) that returns some model
    :param btstrap: Boolean that specifies whether to set up infrastructure for bootstrap
    :param binary_fine_grained: String that specifies the type of plot
    :param mfd_dict: Pandas Dataframe with training data for fine-grained analysis
    :param mfd_dict_binary: Pandas Dataframe with training data for binary analysis
    :param mfd_dict_null: Pandas Dataframe with training data for neutral analysis
    :return:
    """
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


def plot_binary(word, mean_line, head_word, years, btstrap, binary_fine_grained, ylim1, ylim2, xlim1, yticks,
                lower_bound=None, upper_bound=None):
    if binary_fine_grained == 'BINARY':
        cat, dir, color, ylabel, lbound_label, ubound_label = '+', 'binaryseparate', 'salmon', \
                                                              'Polarity', 'Negative', 'Positive'
    else:
        cat, dir, color, ylabel, lbound_label, ubound_label = '1', 'null', 'dodgerblue', 'Relevance', \
                                                              'Irrelevant', 'Relevant'
    cat_prediction, cat_prediction_l, cat_prediction_u \
        = [x[cat] for x in mean_line], \
          [x[cat] for x in lower_bound] if btstrap else None, \
          [x[cat] for x in upper_bound] if btstrap else None
    cat_prediction, cat_prediction_l, cat_prediction_u = log_odds(cat_prediction), \
                                                         log_odds(cat_prediction_l) if btstrap else None, \
                                                         log_odds(cat_prediction_u) if btstrap else None
    plt.ylim(ylim1, ylim2)
    slope, intercept, _, p_val, _ = stats.linregress(years, cat_prediction)
    plt.text(xlim1, ylim2, ubound_label, horizontalalignment='right',
             bbox=dict(boxstyle='square', ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    plt.text(xlim1, ylim1, lbound_label, horizontalalignment='right',
             bbox=dict(boxstyle='square', ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    plt.ylabel(ylabel, labelpad=37)
    if head_word == word:
        color, alpha, linewidth = color, 1, 4.0
    else:
        color, alpha, linewidth = color, 0.4, 2.0
    thresh = None
    if p_val < 0.05:
        thresh = 0.05
    elif p_val < 0.001:
        thresh = 0.001
    elif p_val < 0.0001:
        thresh = 0.0001
    if thresh is not None:
        label = 'Slope: {0:.2e}, p-value < {1:}'.format(slope, thresh)
    else:
        label = 'Slope: {0:.2e}, p-value = {1:.2e}'.format(slope, p_val)
    plt.plot(years, cat_prediction, color=color, linewidth=linewidth,
             label=label, alpha=alpha)
    if btstrap:
        plt.fill_between(years, cat_prediction_l, cat_prediction_u, alpha=0.2, color=color)
    plt.axhline(y=0, color='grey')
    plt.xticks([constant.ALL_YEARS[i] for i in list(yticks)])
    plt.legend(loc='upper right')


def plot_category_hinton(categories, preds, years_locs, max_weight, min_weight):
    """
    Plot category diagrams illustrating the fine-grained distribution of moral
    sentiment predictions

    :param categories: List of fine-grained categories
    :param preds: List prediction numbers for a given concept, over a list of years
    :param years_locs: Years for the prediction numbers
    :param max_weight: Max size to scale to
    :param min_weight: Min size to scale to
    :return:
    """
    ax = plt.gca()
    d = {'care': 'harm', 'authority': 'subversion', 'sanctity': 'degradation', 'loyalty': 'betrayal',
         'fairness': 'cheating'}
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_ylabel('Fine-grained', labelpad=80)
    for x in years_locs:
        ax.text(x - 1 / 2, -2, constant.ALL_YEARS[x], horizontalalignment='center')
    for y, cat in enumerate(categories):
        ax.text(-2, y, d[cat[:-1]] + cat[-1] if '-' in cat else cat, horizontalalignment='right',
                verticalalignment='center')
        for x, pred_dict in enumerate(preds):
            if pred_dict == None:
                continue
            pred = pred_dict[cat]
            color = constant.get_colour(cat)
            slope = 0.5 / (max_weight - min_weight)
            b = 0.5 - (slope * max_weight)
            size = pred * slope + b
            if size == 0:
                size = 0.01
            circle = plt.Circle([x - size / 2, y - size / 2], size, facecolor=color, edgecolor=color)
            ax.add_patch(circle)
    ax.autoscale_view()


def set_ylim(nyt_corpus, binary_fine_grained):
    if nyt_corpus == 'NGRAM':
        if binary_fine_grained == 'BINARY':
            return -0.08, 0.08
        elif binary_fine_grained == 'NULL':
            return -0.08, 0.08
        else:
            return 0.09, 0.11
    elif nyt_corpus == 'NYT':
        if binary_fine_grained == 'BINARY':
            return -8, 8
        return -6, 6
    elif nyt_corpus == 'FICTION':
        if binary_fine_grained == 'BINARY' or binary_fine_grained == 'NULL':
            return -0.4, 0.4
    return None, None

def load_test_df(emb_dict_all, test_words):
    test_df = []
    for year in emb_dict_all.keys():
        yr_emb_dict = emb_dict_all[year]
        for i,word in enumerate(test_words):
            embedding = embeddings.get_sent_embed(yr_emb_dict, word)
            if embedding is not None:
                test_df.append({constant.YEAR:year,constant.WORD:word,constant.VECTOR:embedding})
    return pd.DataFrame(test_df)

def run(btstrap, test_words, all_models, emb_dict_all, nyt_corpus):
    """
    Entry point to running the visualization code.

    :param btstrap: Boolean to specify whether to draw bootstrap intervals
    :param test_words: Words to run visualizations of
    :param all_models: All models tested
    :param emb_dict_all: Embedding dictionary mapping word to vector
    :param nyt_corpus: Embedding corpus
    :return:
    """
    test_df = load_test_df(emb_dict_all, test_words)
    mfd_dict = models.load_mfd_df(emb_dict_all)
    mfd_dict_binary = models.load_mfd_df_binary(emb_dict=emb_dict_all)
    mfd_dict_null = models.load_mfd_df_neutral(emb_dict=emb_dict_all)
    all_results = {}
    results = []
    for binary_fine_grained in ['NULL', 'BINARY', 'FINEGRAINED']:
        all_fitted_models = [set_model(c, btstrap, binary_fine_grained, mfd_dict, mfd_dict_binary, mfd_dict_null)
                             for c in all_models]
        plt.rcParams.update({'font.size': 13})
        all_years = sorted(test_df[constant.YEAR].unique())
        for model_list in all_fitted_models:
            results = {}
            for i, word in enumerate(test_df[constant.WORD].unique()):
                word_df = test_df[test_df[constant.WORD] == word].sort_values(by=[constant.YEAR])
                years = word_df[constant.YEAR].values
                mean_line, lower_bound, upper_bound = models.models_predictions(model_list, word_df, btstrap)
                results[word] = [years, mean_line, lower_bound, upper_bound]
            all_results[binary_fine_grained] = results
    for word in results:
        for binary_fine_grained in ['NULL', 'BINARY', 'FINEGRAINED']:
            ylim1, ylim2 = set_ylim(nyt_corpus, binary_fine_grained)
            results = all_results[binary_fine_grained]
            years, mean_line, lower_bound, upper_bound = results[word]
            yticks_locs = list(range(0, len(constant.ALL_YEARS), 4))
            if 'FINEGRAINED' == binary_fine_grained:
                if len(mean_line) < len(constant.ALL_YEARS):
                    mean_line = [None] * (len(constant.ALL_YEARS) - len(mean_line)) + mean_line
                plt.subplot(313)
                max_weight = max(max(x.values()) for x in mean_line)
                min_weight = min(min(x.values()) for x in mean_line)
                categories = ['care+', 'fairness+', 'loyalty+', 'authority+', 'sanctity+', 'care-', 'fairness-',
                              'loyalty-', 'authority-', 'sanctity-'][::-1]
                plot_category_hinton(categories, mean_line, yticks_locs, max_weight, min_weight)
            elif 'BINARY' == binary_fine_grained or 'NULL' == binary_fine_grained:
                if binary_fine_grained == 'BINARY':
                    plt.subplot(312)
                else:
                    plt.subplot(311)
                    plt.title(word, fontsize=15)
                plot_binary(word, mean_line, word, years, btstrap, binary_fine_grained, ylim1, ylim2, min(all_years),
                            yticks_locs, lower_bound=lower_bound, upper_bound=upper_bound)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('--vars', nargs='+')
args = parser.parse_args()

btstrap = True
test_words = args.vars
nyt_corpus = ['NYT', 'NGRAM', 'FICTION'][1]
all_models = [lambda: models.CentroidModel()]
emb_dict_all,_ = embeddings.choose_emb_dict('NGRAM')
run(btstrap, test_words, all_models, emb_dict_all, nyt_corpus)