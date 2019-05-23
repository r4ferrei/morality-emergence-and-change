import pandas as pd
import constant
import embeddings
import models
import numpy as np
import argparse
from scipy.stats import pearsonr, spearmanr
from os.path import join
from models import CentroidModel
from matplotlib import pyplot as plt

'''
Run correlation analysis for valence arousal ratings and Pew survey data
'''


def group_df(social_df, country_name=None):
    """Restrict social (Pew) data to select countries.

    :param social_df: pandas Dataframe containing Pew survey data
    :param country_name: country to restrict data to (e.g. United States)
    :return: summarized dataframe
    """
    if 'Country' not in social_df:
        return social_df
    if country_name is not None:
        new_df = social_df[social_df['Country'] == country_name]
    else:
        new_df = social_df.copy()
    assert not new_df.empty
    new_df = new_df.drop(columns=['Country'])
    new_df = new_df.groupby(by=constant.CONCEPT).agg('sum')
    return new_df


def make_preds(df, df_type, emb_dict_all, pred_model, test_type):
    """
    Make predictions on the concepts given in social data

    :param df: social data (Pandas dataframe)
    :param df_type: Pew or valence
    :param emb_dict_all: Embedding dictionary
    :param pred_model: Prediction model
    :param test_type: Binary or null
    :return: Pandas dataframe with predictions made
    """
    new_df = df.copy()
    counter_df = df.copy()
    yr = max(emb_dict_all.keys())
    emb_dict = emb_dict_all[max(emb_dict_all.keys())]
    concepts = [str(x) for x in new_df.index.values]
    vectors = [embeddings.get_sent_embed(emb_dict, x) for x in concepts]
    if test_type == 'polarity':
        cls, cls_counter = '+', '-'
        if df_type == 'valence':
            new_df['orig_data'] = df['v_rating']
        elif df_type == 'pew':
            counter_df['orig_data'] = df['-']/(df['-']+df['+']+df['1'])
            new_df['orig_data'] = df['+']/(df['-']+df['+']+df['1'])
        mfd_dict = models.load_mfd_df_binary(emb_dict_all, reload=True)
    elif test_type == 'null':
        cls, cls_counter = '0', '1'
        new_df['orig_data'] = df['1']/(df['-']+df['+']+df['1'])
        counter_df['orig_data'] = (df['-']+df['+'])/(df['-']+df['+']+df['1'])
        mfd_dict = models.load_mfd_df_neutral(emb_dict_all, reload=True)
    else:
        raise NotImplementedError
    mfd_dict = mfd_dict[mfd_dict[constant.YEAR] == yr]
    pred_model.fit(mfd_dict)
    new_df['pred'] = [pred_model.predict_proba([embedding])[0][cls]
                      if embedding is not None else None for embedding in vectors]
    return new_df


def plot_correlations(df, test_type):
    """Visualization code

    :param df:
    :param test_type:
    :return:
    """
    df = df.dropna()
    x1 = df['orig_data'].values.tolist()
    x2 = df['pred'].values.tolist()
    plt.scatter(x1, x2)
    plt.ylim(0.48, 0.51)
    plt.xticks(np.arange(round(min(x1)/0.05)*0.05, round(max(x1)/0.05)*0.05+0.05, 0.1))
    xlabel_i = None
    for i, point in df.iterrows():
        x, y = point['orig_data'], point['pred']
        ha, va = 'center', 'bottom'
        # Fix overlap
        if test_type == 'Moral Polarity':
            xlabel_i = 'acceptance'
            if i == 'extramarital affair':
                ha = 'left'
            elif i == 'abortion':
                x -= 0.01
                ha = 'right'
                va = 'center'
        if test_type == 'Moral Relevance':
            xlabel_i = 'irrelevance'
            if i == 'divorce':
                va = 'top'
            if i == 'extramarital affair':
                ha = 'left'
        plt.text(x, y, i, fontsize=15, horizontalalignment=ha, verticalalignment=va)
    plt.text(round(min(x1)/0.05)*0.05, 0.4801,
             'p={:.2f}'.format(pearsonr(x1, x2)[0]), fontsize=16)
    plt.title(test_type, fontsize=17)
    plt.xlabel('Percentage agreement of {} (%)'.format(xlabel_i), fontsize=15)
    plt.ylabel('Model predicted probability (%)', fontsize=15)
    plt.plot(np.unique(x1), np.poly1d(np.polyfit(x1, x2, 1))(np.unique(x1)), alpha=0.5, linewidth=2)
    plt.show()
    plt.clf()


def make_correlations(pred_df):
    """Run Pearson and Spearman correlation analysis on data.

    :param pred_df: Pandas dataframe containing prediction and social data
    :return:
    """
    pred_df = pred_df.dropna()
    x1 = pred_df['orig_data'].values.tolist()
    x2 = pred_df['pred'].values.tolist()
    print(pred_df)
    print('Pearson correlation: {}, P-val: {}, n: {}'.format(pearsonr(x1, x2)[0],
                                                             pearsonr(x1, x2)[1],
                                                             len(x1)))
    print('Spearman correlation: {}, P-val: {}, n: {}'.format(spearmanr(x1, x2)[0],
                                                              pearsonr(x1, x2)[1],
                                                              len(x1)))


def run():
    all_models = [CentroidModel]
    embedding_style = 'NGRAM'
    emb_dict_all, _ = embeddings.load_all(embedpath)
    all_test_types = ['polarity', 'null'] if filetype == 'pew' else ['polarity']
    for test_type in all_test_types:
        print('{} {}'.format(embedding_style, test_type))
        if embedding_style == 'NGRAM':
            country = None
        else:
            country = 'United States'
        df = pd.read_csv(join(constant.DATA_DIR, filepath))
        grouped_df = group_df(df, country)
        if constant.CONCEPT in grouped_df.columns.values:
            grouped_df = grouped_df.set_index(constant.CONCEPT)
        for c in all_models:
            c = c()
            pred_df = make_preds(grouped_df, filetype, emb_dict_all, c, test_type)
            if filetype == 'pew':
                plot_correlations(pred_df, 'Moral Polarity' if test_type == 'polarity' else 'Moral Relevance')
            pred_df.to_csv('{}_correlation_{}.csv'.format(filetype, test_type))
            make_correlations(pred_df)


parser = argparse.ArgumentParser()
parser.add_argument('--filepath', help="Path to social data csv")
parser.add_argument('--datatype', help="Must be pew or valence")
parser.add_argument('--embedpath', help="Path to embeddings directory")
parser.add_argument('--outputpath', help="Path to output csv file")
args = parser.parse_args()

filepath = args.filepath
filetype = args.datatype
embedpath = args.embedpath
assert filepath
assert filetype

run()