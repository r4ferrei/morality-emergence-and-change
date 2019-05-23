import embeddings
import constant
import pandas as pd
import models
import os
import argparse
import numpy as np
from nltk.corpus import stopwords
from scipy.stats import linregress

"""
Generate the top k words with changing slope (in moral time course trajectory)
at the polarity and relevance tier.
"""


########################################################
# UTILITY FUNCTIONS

def slope(preds, pos_class, row):
    l_regress = linregress(range(len(preds)), preds)
    slope, p_val = l_regress[0], l_regress[3]
    row[pos_class] = slope
    row['P Value'] = p_val
    return row


def mean_function(preds, pos_class, row):
    m = np.mean(preds)
    row[pos_class] = m
    for i, x in enumerate(preds):
        if x >= 0:
            row['start_year'] = constant.ALL_YEARS[len(constant.ALL_YEARS) - len(preds) + i]
            break
    return row


def flank_function(preds, pos_class, row):
    row[pos_class] = np.array(preds)
    return row


def custom_argmax(s, pos=None):
    if pos is None:
        highest_ind = constant.ALL_YEARS.index(s['start_year'])
    else:
        highest_ind = np.argmax([x[pos] for x in s if isinstance(x, np.ndarray)])
    return s.index[highest_ind]


########################################################


def score_words(c_lambda, pos_class_list, mfd_dict, emb_dict_all, all_words, score_function):
    """ The scoring mechanism for how words should be sorted (e.g. slope).

    :param c_lambda: Lambda function that returns a model for moral sentiment prediction
    :param pos_class_list: Target classes (e.g. morally relevant, morally positive)
    :param mfd_dict: Dataframe containing MFD training data
    :param emb_dict_all: Embedding dictionary mapping words to embeddings
    :param all_words: List of most frequent words
    :param score_function: Exact scoring method, specifies how words should be ranked
    :return: sorted pandas Dataframe
    """
    c = c_lambda()
    most_recent_year = max(mfd_dict[constant.YEAR].unique())
    mfd_dict = mfd_dict[mfd_dict[constant.YEAR] == most_recent_year]
    c.fit(mfd_dict)
    df = []
    for word in all_words:
        all_years = sorted(emb_dict_all.keys())
        year_embeddings = [emb_dict_all[year][word] for year in all_years if word in emb_dict_all[year]]
        year_preds = c.predict_proba(year_embeddings)
        row = {constant.WORD: word}
        for pos_class in pos_class_list:
            cls_year_preds = [x[pos_class] for x in year_preds]
            log_odds = np.log(cls_year_preds) - np.log(np.subtract(1, cls_year_preds))
            row = score_function(log_odds, pos_class, row)
        df.append(row)
    return pd.DataFrame(df)


def run():
    if os.environ.get('LOAD_HAMILTON'):
        emb_dict_all, vocab_list = embeddings.load_all()
    else:
        emb_dict_all, vocab_list = embeddings.choose_emb_dict(nyt_corpus)

    most_recent_year = max(emb_dict_all.keys())
    emb_dict = emb_dict_all[most_recent_year]
    vocab = set(pd.read_csv(
        os.path.join(constant.TEMP_DATA_DIR, 'most_frequent.csv'))
                    [constant.WORD].values)

    for c_lambda in all_model_lambdas:
        for score_test in score_tests:

            score_function, score_function_name = score_test[0], score_test[1]
            mfd_dict = models.choose_mfd_df(test_type, emb_dict_all)
            mfd_dict_null = models.choose_mfd_df('RELEVANCE', emb_dict_all)
            mfd_dict_category = models.choose_mfd_df('FINE-GRAINED', emb_dict_all)

            all_classes = ['0', '1'] if test_type == 'RELEVANCE' else ['-', '+']

            df = pd.DataFrame(list(emb_dict.keys()), columns=[constant.WORD])
            df = df \
                [~df[constant.WORD].isin(set(mfd_dict[constant.WORD].values)
                                         | set(stopwords.words('english')))
                 & df[constant.WORD].isin(vocab)]
            all_words = df[constant.WORD].values.tolist()
            df2 = score_words(c_lambda, [all_classes[1]], mfd_dict, emb_dict_all, all_words, score_function)
            df3 = score_words(c_lambda, ['1'], mfd_dict_null, emb_dict_all, all_words, mean_function)
            if INCLUDE_IRRELEVANT:
                df3 = df3.drop(columns=['1'])
            else:
                df3 = df3[df3['1'] >= 0].drop(columns=['1'])
            df4 = score_words(c_lambda, mfd_dict_category[constant.CATEGORY].unique(),
                              mfd_dict_category, emb_dict_all, all_words, flank_function) \
                .merge(df3, on=constant.WORD)

            if not INCLUDE_IRRELEVANT:
                df4['start_cat'] = df4.agg(custom_argmax, 1)
                df4['end_cat'] = df4.agg(custom_argmax, 1, -1)

            df = df \
                .merge(df2, on=constant.WORD) \
                .merge(df4, on=constant.WORD) \
                .sort_values(by=all_classes[1], ascending=False)

            if INCLUDE_IRRELEVANT:
                name = '{}_retrievals_include_irrelevant.csv'
            else:
                name = '{}_retrievals.csv'
            df.to_csv(os.path.join(constant.TEMP_DATA_DIR,
                                   name.format(test_type)))


parser = argparse.ArgumentParser()
parser.add_argument('--moralitytier', choices={"POLARITY", "RELEVANCE"}, help="Morality tier \
of evaluation, must be POLARITY or RELEVANCE")
parser.add_argument('--outputpath', help="Path to output csv file")
args = parser.parse_args()
test_type = args.moralitytier
output_filename = args.outputpath


nyt_corpus = 'NGRAM'
all_model_lambdas = [lambda: models.CentroidModel()]
INCLUDE_IRRELEVANT = False
score_tests = [(slope, 'slope')]
run()
