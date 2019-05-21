import embeddings
import constant
import pandas as pd
import models
import os
import random
import numpy as np
from nltk.corpus import stopwords
from scipy.stats import linregress, zscore

#########################################
# Whether to include morally irrelevant top-changing concepts, or filter
# them out.
INCLUDE_IRRELEVANT = True
#########################################

def get_beta(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    return ((X*Y).mean()-X.mean()*Y.mean())\
    /((X**2).mean()-(X.mean())**2)

# def z_score_slope(preds):
#     x = get_beta(range(len(preds)), preds)
#     all_slopes = []
#     for _ in range(1000):
#         preds_copy = preds.copy()
#         random.shuffle(preds_copy)
#         new_slope = get_beta(range(len(preds_copy)), preds_copy)
#         all_slopes.append(new_slope)
#     mu = np.mean(all_slopes)
#     sigma = np.std(all_slopes)
#     return (x - mu) / sigma

def slope(preds, pos_class, row):
    l_regress = linregress(range(len(preds)), preds)
    slope, p_val = l_regress[0], l_regress[3]
    row[pos_class] = slope
    row['P Value'] = p_val
    return row

def mean_function(preds, pos_class, row):
    m = np.mean(preds)
    row[pos_class] = m
    for i,x in enumerate(preds):
        if x >= 0:
            row['start_year'] = constant.ALL_YEARS[len(constant.ALL_YEARS)-len(preds)+i]
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

def score_words(c_lambda, pos_class_list, mfd_dict, emb_dict_all, all_words, score_function):
    c = c_lambda()
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
            log_odds = np.log(cls_year_preds) - np.log(np.subtract(1,cls_year_preds))
            row = score_function(log_odds, pos_class, row)
        df.append(row)
    return pd.DataFrame(df)

binary_fine_grained = ['BINARY', 'FINEGRAINED', 'NULL'][1]
nyt_corpus = ['NYT', 'NGRAM', 'FICTION'][1]
all_model_lambdas = [lambda: models.CentroidModel()]
k = 100
all_test_types = ['BINARY', 'NULL']
score_tests = [
    # (z_score_slope,'z_score')
    (slope,'slope')
    # (two_end_points,'end_points')
]

if os.environ.get('LOAD_HAMILTON'):
    emb_dict_all,vocab_list = embeddings.load_all()
else:
    emb_dict_all,vocab_list = embeddings.choose_emb_dict(nyt_corpus)

most_recent_year = max(emb_dict_all.keys())
emb_dict = emb_dict_all[most_recent_year]
vocab = set(pd.read_csv(os.path.join(constant.TEMP_DATA_DIR, 'most_frequent.csv'))\
    [constant.WORD].values)

for c_lambda in all_model_lambdas:
    for test_type in all_test_types:
        for score_test in score_tests:

            score_function, score_function_name = score_test[0], score_test[1]
            mfd_dict = models.choose_mfd_df(test_type, emb_dict_all, True)
            mfd_dict_null = models.choose_mfd_df('NULL', emb_dict_all, True)
            mfd_dict_category = models.choose_mfd_df('FINEGRAINED', emb_dict_all, True)

            all_classes = ['0', '1'] if test_type == 'NULL' else ['-', '+']

            df = pd.DataFrame(list(emb_dict.keys()), columns=[constant.WORD])
            df = df\
                [~df[constant.WORD].isin(set(mfd_dict[constant.WORD].values)|set(stopwords.words('english')))\
                 & df[constant.WORD].isin(vocab)]
            all_words = df[constant.WORD].values.tolist()
            df2 = score_words(c_lambda, [all_classes[1]], mfd_dict, emb_dict_all, all_words, score_function)
            df3 = score_words(c_lambda, ['1'], mfd_dict_null, emb_dict_all, all_words, mean_function)
            if INCLUDE_IRRELEVANT:
                df3 = df3.drop(columns=['1'])
            else:
                df3 = df3[df3['1'] >= 0].drop(columns=['1'])
            df4 = score_words(c_lambda, mfd_dict_category[constant.CATEGORY].unique(),
                              mfd_dict_category, emb_dict_all, all_words, flank_function)\
                .merge(df3,on=constant.WORD)

            if not INCLUDE_IRRELEVANT:
                df4['start_cat'] = df4.agg(custom_argmax,1)
                df4['end_cat'] = df4.agg(custom_argmax,1,-1)

            df = df\
                .merge(df2, on=constant.WORD)\
                .merge(df4, on=constant.WORD)\
                .sort_values(by=all_classes[1], ascending=False)

            if INCLUDE_IRRELEVANT:
                name = '{}_{}_retrievals_include_irrelevant.csv'
            else:
                name = '{}_{}_retrievals.csv'
            df.to_csv(os.path.join(constant.TEMP_DATA_DIR,
                                          name.format(score_function_name, test_type)))
