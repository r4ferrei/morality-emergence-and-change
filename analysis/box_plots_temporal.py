import embeddings
import constant
import pandas as pd
import models
import os
import random
import numpy as np
from nltk.corpus import stopwords
from scipy.stats import linregress, zscore

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

emb_dict_all,vocab_list = embeddings.choose_emb_dict(nyt_corpus)
most_recent_year = max(emb_dict_all.keys())
emb_dict = emb_dict_all[most_recent_year]
vocab = set(pd.read_csv(os.path.join(constant.TEMP_DATA_DIR, 'most_frequent.csv'))\
    [constant.WORD].values)

for c_lambda in all_model_lambdas:
    for test_type in all_test_types:
        mfd_dict = models.choose_mfd_df(test_type, emb_dict_all, True)
        all_classes = ['0', '1'] if test_type == 'NULL' else ['-', '+']
        df = pd.DataFrame(list(emb_dict.keys()), columns=[constant.WORD])
        df = df\
            [~df[constant.WORD].isin(set(mfd_dict[constant.WORD].values)|set(stopwords.words('english')))\
             & df[constant.WORD].isin(vocab)]
        all_words = df[constant.WORD].values.tolist()
        df2 = score_words(c_lambda, [all_classes[1]], mfd_dict, emb_dict_all, all_words, score_function)
        all_words = df[constant.WORD].values.tolist()
        df.to_csv(os.path.join(constant.TEMP_DATA_DIR,
                                      '{}_{}_retrievals.csv'.format(score_function_name, test_type)))