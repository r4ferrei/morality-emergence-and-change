import embeddings
import constant
import pandas as pd
import models
import os
from nltk.corpus import stopwords

def two_end_points(preds):
    return max(preds) - min(preds)

def score_words(c, emb_dict_all, all_words, score_function):
    df = []
    for word in all_words:
        all_years = sorted(emb_dict_all.keys())
        year_embeddings = [emb_dict_all[year][word] for year in all_years if word in emb_dict_all[year]]
        if len(year_embeddings) < 5:
            continue
        year_preds = [c.predict_proba(x) for x in year_embeddings]
        row = {constant.WORD: word}
        for cls in year_preds[0]:
            cls_year_preds = [x[cls] for x in year_preds]
            score = score_function(cls_year_preds)
            row[cls] = score
        df.append(row)
    return pd.DataFrame(df)


binary_fine_grained = ['BINARY', 'FINEGRAINED', 'NULL'][1]
nyt_corpus = ['NYT', 'NGRAM', 'FICTION'][1]
all_model_lambdas = [lambda: models.CentroidModel()]
k = 100
all_test_types = ['BINARY', 'FINEGRAINED', 'NULL']
score_tests = [(two_end_points,'end_points')]

emb_dict_all,vocab_list = embeddings.choose_emb_dict(nyt_corpus)
most_recent_year = max(emb_dict_all.keys())
emb_dict = emb_dict_all[most_recent_year]
all_models = [x() for x in all_model_lambdas]

for c in all_models:
    for test_type in all_test_types:
        for score_test in score_tests:
            score_function, score_function_name = score_test[0], score_test[1]
            if test_type == 'BINARY':
                mfd_dict = models.load_mfd_df_binary(emb_dict_all, reload=True)
            elif test_type == 'FINEGRAINED':
                mfd_dict = models.load_mfd_df(emb_dict_all, reload=True)
            elif test_type == 'NULL':
                mfd_dict = models.load_mfd_df_neutral(emb_dict_all, reload=True)
            else:
                raise NotImplementedError

            mfd_dict = mfd_dict[mfd_dict[constant.YEAR] == most_recent_year]
            c.fit(mfd_dict)
            all_classes = mfd_dict[constant.CATEGORY].unique()
            df = pd.DataFrame(list(emb_dict.keys()), columns=[constant.WORD])
            df = df[~df[constant.WORD].isin(set(mfd_dict[constant.WORD].values)|set(stopwords.words('english')))]
            df2 = score_words(c, emb_dict_all, df[constant.WORD].values.tolist(), score_function)
            df = pd.merge(df, df2, how='inner', on=constant.WORD)
            result_df = pd.DataFrame()
            for cls in all_classes:
                df = df.sort_values(by=cls, ascending=False)
                result_df[cls] = df.head(n=k)[constant.WORD].values
                result_df['{} score'.format(cls)] = df.head(n=k)[cls].values
            result_df.to_csv(os.path.join(constant.TEMP_DATA_DIR,
                                          '{}_{}_retrievals.csv'.format(score_function_name, test_type)))