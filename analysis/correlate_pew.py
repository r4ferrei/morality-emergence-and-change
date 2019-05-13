import pandas as pd
import constant
import embeddings
import models
import configparser
from scipy.stats import pearsonr, spearmanr
from os.path import join
from models import CentroidModel

def group_df(df, country):
    if country != 'ALL':
        new_df = df[df['Country'] == country]
    else:
        new_df = df.copy()
    assert not new_df.empty
    new_df = new_df.drop(columns=['Country'])
    new_df = new_df.groupby(by=constant.CONCEPT).agg('sum')
    return new_df

def make_preds(df, df_type, emb_dict_all, c, test_type):
    new_df = df.copy()
    yr = max(emb_dict_all.keys())
    emb_dict = emb_dict_all[max(emb_dict_all.keys())]
    print(df.head())
    concepts = [str(x) for x in new_df.index.values]
    vectors = [embeddings.get_sent_embed(emb_dict, x) for x in concepts]
    if test_type == 'binary':
        cls = '+'
        if df_type == 'valence':
            new_df['orig_data'] = df['v_rating']
        else:
            new_df['orig_data'] = df['+']/(df['-'] + df['+'])
        mfd_dict = models.load_mfd_df_binary(emb_dict_all, reload=True)
    elif test_type == 'null':
        cls = '0'
        new_df['orig_data'] = df['1']/(df['-'] + df['+'] + df['1'])
        mfd_dict = models.load_mfd_df_neutral(emb_dict_all, reload=True)
    else:
        raise NotImplementedError
    mfd_dict = mfd_dict[mfd_dict[constant.YEAR] == yr]
    c.fit(mfd_dict)
    predictions = [c.predict_proba([embedding])[0][cls] if embedding is not None else None for embedding in vectors]
    new_df['pred'] = predictions
    return new_df

def make_correlations(df):
    df = df.dropna()
    x1 = df['orig_data'].values.tolist()
    x2 = df['pred'].values.tolist()
    print(df)
    print('Pearson correlation: {}, P-val: {}, n: {}'.format(pearsonr(x1,x2)[0], pearsonr(x1,x2)[1], len(x1)))
    print('Spearman correlation: {}, P-val: {}, n: {}'.format(spearmanr(x1,x2)[0], pearsonr(x1,x2)[1], len(x1)))

# config = configparser.ConfigParser()
# config.read('correlate_pew.ini')
# embedding_style = config['DEFAULT']['embedding_style']
# country = config['DEFAULT']['country']
# test_type = config['DEFAULT']['test_type']
# all_models = eval(config['DEFAULT']['all_models'])

all_models = [CentroidModel]
for embedding_style in ['NGRAM']:
    for test_type in ['binary', 'null']:
        print('{} {}'.format(embedding_style, test_type))
        if embedding_style == 'NGRAM':
            country = 'ALL'
        else:
            country = 'United States'

        emb_dict_all,_ = embeddings.choose_emb_dict(embedding_style)
        df = pd.read_csv(join(constant.DATA_DIR,'valencewords.csv'))
        # grouped_df = df
        # grouped_df = group_df(df, country)
        if constant.CONCEPT in grouped_df.columns.values:
            grouped_df = grouped_df.set_index(constant.CONCEPT)
        for c in all_models:
            c = c()
            pred_df = make_preds(grouped_df, 'orig', emb_dict_all, c, test_type)
            pred_df.to_csv('{}_{}.csv'.format(embedding_style, test_type))
            make_correlations(pred_df)