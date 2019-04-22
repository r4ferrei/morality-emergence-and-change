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

def make_preds(df, emb_dict_all, c, test_type):
    new_df = df.copy()
    yr = max(emb_dict_all.keys())
    emb_dict = emb_dict_all[max(emb_dict_all.keys())]
    concepts = new_df.index.values
    vectors = [embeddings.get_sent_embed(emb_dict, x) for x in concepts]
    if test_type == 'binary':
        cls = '+'
        new_df['pew_data'] = df['+']/(df['-'] + df['+'])
        mfd_dict = models.load_mfd_df_binary(emb_dict_all, reload=True)
    elif test_type == 'null':
        cls = '1'
        new_df['pew_data'] = df['1']/(df['-'] + df['+'] + df['1'])
        mfd_dict = models.load_mfd_df_neutral(emb_dict_all, reload=True)
    else:
        raise NotImplementedError
    mfd_dict = mfd_dict[mfd_dict[constant.YEAR] == yr]
    c.fit(mfd_dict)
    predictions = [c.predict_proba([embedding])[0][cls] if embedding is not None else None for embedding in vectors]
    new_df['pred'] = predictions
    return new_df

def make_correlations(df):
    x1 = df['pew_data'].values.tolist()
    x2 = df['pred'].values.tolist()
    print('Pearson correlation: {}'.format(pearsonr(x1,x2)[0]))
    print('Spearman correlation: {}'.format(spearmanr(x1,x2)[0]))

config = configparser.ConfigParser()
config.read('correlate_pew.ini')
embedding_style = config['DEFAULT']['embedding_style']
country = config['DEFAULT']['country']
all_models = eval(config['DEFAULT']['all_models'])
test_type = config['DEFAULT']['test_type']
emb_dict_all,_ = embeddings.choose_emb_dict(embedding_style)
df = pd.read_csv(join(constant.DATA_DIR,'pewwords.csv'))
grouped_df = group_df(df, country)

for c in all_models:
    c = c()
    pred_df = make_preds(grouped_df, emb_dict_all, c, test_type)
    print(pred_df)
    make_correlations(pred_df)