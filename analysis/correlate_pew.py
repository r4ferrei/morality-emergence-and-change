import pandas as pd
import constant
import embeddings
from os.path import join

def group_df(df, country):
    if country != 'ALL':
        new_df = df.loc[df['Country'] == country]
    else:
        new_df = df.copy()
    assert not new_df.empty
    new_df = new_df.drop(columns=['Country'])
    new_df = new_df.groupby(by=constant.CONCEPT).agg('sum')
    return new_df

def make_preds(df, emb_dict_all):
    new_df = df.copy()
    emb_dict = emb_dict_all[max(emb_dict_all.keys())]
    concepts = new_df.index.values
    vectors = [embeddings.get_sent_embed(emb_dict, x) for x in concepts]

    new_df['pred'] = 0

# embedding_style = 'NGRAMS'
# emb_dict_all,_ = embeddings.choose_emb_dict(embedding_style)

df = pd.read_csv(join(constant.DATA_DIR,'pewwords.csv'))
grouped_df = group_df(df, 'United States')
print(make_preds(grouped_df))