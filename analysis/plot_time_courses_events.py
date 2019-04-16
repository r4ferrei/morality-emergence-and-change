import constant
import pickle
import pandas as pd
import os
import embeddings
import matplotlib.pyplot as plt
from plot_time_courses import set_plot
from models import CentroidModel

def get_scale(l):
    return dict(zip(l, list(range(len(l)))))

def get_values_df():
    m = {
        'Divorce laws': {
            'mapping': get_scale(['More difficult', 'Stay same', 'Eaiser']),
            'summary': 'divorce'
            },
        'Abortion if woman wants for any reason': {
            'mapping': get_scale(['No', 'Yes']),
            'summary': 'abortion'
        },
        'Favor law against racial intermarriage': {
            'mapping': get_scale(['No', 'Yes']),
            'summary': 'intermarriage'
        },
        'Strength of affiliation': {
            'mapping': get_scale(['No religion', 'Not very strong', 'Somewhat strong', 'Strong']),
            'summary': 'religion'
        },
        'Should marijuana be made legal': {
            'mapping': get_scale(['Not legal', 'Legal']),
            'summary': 'marijuana'
        },
        'Favor or oppose gun permits': {
            'summary': 'guns',
            'mapping': get_scale(['Oppose', 'Favor'])
        },
        'Favor or oppose death penalty for murder': {
            'summary': 'execution',
            'mapping': get_scale(['Oppose', 'Favor'])
        },
        'Welfare': {
            'summary': 'welfare',
            'mapping': get_scale(['Too much', 'About right', 'Too little'])
        },
        'Improving nations education system': {
            'summary': 'education',
            'mapping': get_scale(['Too much', 'About right', 'Too little'])
        },
        'Improving & protecting nations health':  {
            'summary': 'health',
            'mapping': get_scale(['Too much', 'About right', 'Too little'])
        },
        'Improving & protecting environment':  {
            'summary': 'environment',
            'mapping': get_scale(['Too much', 'About right', 'Too little'])
        },
        'Homosexual sex relations': {
            'summary': 'homosexual',
            'mapping': get_scale(['Always wrong', 'Almst always wrg', 'Sometimes wrong', 'Not wrong at all'])
        }
    }
    
    df = pd.read_csv(os.path.join(constant.TEMP_DATA_DIR, 'MoralValues', 'GSS.csv'))
    df = df.drop(columns=['Respondent id number'])
    for column in df:
        if column != 'Year':
            print(column, set(df[column].values.tolist()))
            df[column] = df[column].map(m[column]['mapping'], na_action='ignore')
            df = df.rename(index=str, columns={column: m[column]['summary']})
    return(df.groupby(['Year']).mean())

def get_ref_df():
    headers = [constant.WORD, 'description', constant.YEAR, constant.CONCEPT] + ['w{}'.format(i) for i in range(10)]
    df = pd.read_csv(os.path.join(constant.TEMP_DATA_DIR, 'ciment_data.csv'), names=headers)
    return df

def load_test_df_events(years, emb_dict_all=None, reload=False):
    if reload:
        df = get_ref_df()
        test_words = list(df[constant.WORD].unique())
        test_df = []
        for year in years:
            yr_emb_dict = emb_dict_all[year]
            for i,word in enumerate(test_words):
                embedding = embeddings.get_sent_embed(yr_emb_dict, word)
                if embedding is not None:
                    test_df.append({constant.YEAR:year,constant.WORD:word,constant.VECTOR:embedding})
        pickle.dump(pd.DataFrame(df),open(os.path.join(constant.TEMP_DATA_DIR, 'words_event.pkl'), 'wb'))
    return pickle.load(open(os.path.join(constant.TEMP_DATA_DIR, 'words_event.pkl'), 'rb'))

def plot_extra(word):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    df = get_ref_df()
    word_df = df[df[constant.WORD] == word]
    for i, row in word_df.iterrows():
        plt.axvline(x=row[constant.YEAR], color=colors[i%len(colors)], label=row[constant.CONCEPT])

def plot_extra_values(word):
    col_vals = value_df[word].values.tolist()
    val_years = [years[i] for i in range(len(col_vals)) if col_vals[i] != None]
    col_vals = [x for x in col_vals if x != None]
    plt.plot(val_years, col_vals, label=word)

value_df = get_values_df()
# Params
binary_fine_grained = ['BINARY', 'FINEGRAINED', 'NULL'][0]
plot_separate = True # Only has effect in the binary case
mfd_year = 1990
load = True
years = list(value_df.index.values)
nyt_corpus = ['NYT', 'NGRAM', 'FICTION'][2]
test_words = list(value_df.columns.values)
all_models = [lambda: CentroidModel()]

set_plot(test_words, binary_fine_grained, plot_separate, mfd_year, load, years, nyt_corpus, all_models, plot_extra_values)   