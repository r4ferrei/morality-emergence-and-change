import sys
sys.path.append('C:/Users/Jing/Documents/GitHub/morality-emergence-and-change/src')
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

def plot_extra(word, model_list):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    df = get_ref_df()
    word_df = df[df[constant.WORD] == word]
    for i, row in word_df.iterrows():
        plt.axvline(x=row[constant.YEAR], color=colors[i%len(colors)], label=row[constant.CONCEPT])

def plot_extra_values(word, model_list):
    col_vals = value_df[word].values.tolist()
    val_years = [years[i] for i in range(len(col_vals)) if col_vals[i] != None]
    col_vals = [x for x in col_vals if x != None]
    plt.plot(val_years, col_vals, label=word)

def load_test_df(emb_dict_all=None, reload=False):
    test_words = []
    if reload:
        test_df = []
        for year in emb_dict_all.keys():
            yr_emb_dict = emb_dict_all[year]
            for i,word in enumerate(test_words):
                embedding = embeddings.get_sent_embed(yr_emb_dict, word)
                if embedding is not None:
                    test_df.append({constant.YEAR:year,constant.WORD:word,constant.VECTOR:embedding})
        pickle.dump(pd.DataFrame(test_df),open(os.path.join(constant.TEMP_DATA_DIR, 'words.pkl'), 'wb'))
    return pickle.load(open(os.path.join(constant.TEMP_DATA_DIR, 'words.pkl'), 'rb'))

def load_test_df_topics(emb_dict_all=None, reload=False):
    df = []
    if emb_dict_all is not None:
        with open(os.path.join(constant.TEMP_DATA_DIR, 'moralwords.csv')) as f:
            for i, line in enumerate(f):
                line_arr = line.split(',')
                head_word = line_arr[0]
                words = [str(x) for x in line_arr]
                print(words)
                for word in words:
                    for year in emb_dict_all:
                        emb_dict_yr = emb_dict_all[year]
                        if word in emb_dict_yr:
                            df.append({constant.WORD: word, constant.CONCEPT: head_word,
                                       constant.VECTOR: emb_dict_yr[word], constant.YEAR: year})
        print(len(df))
        pickle.dump(pd.DataFrame(df), open(os.path.join(constant.TEMP_DATA_DIR, 'topic_dict.pkl'), 'wb'))
    df = pickle.load(open(os.path.join(constant.TEMP_DATA_DIR, 'topic_dict.pkl'), 'rb'))
    return df


# value_df = get_values_df()
# Params
binary_fine_grained = ['BINARY', 'FINEGRAINED', 'NULL'][1]
btstrap = True
load = True
nyt_corpus = ['NYT', 'NGRAM', 'FICTION'][0]
all_models = [lambda: CentroidModel()]
emb_dict_all = None
if load:
    emb_dict_all,_ = embeddings.choose_emb_dict(nyt_corpus)
load_test_df = load_test_df_topics

set_plot(binary_fine_grained, btstrap,
             load, all_models, emb_dict_all, load_test_df, nyt_corpus, plot_extra=None)