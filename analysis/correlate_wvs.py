import os
import pandas as pd
import pickle
import embeddings
import math
import models
import constant
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
from numpy import nanmean
from sklearn import preprocessing

code_book = {'F114': 'claiming government benefits', 'F114_01': 'Stealing property', 'F114_02': 'Parents beating children', 
'F114_03': 'Violence against other people', 'F115': 'avoiding a fare on public transport', 'F116': 'cheating on taxes', 
'F117': 'someone accepting a bribe', 'F118': 'homosexuality', 'F119': 'prostitution', 'F120': 'abortion', 
'F121': 'divorce', 'F122': 'euthanasia', 'F123': 'suicide', 'F124': 'drinking alcohol', 'F125': 'joyriding', 
'F126': 'taking soft drugs', 'F127': 'lying', 'F128': 'adultery', 'F129': 'throwing away litter', 
'F130': 'driving under influence of alcohol', 'F135': 'sex under the legal age of consent', 'F135A': 'Sex before marriage', 
'F136': 'political assassination', 'F139': 'buy stolen goods', 'F140': 'keeping money that you have found', 
'F141': 'fighting with the police', 'F142': 'failing to report damage you’ve done accidentally to a parked vehicle', 
'F143': 'threatening workers who refuse to join a strike', 'F144': 'killing in self-defence', 'S003': 'Country', 'S020': constant.YEAR}
mfd_dict, wvs_df, emb_dict_all = None, None, None
EMBEDDING_STYLES = set(['NYT', 'NGRAM'])

def synchronize(ref_arr, other_arr):
    not_nan = np.logical_not(np.isnan(ref_arr))
    condensed_arr = [other_arr[i] for i in range(len(not_nan)) if not_nan[i]]
    return condensed_arr

def load_wvs_df(reload=False):
    '''
    840 - United States, 124 - Canada, 826 - Britain
    Returns a dataframe where each column is either the name of a concept (e.g. claiming government benefits) or the year.
    | claiming government benefits | stealing property | ... | Year |

    '''
    if reload:
        wvs_df = pd.read_sas(os.path.join('data', 'world_values_survey.sas7bdat'))[code_book.keys()]
        wvs_df = wvs_df.loc[wvs_df['S003'].isin([840])]
        wvs_df = wvs_df.drop(columns=['S003'])
        wvs_df = wvs_df.rename(columns=code_book)
        pickle.dump(wvs_df, open(os.path.join('data', 'wvs_df.pkl'), 'wb'))
    wvs_df = pickle.load(open(os.path.join('data', 'wvs_df.pkl'), 'rb'))
    wvs_df = wvs_df.reindex()
    return wvs_df

def load_wvs_df_predictions(embedding_style, mfd_dict=None, wvs_df=None, emb_dict_all=None, reload=False):
    '''
    Adds prediction the wvs_df (see above). For a given concept, predict the moral rhetoric surrounding it.
    Building on the dataframe given before:
    | Model1 | Model2 | ... | concept | score | year
    '''
    if reload:
        al = []
        for year in wvs_df[constant.YEAR].unique():
            year = int(year)
            if embedding_style == 'NYT':
                round_year = year
            else: # NGRAM
                if int(year) < 1990:
                    round_year = 1980
                else:
                    round_year = 1990
            if round_year in emb_dict_all.keys():
                emb_dict = emb_dict_all[round_year]
            else:
                emb_dict = None
            year_df = wvs_df[wvs_df[constant.YEAR] == year]
            for col in year_df:
                word = col
                if -4.0 not in year_df[col].values:
                    score = sum(year_df[col].values)/len(year_df[col].values)
                    if word != constant.YEAR:
                        entry = {constant.YEAR: round_year, constant.CONCEPT: word, constant.SCORE: score}
                        pred = embeddings.get_sent_embed(emb_dict, word)
                        if pred is not None:
                            for model in all_models:
                                model.fit(mfd_dict[mfd_dict[constant.YEAR] == int(round_year)])
                                entry[model.name] = model.predict_proba([pred])[0]['+']
                        al.append(entry)
        df = pd.DataFrame(al)
        df.drop_duplicates()
        df.to_csv(os.path.join('reports', 'world_values.csv'), index=False)
    df = pd.read_csv(os.path.join('reports', 'world_values.csv'),index_col=False)
    return df

def make_consolidated_df(df):
    alll = []
    for year in df[constant.YEAR].unique():
        year_df = df[df[constant.YEAR] == year]
        scores = list(year_df[constant.SCORE].values)
        entry = {constant.YEAR: year}
        for col in year_df:
            if col not in [constant.YEAR,constant.CONCEPT,constant.SCORE] and not np.isnan(year_df[col].values).all():
                preds = list(year_df[col].values)
                not_nan = np.logical_not(np.isnan(preds))
                preds = [preds[i] for i in range(len(not_nan)) if not_nan[i]]
                scores2 = [scores[i] for i in range(len(not_nan)) if not_nan[i]]
                rho,_ = spearmanr(scores2, preds)
                entry['%s Spearman' % col] = '%.2f' % rho
                rho,_ = pearsonr(scores2, preds)
                entry['%s Pearson' % col] = '%.2f' % rho
        alll.append(entry)
    df = pd.DataFrame(alll).to_csv(os.path.join('reports', 'consolidated_world_values.csv'), index=False)
    return df

def draw_graph(df, all_models):
    cmap = plt.get_cmap("tab20")
    for model in all_models:
        model_df = df[[model.name, constant.YEAR, constant.SCORE, constant.CONCEPT]]
        print(model_df)
        model_df = model_df.dropna()
        model_df[constant.SCORE], model_df[model.name] = preprocessing.normalize([list(model_df[constant.SCORE].values), list(model_df[model.name].values)])
        for i,concept in enumerate(model_df[constant.CONCEPT].unique()):
            concept_df = model_df[model_df[constant.CONCEPT] == concept]
            concept_df.sort_values(by=constant.YEAR)
            years = list(concept_df[constant.YEAR].values)
            plt.plot(years, list(concept_df[model.name].values), color=cmap(i), label=concept)
            plt.plot(years, list(concept_df[constant.SCORE].values), color=cmap(i), linestyle='dashed')
        plt.title('%s Predictions vs. Actual Survey Scores' % model.name)
        sum_df = model_df.groupby(constant.YEAR).mean()
        years = list(sum_df.index.values.tolist())
        plt.plot(years, list(sum_df[model.name].values), color='black', label='average', linewidth=4.0)
        plt.plot(years, list(sum_df[constant.SCORE].values), color='black', linestyle='dashed', linewidth=7.0)
        plt.legend(loc='upper right')
        plt.show()
        plt.clf()

def correlate_words(df, all_models):
    al_df = []
    for model in all_models:
        model_df = df[[model.name, constant.YEAR, constant.SCORE, constant.CONCEPT]]
        model_df = model_df.dropna()
        for i,concept in enumerate(model_df[constant.CONCEPT].unique()):
            concept_df = model_df[model_df[constant.CONCEPT] == concept]
            al_df.append({
                'model': model.name,
                'concept': concept,
                'Pearson':pearsonr(list(concept_df[model.name].values), list(concept_df[constant.SCORE].values))[0],
                'Spearman': spearmanr(list(concept_df[model.name].values), list(concept_df[constant.SCORE].values))[0]})
    pd.DataFrame(al_df).to_csv('world_values_concept.csv', index=False)

all_models = [models.CentroidModel(), models.KNNModel(k=16)]
load = True
embedding_style = 'NGRAM'

assert embedding_style in EMBEDDING_STYLES
if load == True:
    if embedding_style == 'NYT':
        emb_dict_all,_ = embeddings.load_all_nyt()
    else: # NGRAM
        emb_dict_all,_ = embeddings.load_all(dir='E:/sgns')
    mfd_dict = models.load_mfd_df_binary(emb_dict_all, reload=True)
wvs_df = load_wvs_df(reload=load)
df = load_wvs_df_predictions(embedding_style, mfd_dict=mfd_dict, wvs_df=wvs_df, emb_dict_all=emb_dict_all, reload=load)

df = df.groupby([constant.CONCEPT, constant.YEAR]).agg(nanmean).reset_index()
df = df[df[constant.YEAR] != 2011]

draw_graph(df, all_models)
correlate_words(df, all_models)
make_consolidated_df(df)