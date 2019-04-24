import os
import pandas as pd
import pickle
import embeddings
import math
import models
import constant
from scipy.stats import spearmanr, ttest_ind
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
from numpy import nanmean
from sklearn import preprocessing
from scipy import stats

topic_book = {
    'homosexuality': ['gay', 'sexual', 'homosexual', 'marriage', 'sex'],
    'prostitution': ['sex', 'prostitute', 'sexual', 'trafficking', 'victim'],
    'abortion': ['pregnancy', 'woman', 'pregnant', 'birth', 'baby'],
    'divorce': ['marriage', 'spouse', 'married', 'marry', 'court'],
    'suicide': ['mental', 'death', 'depression', 'kill', 'die']
}
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
        wvs_df = pd.read_sas(os.path.join(constant.DATA_DIR, 'world_values_survey.sas7bdat'))[code_book.keys()]
        wvs_df = wvs_df.loc[wvs_df['S003'].isin([840, 124, 826])]
        wvs_df = wvs_df.drop(columns=['S003'])
        wvs_df = wvs_df.rename(columns=code_book)
        pickle.dump(wvs_df, open(os.path.join(constant.TEMP_DATA_DIR, 'wvs_df.pkl'), 'wb'))
    wvs_df = pickle.load(open(os.path.join(constant.TEMP_DATA_DIR, 'wvs_df.pkl'), 'rb'))
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
            if embedding_style == 'FICTION':
                round_year = 5*round(year/5) 
            elif embedding_style == 'NYT':
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
                                prediction = model.predict_proba([pred])[0]
                                entry[model.name] = models.log_odds(prediction['+'], prediction['-'])
                        al.append(entry)
        df = pd.DataFrame(al)
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
                entry['%s Spearman' % col] = '%.4f' % rho
                rho,_ = pearsonr(scores2, preds)
                entry['%s Pearson' % col] = '%.4f' % rho
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
                'Pearson':'%.4f' % pearsonr(list(concept_df[model.name].values), list(concept_df[constant.SCORE].values))[0],
                'Spearman':'%.4f' % spearmanr(list(concept_df[model.name].values), list(concept_df[constant.SCORE].values))[0]})
    pd.DataFrame(al_df).to_csv(os.path.join(constant.DATA_DIR, 'world_values_concept.csv'), index=False)

def round_year(year):
    return round(math.floor(year)/10)*10

def get_closest_year(year, all_years):
    return min(all_years, key=lambda x:abs(x-year))

def ttest(l_mean, l_std, u_mean, u_std):
    sed = math.sqrt(l_std**2 + u_std**2)
    t = (l_mean - u_mean)/sed
    return t

def consolidate(df):
    new_df = df.copy()
    # new_df[constant.YEAR] = df.apply(lambda row: round_year(row[constant.YEAR]), axis=1)
    new_df = new_df.replace([-5,-4,-3,-2,-1], float('NaN'))
    new_df = new_df.groupby([constant.YEAR]).agg([np.nanmean, np.nanstd, lambda x: np.count_nonzero(~np.isnan(x))]).reset_index()
    new_df = new_df.set_index(constant.YEAR)
    new_df = new_df.transpose()
    return new_df

def get_change_df(df, lower_year, upper_year):
    al_df = []
    # new_df = df.drop(columns=[x for x in df.columns.values if x - 10 > upper_year or x + 10 < lower_year])
    new_df = df.copy()
    concepts = set([x[0] for x in new_df.index.values])
    for concept in concepts:
        available_years = [year for year in new_df.columns if not np.isnan(new_df[year][concept]['nanmean'])]
        if len(available_years) < 2:
            continue
        l_year, u_year = get_closest_year(lower_year, available_years), get_closest_year(upper_year, available_years)
        count = new_df[l_year][concept]['<lambda>']
        df = count-1
        #p-value after comparison with the t 
        l_mean, l_std = new_df[l_year][concept]['nanmean'], new_df[l_year][concept]['nanstd']
        u_mean, u_std = new_df[u_year][concept]['nanmean'], new_df[u_year][concept]['nanstd']
        t_stat = ttest(l_mean, l_std, u_mean, u_std)
        p_val = 2*(1-stats.t.cdf(t_stat,df=df))
        al_df.append({constant.CONCEPT: concept, 'direction': -1 if l_mean > u_mean else 1, 't_stat': abs(t_stat),
         'lower_year':l_year, 'upper_year':u_year, 'p_val': p_val})
    al_df = pd.DataFrame(al_df)
    return al_df

def binary_preds(df, emb_dict_all, c, name):
    dic = {}
    df = df.copy()
    for _,row in df.iterrows():
        concept,lower_year,upper_year = row[constant.CONCEPT],row['lower_year'],row['upper_year']
        if concept in topic_book:
            query = ' '.join([concept]+topic_book[concept])
        else:
            query = concept
        # lower_years = [x for x in emb_dict_all.keys() if round_year(x) == lower_year]
        # upper_years = [x for x in emb_dict_all.keys() if round_year(x) == upper_year]
        # lower_years_vecs = [embeddings.get_sent_embed(emb_dict_all[x], query) for x in lower_years]
        # upper_years_vecs = [embeddings.get_sent_embed(emb_dict_all[x], query) for x in upper_years]
        # lower_years_vecs = [x for x in lower_years_vecs if x is not None]
        # upper_years_vecs = [x for x in upper_years_vecs if x is not None]
        lower_year = get_closest_year(lower_year, emb_dict_all.keys())
        upper_year = get_closest_year(upper_year, emb_dict_all.keys())
        print(query)
        l_vec = embeddings.get_sent_embed(emb_dict_all[lower_year], query)
        u_vec = embeddings.get_sent_embed(emb_dict_all[upper_year], query)
        if l_vec is None or u_vec is None:
            continue
        dic[concept] = {'l': l_vec, 'u': u_vec}
    df = df.set_index(constant.CONCEPT)
    df['prediction'] = 0
    for concept in dic:
        l_prediction = c.predict_proba([dic[concept]['l']])[0]['+']
        u_prediction = c.predict_proba([dic[concept]['u']])[0]['+']
        df.set_value(concept, 'lower_prediction', l_prediction)
        df.set_value(concept, 'upper_prediction', u_prediction)
        df.set_value(concept, 'prediction', -1 if l_prediction > u_prediction else 1)
    df = df.sort_values(by='p_val')
    df = df.reindex(['lower_year', 'upper_year', 'direction', 't_stat', 'p_val', 'prediction', 'lower_prediction', 'upper_prediction'], axis=1)
    df = df.round(3)
    df.to_csv(os.path.join(constant.TEMP_DATA_DIR, '{}.csv'.format(name)))
    print(df)

all_models = [models.CentroidModel()]
load = True
for embedding_style in ['NGRAM', 'FICTION', 'COHA']:
    name = embedding_style.lower()
    if load:
        emb_dict_all,_ = embeddings.choose_emb_dict(embedding_style)
    mfd_dict = models.load_mfd_df_binary(emb_dict_all, reload=load)
    year_dict = mfd_dict[mfd_dict[constant.YEAR] == max(emb_dict_all.keys())]
    for model in all_models:
        wvs_df = load_wvs_df(reload=False)
        cons_wvs_df = consolidate(wvs_df)
        model.fit(year_dict)
        cons_wvs_df = get_change_df(cons_wvs_df, min(emb_dict_all.keys()), max(emb_dict_all.keys()))
        binary_preds(cons_wvs_df, emb_dict_all, model, name)