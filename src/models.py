import numpy as np
import pandas as pd
import seeds
import constant
import embeddings
import pickle
import os
from hyperopt import hp, tpe, fmin, Trials
from scipy.spatial.distance import cosine

def load_mfd_df():
    '''
    Load the moral foundations dictionary and pull word representations for each seed word.
    Returns a dataframe with the following columns:
    WORD | VECTOR | CATEGORY | YEAR
    '''
    emb_dict,_ = embeddings.load_all(dir=constant.SGNS_DIR)
    s = seeds.load(constant.DATA_DIR)
    s_plus = {k+'+':v for k,v in s['+'].items()}
    s_neg = {k+'-':v for k,v in s['-'].items()}
    s_all = {**s_plus, **s_neg}
    items = []
    for cat in s_all:
        for word in s_all[cat]:
            for year in constant.ALL_YEARS:
                yr_emb_dict = emb_dict[year]
                if word in yr_emb_dict:
                    items.append({constant.WORD:word, constant.CATEGORY:cat, 
                    constant.YEAR:year, constant.VECTOR: yr_emb_dict[word]})
    cat_df = pd.DataFrame(items)
    return cat_df
    pickle.dump(cat_df, open(constant.MFD_DF, 'wb'))

class CentroidModel():
    '''
    A single centroid classification layer
    '''

    def __calc_prob(self, X, h):
        X_1 = np.exp(np.multiply(X,-1/h))
        softmax_prob = X_1 / np.sum(X_1, axis=0)
        if np.any(np.isnan(softmax_prob)):
            print('wrong', softmax_prob)
            return [0]*len(X)
        return softmax_prob

    def predict_proba(self, data):
        result = []
        for d in data:
            distances = {k: np.linalg.norm(d-v) for k, v in self.mean_vectors.items()}
            cat_names = sorted(self.mean_vectors.keys())
            probabilities = self.__calc_prob([distances[k] for k in cat_names], self.h)
            x_3 = dict(zip(cat_names, probabilities))
            result.append(x_3)
        assert all(sum(x.values()) > 0.9 for x in result)
        return result
    
    def objective(self, df, x):
        all_categories = list(df[constant.CATEGORY].values)
        self.h = x
        all_predicts = self.predict_proba(list(df[constant.VECTOR].values))
        all_proba = [1-all_predicts[i][all_categories[i]] for i in range(len(df))]
        total_loss = sum(all_proba)
        return total_loss

    def fit(self, df):
        self.mean_vectors = {}
        for i in df[constant.CATEGORY].unique():
            mean_vector = np.mean(list(df[df[constant.CATEGORY] == i][constant.VECTOR].values), axis=0)
            self.mean_vectors[i] = mean_vector
        
        obj_func = lambda x : self.objective(df, x)
        space = hp.uniform('x', 0, 1)
        tpe_algo = tpe.suggest
        tpe_trials = Trials()

        tpe_best = fmin(fn=obj_func, space=space, algo=tpe_algo, trials=tpe_trials, max_evals=500)
        self.h = tpe_best['x']
        assert self.h != 0
        print(tpe_best)

    def predict(self, data):
        all_guesses = []
        probs_data = self.predict_proba(data)
        for d in probs_data:
            all_guesses.append(max(d.keys(), key=(lambda key: d[key])))
        return all_guesses

class TwoTierCentroidModel():
    '''
    Multi-layered centroid model
    '''
    
    def predict_proba(self, data):
        binary_proba = self.t1.predict_proba(data)
        pos_cat_proba = [{k:binary_proba[i]['+']*v for k,v in x.items()} for i,x in enumerate(self.t2_pos.predict_proba(data))]
        neg_cat_proba = [{k:binary_proba[i]['-']*v for k,v in x.items()} for i,x in enumerate(self.t2_neg.predict_proba(data))]
        all_pred = [{**pos_cat_proba[i], **neg_cat_proba[i]} for i in range(len(pos_cat_proba))]
        assert all(sum(x.values()) > 0.9 for x in all_pred)
        return all_pred

    def fit(self, df):
        self.t1 = CentroidModel()
        self.t2_pos = CentroidModel()
        self.t2_neg = CentroidModel()
        binary_df = df.copy()
        binary_df[constant.CATEGORY] = ['+' if '+' in x else '-' for x in df[constant.CATEGORY].values]
        pos_df = df.loc[df[constant.CATEGORY].isin(set(['care+','loyalty+','fairness+','authority+','sanctity+']))]
        neg_df = df.loc[df[constant.CATEGORY].isin(set(['care-','loyalty-','fairness-','authority-','sanctity-']))]
        self.t1.fit(binary_df)
        self.t2_pos.fit(pos_df)
        self.t2_neg.fit(neg_df)

    def predict(self, data):
        all_guesses = []
        probs_data = self.predict_proba(data)
        for d in probs_data:
            all_guesses.append(max(d.keys(), key=(lambda key: d[key])))
        return all_guesses
