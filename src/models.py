import numpy as np
import pandas as pd
import seeds
import constant
import embeddings
import pickle
import os
import math
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cosine
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import norm


def load_mfd_df(emb_dict=None, reload=False):
    '''
    Load the moral foundations dictionary and pull word representations for each seed word.
    Returns a dataframe with the following columns:
    WORD | VECTOR | CATEGORY | YEAR
    '''
    if reload:
        s = seeds.load(constant.DATA_DIR)
        s_plus = {k+'+':v for k,v in s['+'].items()}
        s_neg = {k+'-':v for k,v in s['-'].items()}
        s_all = {**s_plus, **s_neg}
        items = []
        for cat in s_all:
            for word in s_all[cat]:
                for year in emb_dict.keys():
                    yr_emb_dict = emb_dict[year]
                    if word in yr_emb_dict:
                        items.append({constant.WORD:word, constant.CATEGORY:cat, 
                        constant.YEAR:year, constant.VECTOR: yr_emb_dict[word]})
        cat_df = pd.DataFrame(items)
        pickle.dump(cat_df, open(constant.MFD_DF, 'wb'))
        return cat_df
    return pickle.load(open(constant.MFD_DF, 'rb'))

def load_mfd_df_binary(emb_dict=None, reload=False):
    mfd_dict = load_mfd_df(emb_dict, reload)
    mfd_dict[constant.CATEGORY] = ['+' if '+' in x else '-' for x in mfd_dict[constant.CATEGORY].values]
    return mfd_dict

def log_odds(pos_prob, neg_prob):
    return math.log(pos_prob/neg_prob)

class BaseModel():
    '''
    Base Model
    '''
    def __convert_proba_arr_to_dict(self, proba_arr):
        d = {}
        for i,x in enumerate(proba_arr):
            d[i+1] = x
        return d

    def fit(self, df):
        raise NotImplementedError
    
    def predict(self, data):
        raise NotImplementedError
    
    def predict_proba(self, data):
        raise NotImplementedError

class NBModel(BaseModel):

    name = 'Naive Bayes Model'

    def __init__(self):
        self.c_classifier = GaussianNB()
    
    def fit(self, df):
        self.labels = sorted(df[constant.CATEGORY].unique())
        self.c_classifier.fit(list(df[constant.VECTOR].values), list(df[constant.CATEGORY].values))

    def predict(self, data):
        return self.c_classifier.predict([list(x) for x in data])

    def predict_proba(self, data):
        all_preds = self.c_classifier.predict_proba([list(x) for x in data])
        return [dict(zip(self.labels, x)) for x in all_preds]

class KNNModel(BaseModel):
    '''
    A single kNN classification layer
    '''
    name='kNN Model'
    def __init__(self, k=15):
        self.k = k
        self.c_classifier = KNeighborsClassifier(n_neighbors=self.k,p=1)

    def fit(self, df):
        self.labels = sorted(df[constant.CATEGORY].unique())
        self.c_classifier.fit(list(df[constant.VECTOR].values), list(df[constant.CATEGORY].values))

    def predict(self, data):
        return self.c_classifier.predict([list(x) for x in data])
    
    def predict_proba(self, data):
        all_preds = self.c_classifier.predict_proba([list(x) for x in data])
        return [dict(zip(self.labels, x)) for x in all_preds]

class CentroidModel(BaseModel):
    '''
    A single centroid classification layer
    '''
    name = 'Centroid'

    def __calc_prob(self, X):
        # X_1 = np.exp(np.multiply(X,-1))
        X_1 = np.exp(X)
        softmax_prob = X_1 / np.sum(X_1, axis=0)
        if np.any(np.isnan(softmax_prob)):
            print('wrong', softmax_prob)
            return [0]*len(X)
        return softmax_prob

    def predict_proba(self, data):
        result = []
        for d in data:
            distances = {k: cosine_similarity([d],[v])[0][0] for k, v in self.mean_vectors.items()}
            cat_names = sorted(self.mean_vectors.keys())
            probabilities = self.__calc_prob([distances[k] for k in cat_names])
            x_3 = dict(zip(cat_names, probabilities))
            result.append(x_3)
        assert all(sum(x.values()) > 0.9 for x in result)
        return result
    
    # def objective(self, df, x):
    #     all_categories = list(df[constant.CATEGORY].values)
    #     self.h = x
    #     all_predicts = self.predict_proba(list(df[constant.VECTOR].values))
    #     all_proba = [1-all_predicts[i][all_categories[i]] for i in range(len(df))]
    #     total_loss = sum(all_proba)
    #     return total_loss

    def fit(self, df):
        self.h = 1
        self.mean_vectors = {}
        for i in df[constant.CATEGORY].unique():
            mean_vector = np.mean(list(df[df[constant.CATEGORY] == i][constant.VECTOR].values), axis=0)
            self.mean_vectors[i] = mean_vector
    
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
    name = 'TwoTier'

    def __init__(self, t1Model, t2Model1, t2Model2):
        self.t1 = t1Model
        self.t2_pos = t2Model1
        self.t2_neg = t2Model2
    
    def predict_proba(self, data):
        binary_proba = self.t1.predict_proba(data)
        pos_cat_proba = [{k:binary_proba[i]['+']*v for k,v in x.items()} for i,x in enumerate(self.t2_pos.predict_proba(data))]
        neg_cat_proba = [{k:binary_proba[i]['-']*v for k,v in x.items()} for i,x in enumerate(self.t2_neg.predict_proba(data))]
        all_pred = [{**pos_cat_proba[i], **neg_cat_proba[i]} for i in range(len(pos_cat_proba))]
        assert all(sum(x.values()) > 0.9 for x in all_pred)
        return all_pred

    def fit(self, df):
        binary_df = df.copy()
        binary_df[constant.CATEGORY] = ['+' if '+' in x else '-' for x in df[constant.CATEGORY].values]
        pos_df = df.loc[df[constant.CATEGORY].isin({'care+','loyalty+','fairness+','authority+','sanctity+'})]
        neg_df = df.loc[df[constant.CATEGORY].isin({'care-','loyalty-','fairness-','authority-','sanctity-'})]
        self.t1.fit(binary_df)
        self.t2_pos.fit(pos_df)
        self.t2_neg.fit(neg_df)

    def predict(self, data):
        all_guesses = []
        probs_data = self.predict_proba(data)
        for d in probs_data:
            all_guesses.append(max(d.keys(), key=(lambda key: d[key])))
        return all_guesses

class TwoTierModel():
    '''
    Multi-layered centroid model
    '''
    name = 'TwoTier'

    def __init__(self, t1Model, t2Model):
        self.t1 = t1Model
        self.t2 = t2Model
    
    def predict_proba(self, data):
        binary_proba = self.t1.predict_proba(data)
        pos_cat_proba = [{k:binary_proba[i]['+']*v if '+' in k else binary_proba[i]['-']*v for k,v in x.items()} for i,x in enumerate(self.t2.predict_proba(data))]
        return pos_cat_proba

    def fit(self, df):
        binary_df = df.copy()
        binary_df[constant.CATEGORY] = ['+' if '+' in x else '-' for x in df[constant.CATEGORY].values]
        self.t1.fit(binary_df)
        self.t2.fit(df)

    def predict(self, data):
        all_guesses = []
        probs_data = self.predict_proba(data)
        for d in probs_data:
            all_guesses.append(max(d.keys(), key=(lambda key: d[key])))
        return all_guesses
