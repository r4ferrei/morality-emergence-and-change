import numpy as np
import pandas as pd
import seeds
import constant
import embeddings
import pickle
from sklearn.manifold import TSNE
import os
import math
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cosine
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import norm
from sklearn.decomposition import PCA, LatentDirichletAllocation, TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.naive_bayes import GaussianNB
import itertools
from collections import Counter
from sklearn.metrics import mean_squared_error

def choose_mfd_df(switch, emb_dict_all, reload):
    if switch == 'BINARY':
        mfd_dict = load_mfd_df_binary(emb_dict_all, reload)
    elif switch == 'FINEGRAINED':
        mfd_dict = load_mfd_df(emb_dict_all, reload)
    elif switch == 'NULL':
        mfd_dict = load_mfd_df_neutral(emb_dict_all, reload)
    else:
        raise NotImplementedError
    return mfd_dict

def load_mfd_df(emb_dict=None, reload=False, **kwargs):
    '''
    Load the moral foundations dictionary and pull word representations for each seed word.
    Returns a dataframe with the following columns:
    WORD | VECTOR | CATEGORY | YEAR
    '''
    if reload:
        s = seeds.load(constant.DATA_DIR, **kwargs)
        s_plus = {v:k+'+' for k,v in reshape(s['+'])}
        s_neg = {v:k+'-' for k,v in reshape(s['-'])}
        s_all = {**s_plus, **s_neg}
        items = []
        for word,cat in s_all.items():
            yr_items = []
            for year in emb_dict.keys():
                yr_emb_dict = emb_dict[year]
                if word in yr_emb_dict:
                    yr_items.append({constant.WORD:word, constant.CATEGORY:cat, 
                    constant.YEAR:year, constant.VECTOR: yr_emb_dict[word]})
            if len(yr_items) == len(emb_dict.keys()):
                items += yr_items
        cat_df = pd.DataFrame(items)
        pickle.dump(cat_df, open(constant.MFD_DF, 'wb'))
        return cat_df
    return pickle.load(open(constant.MFD_DF, 'rb'))

def load_mfd_df_binary(emb_dict=None, reload=False):
    mfd_dict = load_mfd_df(emb_dict, reload)
    mfd_dict[constant.CATEGORY] = ['+' if '+' in x else '-' for x in mfd_dict[constant.CATEGORY].values]
    return mfd_dict

def reshape(shape_dict):
    list_of_lists = [zip([x[0]]*len(x[1]),x[1]) for x in shape_dict.items()]
    return list(itertools.chain(*list_of_lists))

def load_mfd_df_neutral(emb_dict=None, reload=False, **kwargs):
    '''
    Load the moral foundations dictionary and pull word representations for each seed word.
    Returns a dataframe with the following columns:
    WORD | VECTOR | CATEGORY | YEAR
    '''
    if reload:
        s = seeds.load(constant.DATA_DIR, **kwargs)
        s_plus = {v:'1' for k,v in reshape(s['+'])}
        s_neg = {v:'1' for k,v in reshape(s['-'])}
        s_neutral = {}
        for n_word in s['0']:
            if n_word in emb_dict[min(emb_dict.keys())]:
                s_neutral[n_word] = '0'
            if len(s_neutral) == len(s_plus)+len(s_neg):
                break
        s_all = {**s_plus, **s_neg, **s_neutral}
        items = []
        for word,cat in s_all.items():
            yr_items = []
            for year in emb_dict.keys():
                yr_emb_dict = emb_dict[year]
                if word in yr_emb_dict:
                    yr_items.append({constant.WORD:word, constant.CATEGORY:cat, 
                    constant.YEAR:year, constant.VECTOR: yr_emb_dict[word]})
            if len(yr_items) == len(emb_dict.keys()):
                items += yr_items
        cat_df = pd.DataFrame(items)
        pickle.dump(cat_df, open(constant.MFD_DF_NEUTRAL, 'wb'))
        return cat_df
    return pickle.load(open(constant.MFD_DF_NEUTRAL, 'rb'))

def log_odds(pos_prob, neg_prob):
    return math.log(pos_prob/neg_prob)

def models_predictions(model_list, word_df, bt_strap):
    mean_line_agg, lower_bound_agg, upper_bound_agg = [], [], []
    X =  word_df[constant.VECTOR].values
    years = word_df[constant.YEAR].values.tolist()
    if bt_strap:
        c = model_list[years[0]]
        mean_line, lower_bound, upper_bound = c.predict_proba_bootstrap(X)
        return mean_line, lower_bound, upper_bound
    for i,year in enumerate(years):
        c = model_list[year]
        mean_line = c.predict_proba([X[i]])
        mean_line_agg.append(mean_line[0])
    return mean_line_agg, lower_bound_agg, upper_bound_agg

class BaseModel():
    '''
    Base Model
    '''
    def __convert_proba_arr_to_dict(self, proba_arr):
        d = {}
        for i,x in enumerate(proba_arr):
            d[i+1] = x
        return d

    def fit_bootstrap(self, mfd_dict, n=1000):
        assert n > 0
        self.fit(mfd_dict)
        self.bootstrap_refs = []
        for i in range(n):
            resample_mfd_dict = mfd_dict.sample(n, replace=True)
            self.bootstrap_refs.append(resample_mfd_dict)

    def predict_proba_bootstrap(self, word_vectors):
        assert hasattr(self, 'bootstrap_refs')
        categories = self.bootstrap_refs[0][constant.CATEGORY].unique()
        mean_predictions = self.predict_proba(word_vectors)
        cons_predictions = []
        for resample_mfd_dict in self.bootstrap_refs:
            self.fit(resample_mfd_dict)
            all_predictions = self.predict_proba(word_vectors)
            cons_predictions.append(all_predictions)
        # lower_bound = []
        # upper_bound = []
        # for i,year in enumerate(word_vectors):
        #     year_predictions = [x[i] for x in cons_predictions]
        #     l_entry, u_entry = {}, {}
        #     n = len(cons_predictions)
        #     for cat in categories:
        #         cat_predictions = sorted([x[cat] for x in year_predictions])
        #         l_entry[cat] = cat_predictions[int(n*0.05)]
        #         u_entry[cat] = cat_predictions[int(n*0.95)]
        #     lower_bound.append(l_entry)
        #     upper_bound.append(u_entry)

        lower_bound, upper_bound = [{} for _ in mean_predictions], [{} for _ in mean_predictions]
        for cat in categories:
            y, yhat = [np.std([x[i][cat] for x in cons_predictions]) for i in range(len(word_vectors))], [x[cat] for x in mean_predictions]
            # interval = 1.96*mean_squared_error(y, yhat)
            interval = np.multiply(y, 1.96)
            lower_bound_arr, upper_bound_arr = np.subtract(yhat, interval), np.add(yhat, interval)
            for i,x in enumerate(lower_bound): x.update({cat: lower_bound_arr[i]})
            for i,x in enumerate(upper_bound): x.update({cat: upper_bound_arr[i]})
        return mean_predictions, lower_bound, upper_bound

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
            return [0]*len(X)
        return softmax_prob

    def predict_proba(self, data):
        result = []
        for d in data:
            distances = {k: -np.linalg.norm(d-v) for k, v in self.mean_vectors.items()}
            # distances = {k: cosine_similarity([d],[v])[0][0] for k, v in self.mean_vectors.items()}
            cat_names = sorted(self.mean_vectors.keys())
            probabilities = self.__calc_prob([distances[k] for k in cat_names])
            x_3 = dict(zip(cat_names, probabilities))
            result.append(x_3)
        return result

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

class FDACentroidModel(CentroidModel):

    name = 'FDA Centroid'

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, df):
        self.f = TruncatedSVD(n_components=self.n_components)
        new_df = df.copy()
        self.original_size = len(df[constant.VECTOR].values.tolist()[0])
        new_df[constant.VECTOR] = [np.array(x) for x in
                               self.f.fit_transform(new_df[constant.VECTOR].values.tolist(),
                                                    new_df[constant.CATEGORY].values.tolist())]
        super(FDACentroidModel, self).fit(new_df)

    def predict_proba(self, data):
        data_t = self.f.transform(data)
        return super(FDACentroidModel, self).predict_proba(data_t)

class MixtureModel(BaseModel):
    name = 'Mixture Model'

    def __init__(self, h):
        self.h = h

    def __calc_prob(self, X):
        # X_1 = np.exp(np.multiply(X,-1))
        X_1 = np.exp(X)
        softmax_prob = X_1 / np.sum(X_1, axis=0)
        if np.any(np.isnan(softmax_prob)):
            return [0]*len(X)
        return softmax_prob

    def fit(self, df):
        self.classifier = GaussianMixture(n_components=70,covariance_type='spherical')
        clusters = self.classifier.fit_predict(list(df[constant.VECTOR].values))
        labels = list(df[constant.CATEGORY].values)
        self.mat = pd.DataFrame(list(zip(clusters, labels)),
              columns=['Clusters','Labels'])

    def predict(self, data):
        assigned_clusters = self.classifier.predict(data)
        all_guesses = []
        for assigned_cluster in assigned_clusters:
            red_df = self.mat[self.mat['Clusters'] == assigned_cluster]
            all_guesses.append(Counter(red_df['Labels'].values.tolist()).most_common(1)[0][0])
        return all_guesses

class MixtureModel2(BaseModel):
    name = 'KDE Model'

    def __calc_prob(self, X):
        # X_1 = np.exp(np.multiply(X,-1))
        X_1 = np.exp(X)
        softmax_prob = X_1 / np.sum(X_1, axis=0)
        if np.any(np.isnan(softmax_prob)):
            return [0]*len(X)
        return softmax_prob

    def predict_proba(self, data):
        result = []
        for d in data:
            distances = {k: v.score_samples([d])[0] for k, v in self.classifiers.items()}
            # distances = {k: cosine_similarity([d],[v])[0][0] for k, v in self.mean_vectors.items()}
            cat_names = sorted(self.classifiers.keys())
            probabilities = self.__calc_prob([distances[k] for k in cat_names])
            x_3 = dict(zip(cat_names, probabilities))
            result.append(x_3)
        # assert all(sum(x.values()) > 0.9 for x in result)
        return result

    def fit(self, df):
        self.classifiers = {}
        for i in df[constant.CATEGORY].unique():
            c = GaussianMixture(n_components=10, covariance_type='spherical')
            c.fit(list(df[df[constant.CATEGORY] == i][constant.VECTOR].values))
            self.classifiers[i] = c
    
    def predict(self, data):
        all_guesses = []
        probs_data = self.predict_proba(data)
        for d in probs_data:
            all_guesses.append(max(d.keys(), key=(lambda key: d[key])))
        return all_guesses

class KDEModel(BaseModel):
    name = 'KDE Model'

    def __init__(self, h):
        self.h = h

    def __calc_prob(self, X):
        # X_1 = np.exp(np.multiply(X,-1))
        X_1 = np.exp(X)
        softmax_prob = X_1 / np.sum(X_1, axis=0)
        if np.any(np.isnan(softmax_prob)):
            return [0]*len(X)
        return softmax_prob

    def predict_proba(self, data):
        result = []
        for d in data:
            distances = {k: v.score_samples([d])[0] for k, v in self.classifiers.items()}
            # distances = {k: cosine_similarity([d],[v])[0][0] for k, v in self.mean_vectors.items()}
            cat_names = sorted(self.classifiers.keys())
            probabilities = self.__calc_prob([distances[k] for k in cat_names])
            x_3 = dict(zip(cat_names, probabilities))
            result.append(x_3)
        assert all(sum(x.values()) > 0.9 for x in result)
        return result

    def fit(self, df):
        self.classifiers = {}
        for i in df[constant.CATEGORY].unique():
            c = KernelDensity(bandwidth=self.h)
            c.fit(list(df[df[constant.CATEGORY] == i][constant.VECTOR].values))
            self.classifiers[i] = c
    
    def predict(self, data):
        all_guesses = []
        probs_data = self.predict_proba(data)
        for d in probs_data:
            all_guesses.append(max(d.keys(), key=(lambda key: d[key])))
        return all_guesses

# class SVCModel(BaseModel):
#     name = 'Custom Classifier Model'

#     def __init__(self):
#         self.t_classifier = LinearDiscriminantAnalysis(shrinkage=None, solver='eigen')
#         self.c_classifier = KNeighborsClassifier(n_neighbors=7, weights='distance')
#         # self.c_classifier = AdaBoostClassifier()
    
#     def normalize(self, word_vec):
#         norm=np.linalg.norm(word_vec)
#         if norm == 0: 
#             return word_vec
#         return word_vec/norm

#     def fit(self, df):
#         self.labels = sorted(df[constant.CATEGORY].unique())
#         x_t = self.t_classifier.fit_transform(list(df[constant.VECTOR].values),
#             list(df[constant.CATEGORY].values))
#         self.c_classifier.fit(x_t, list(df[constant.CATEGORY].values))
        

#     def predict(self, data):
#         data_t = self.t_classifier.transform([list(x) for x in data])
#         return self.c_classifier.predict(data_t)

#     def predict_proba(self, data):
#         all_preds = self.c_classifier.predict_proba([list(x) for x in data])
#         return [dict(zip(self.labels, x)) for x in all_preds]


# class TwoTierCentroidModel():
#     '''
#     Multi-layered centroid model
#     '''
#     name = 'TwoTier'

#     def __init__(self, t1Model, t2Model1, t2Model2):
#         self.t1 = t1Model
#         self.t2_pos = t2Model1
#         self.t2_neg = t2Model2
    
#     def predict_proba(self, data):
#         binary_proba = self.t1.predict_proba(data)
#         pos_cat_proba = [{k:binary_proba[i]['+']*v for k,v in x.items()} for i,x in enumerate(self.t2_pos.predict_proba(data))]
#         neg_cat_proba = [{k:binary_proba[i]['-']*v for k,v in x.items()} for i,x in enumerate(self.t2_neg.predict_proba(data))]
#         all_pred = [{**pos_cat_proba[i], **neg_cat_proba[i]} for i in range(len(pos_cat_proba))]
#         assert all(sum(x.values()) > 0.9 for x in all_pred)
#         return all_pred

#     def fit(self, df):
#         binary_df = df.copy()
#         binary_df[constant.CATEGORY] = ['+' if '+' in x else '-' for x in df[constant.CATEGORY].values]
#         pos_df = df.loc[df[constant.CATEGORY].isin({'care+','loyalty+','fairness+','authority+','sanctity+'})]
#         neg_df = df.loc[df[constant.CATEGORY].isin({'care-','loyalty-','fairness-','authority-','sanctity-'})]
#         self.t1.fit(binary_df)
#         self.t2_pos.fit(pos_df)
#         self.t2_neg.fit(neg_df)

#     def predict(self, data):
#         all_guesses = []
#         probs_data = self.predict_proba(data)
#         for d in probs_data:
#             all_guesses.append(max(d.keys(), key=(lambda key: d[key])))
#         return all_guesses

# class TwoTierModel():
#     '''
#     Multi-layered centroid model
#     '''
#     name = 'TwoTier'

#     def __init__(self, t1Model, t2Model):
#         self.t1 = t1Model
#         self.t2 = t2Model
    
#     def predict_proba(self, data):
#         binary_proba = self.t1.predict_proba(data)
#         pos_cat_proba = [{k:binary_proba[i]['+']*v if '+' in k else binary_proba[i]['-']*v for k,v in x.items()} for i,x in enumerate(self.t2.predict_proba(data))]
#         return pos_cat_proba

#     def fit(self, df):
#         binary_df = df.copy()
#         binary_df[constant.CATEGORY] = ['+' if '+' in x else '-' for x in df[constant.CATEGORY].values]
#         self.t1.fit(binary_df)
#         self.t2.fit(df)

#     def predict(self, data):
#         all_guesses = []
#         probs_data = self.predict_proba(data)
#         for d in probs_data:
#             all_guesses.append(max(d.keys(), key=(lambda key: d[key])))
#         return all_guesses