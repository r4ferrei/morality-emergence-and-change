import numpy as np
import pandas as pd
import seeds
import constant
import pickle
import math
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.naive_bayes import GaussianNB
import itertools


def choose_mfd_df(switch, emb_dict_all):
    """
    Loader function to acquire the moral foundations dictionary dataframe

    :param switch: One of POLARITY, FINE-GRAINED, or RELEVANCE
    :param emb_dict_all: embedding dictionary
    :return: Pandas dataframe with data of moral foundations dictionary
    """
    if switch == 'POLARITY':
        mfd_dict = load_mfd_df_binary(emb_dict_all)
    elif switch == 'FINE-GRAINED':
        mfd_dict = load_mfd_df(emb_dict_all)
    elif switch == 'RELEVANCE':
        mfd_dict = load_mfd_df_neutral(emb_dict_all)
    else:
        raise NotImplementedError
    return mfd_dict


def load_mfd_df(emb_dict=None, **kwargs):
    """
    Categories are fine-grained
    Load the moral foundations dictionary and pull word representations for each seed word.
    Returns a dataframe with the following columns:
    WORD | VECTOR | CATEGORY | YEAR

    :emb_dict: Embedding dictionary mapping words to vectors
    :return: Pandas dataframe with data of moral foundations dictionary
    """
    s = seeds.load(constant.DATA_DIR, **kwargs)
    s_plus = {v: k + '+' for k, v in reshape(s['+'])}
    s_neg = {v: k + '-' for k, v in reshape(s['-'])}
    s_all = {**s_plus, **s_neg}
    items = []
    for word, cat in s_all.items():
        yr_items = []
        for year in emb_dict.keys():
            yr_emb_dict = emb_dict[year]
            if word in yr_emb_dict:
                yr_items.append({constant.WORD: word, constant.CATEGORY: cat,
                                 constant.YEAR: year, constant.VECTOR: yr_emb_dict[word]})
        if len(yr_items) == len(emb_dict.keys()):
            items += yr_items
    cat_df = pd.DataFrame(items)
    pickle.dump(cat_df, open(constant.MFD_DF, 'wb'))
    return cat_df


def load_mfd_df_binary(emb_dict=None):
    """
    Categories are positive and negative
    Load the moral foundations dictionary and pull word representations for each seed word.
    Returns a dataframe with the following columns:
    WORD | VECTOR | CATEGORY | YEAR

    :emb_dict: Embedding dictionary mapping words to vectors
    :return: Pandas dataframe with data of moral foundations dictionary
    """
    mfd_dict = load_mfd_df(emb_dict)
    mfd_dict[constant.CATEGORY] = ['+' if '+' in x else '-'
                                   for x in mfd_dict[constant.CATEGORY].values]
    return mfd_dict


def load_mfd_df_neutral(emb_dict=None, **kwargs):
    """
    Categories are positive and negative
    Load the moral foundations dictionary and pull word representations for each seed word.
    Returns a dataframe with the following columns:
    WORD | VECTOR | CATEGORY | YEAR

    :emb_dict: Embedding dictionary mapping words to vectors
    :return: Pandas dataframe with data of moral foundations dictionary
    """
    s = seeds.load(constant.DATA_DIR, **kwargs)
    s_plus = {v: '1' for k, v in reshape(s['+'])}
    s_neg = {v: '1' for k, v in reshape(s['-'])}
    s_neutral = {}
    for n_word in s['0']:
        if n_word in emb_dict[min(emb_dict.keys())]:
            s_neutral[n_word] = '0'
        if len(s_neutral) == len(s_plus) + len(s_neg):
            break
    s_all = {**s_plus, **s_neg, **s_neutral}
    items = []
    for word, cat in s_all.items():
        yr_items = []
        for year in emb_dict.keys():
            yr_emb_dict = emb_dict[year]
            if word in yr_emb_dict:
                yr_items.append({constant.WORD: word, constant.CATEGORY: cat,
                                 constant.YEAR: year, constant.VECTOR: yr_emb_dict[word]})
        if len(yr_items) == len(emb_dict.keys()):
            items += yr_items
    cat_df = pd.DataFrame(items)
    pickle.dump(cat_df, open(constant.MFD_DF_NEUTRAL, 'wb'))
    return cat_df

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

def reshape(shape_dict):
    list_of_lists = [zip([x[0]] * len(x[1]), x[1]) for x in shape_dict.items()]
    return list(itertools.chain(*list_of_lists))


def log_odds(pos_prob, neg_prob):
    return math.log(pos_prob / neg_prob)


class BaseModel:
    """
    Base model for implementation
    """
    def __init__(self):
        self.bootstrap_refs = []

    def __convert_proba_arr_to_dict(self, proba_arr):
        d = {}
        for i, x in enumerate(proba_arr):
            d[i + 1] = x
        return d

    def fit_bootstrap(self, mfd_dict, n=1000):
        assert n > 0
        self.fit(mfd_dict)
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
        lower_bound, upper_bound = [{} for _ in mean_predictions], [{} for _ in mean_predictions]
        for cat in categories:
            y, yhat = [np.std([x[i][cat] for x in cons_predictions]) for i in range(len(word_vectors))], \
                      [x[cat] for x in mean_predictions]
            interval = np.multiply(y, 1.96)
            lower_bound_arr, upper_bound_arr = np.subtract(yhat, interval), np.add(yhat, interval)
            for i, x in enumerate(lower_bound):
                x.update({cat: lower_bound_arr[i]})
            for i, x in enumerate(upper_bound):
                x.update({cat: upper_bound_arr[i]})
        return mean_predictions, lower_bound, upper_bound

    def fit(self, df):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    def predict_proba(self, data):
        raise NotImplementedError


class NBModel(BaseModel):
    """
    Naive bayes classifier for moral sentiments
    """

    name = 'Naive Bayes Model'

    def __init__(self):
        super(NBModel, self).__init__()
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
    """
    k nearest neighbours classifier for moral sentiments
    """

    name = 'kNN Model'

    def __init__(self, k=15):
        super(KNNModel, self).__init__()
        self.k = k
        self.c_classifier = KNeighborsClassifier(n_neighbors=self.k, p=1)

    def fit(self, df):
        self.labels = sorted(df[constant.CATEGORY].unique())
        self.c_classifier.fit(list(df[constant.VECTOR].values), list(df[constant.CATEGORY].values))

    def predict(self, data):
        return self.c_classifier.predict([list(x) for x in data])

    def predict_proba(self, data):
        all_preds = self.c_classifier.predict_proba([list(x) for x in data])
        return [dict(zip(self.labels, x)) for x in all_preds]


class CentroidModel(BaseModel):
    """
    Centroid classifier for moral sentiments
    """
    name = 'Centroid'

    def __calc_prob(self, X):
        X_1 = np.exp(X)
        softmax_prob = X_1 / np.sum(X_1, axis=0)
        if np.any(np.isnan(softmax_prob)):
            return [0] * len(X)
        return softmax_prob

    def predict_proba(self, data):
        result = []
        for d in data:
            distances = {k: -np.linalg.norm(d - v) for k, v in self.mean_vectors.items()}
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


class KDEModel(BaseModel):
    """
    Kernel Density Estimation model
    """
    name = 'KDE Model'

    def __init__(self, h):
        super(BaseModel, self).__init__()
        self.h = h

    def __calc_prob(self, X):
        X_1 = np.exp(X)
        softmax_prob = X_1 / np.sum(X_1, axis=0)
        if np.any(np.isnan(softmax_prob)):
            return [0] * len(X)
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
