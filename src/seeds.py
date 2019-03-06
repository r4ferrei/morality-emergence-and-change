import os
import copy
import pandas as pd

# Assumed to correspond to order of category IDs in files, i.e. first
# category for 1 and 2, second for 3 and 4, and so on.
MFT_CATEGORY_NAMES = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']

NON_NEUTRAL_POLARITY_NAMES = ['+', '-']
NEUTRAL_POLARITY_NAME = '0'

def load(
        dir='data/seed-words',
        mfd_cleaned_file='cleaned_words.csv',
        neutral_file='neutral_words.csv'):
    '''
    Loads seed words for all MFT and neutral categories.

    Args:
        dir: directory containing seed word files.
        mfd_cleaned_file: name of file containing CSV with 'category' and
            'word' columns. The category column should be 1-indexed and
            follow the order in `MFT_CATEGORY_NAMES`.
        neutral_words: name of file containing a simple list of neutral words.

    Returns: multi-level dictionary where:
        - first level (polarity) contains keys '+', '-', '0';
        - second level for '+' and '-' contains keys `MFT_CATEGORY_NAMES`;
        - neutral polarity ('0') contains no second level index;
        - dictionary values are lists of words.
    '''

    res = {}
    res[NEUTRAL_POLARITY_NAME] = []
    for pol in NON_NEUTRAL_POLARITY_NAMES:
        res[pol] = {}
        for cat in MFT_CATEGORY_NAMES:
            res[pol][cat] = []

    res[NEUTRAL_POLARITY_NAME] = list(pd.read_csv(
        filepath_or_buffer = os.path.join(dir, neutral_file),
        header             = None,
        squeeze            = True))

    mfd_words = pd.read_csv(os.path.join(dir, mfd_cleaned_file))
    mfd_word_set = set()
    for _, row in mfd_words.iterrows():
        cat_id = int(row['category'])
        word = row['word']

        cat = MFT_CATEGORY_NAMES[(cat_id-1) // 2]
        pol = NON_NEUTRAL_POLARITY_NAMES[(cat_id-1) % 2]

        res[pol][cat].append(word)
        mfd_word_set.add(word)

    # Remove neutral words that also appear as MFD words.
    res[NEUTRAL_POLARITY_NAME] = list(
            set(res[NEUTRAL_POLARITY_NAME]) - mfd_word_set)

    return res

def seed_counts(seeds):
    '''
    Helper function that takes the seed words loaded by `load` and returns
    a dictionary where the lists of words are replaced by their lengths.
    '''

    res = copy.deepcopy(seeds)
    for k, v in res.items():
        if isinstance(v, list):
            res[k] = len(v)
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                assert(isinstance(v2, list))
                res[k][k2] = len(v2)
        else:
            assert(False)
    return res

def filter_by_vocab(seeds, vocab):
    '''
    Filters a seed word structure to those present in the vocabulary.

    Args:
        seeds: seed words structure as produced by `load`.
        vocab: list of words in the vocabulary.

    Returns: similar structure to `seeds` where each word list is intersected
    with `vocab`.
    '''

    vocab = set(vocab)

    res = copy.deepcopy(seeds)
    for k, v in res.items():
        if isinstance(v, list):
            res[k] = list(set(v) & vocab)
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                assert(isinstance(v2, list))
                res[k][k2] = list(set(v2) & vocab)
        else:
            assert(False)
    return res

def split_pos_neg(seeds):
    '''
    Given seeds from `load`, returns a tuple of:
        - tuple of string labels for the classes;
        - tuple with two lists of words:
          positive polarity seed words, and negative polarity seed words.
    '''

    pos = []
    neg = []

    plus = '+'
    minus = '-'

    assert(plus in NON_NEUTRAL_POLARITY_NAMES)
    assert(minus in NON_NEUTRAL_POLARITY_NAMES)

    for cat in MFT_CATEGORY_NAMES:
        for word in seeds[plus][cat]:
            pos.append(word)
        for word in seeds[minus][cat]:
            neg.append(word)

    return ('pos', 'neg'), (pos, neg)

def split_neutral_moral(seeds):
    '''
    Given seeds from `load`, returns tuple with:
    
        - tuple of string labels;
        - two lists of words:
          neutral seed words, and non-neutral (pos/neg polarity) seed words.
    '''

    neutral = []
    moral = []

    for word in seeds[NEUTRAL_POLARITY_NAME]:
        neutral.append(word)

    for pol in NON_NEUTRAL_POLARITY_NAMES:
        for cat in MFT_CATEGORY_NAMES:
            for word in seeds[pol][cat]:
                moral.append(word)

    return ('neutral', 'moral'), (neutral, moral)

def split_10_categories(seeds):
    '''
    Given seeds from `load`, produces a list of 10 lists, each being a word
    list from a polarity-category MFT pair.

    Returns: a tuple `labels, cats`, where `labels` is a list of label names
    for identification of the categories, and `cats` is a list of 10 word
    lists.
    '''

    words = {} # keys are concatention of multi-level keys, e.g. '0', '+care'
    for pol in NON_NEUTRAL_POLARITY_NAMES:
        for cat in MFT_CATEGORY_NAMES:
            words[pol+cat] = seeds[pol][cat].copy()
    return list(words.keys()), list(words.values())
