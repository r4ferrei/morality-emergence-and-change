import os
import copy
import pandas as pd

# Assumed to correspond to order of category IDs in files, i.e. first
# category for 1 and 2, second for 3 and 4, and so on.
MFT_CATEGORY_NAMES = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']

NON_NEUTRAL_POLARITY_NAMES = ['+', '-']
NEUTRAL_POLARITY_NAME = '0'

def load(
        dir='local-data/',
        mfd_cleaned_file='mfd_v1.csv',
        neutral_file='words_sorted_by_valence_deviation.csv',
        remove_duplicates=False):
    '''
    Loads seed words for all MFT and neutral categories.

    Args:
        dir: directory containing seed word files.
        mfd_cleaned_file: name of file containing CSV with 'category' and
            'word' columns. The category column should be 1-indexed and
            follow the order in `MFT_CATEGORY_NAMES`.
        neutral_words: name of file containing a CSV ordered by "most neutral"
            word first, and containing column 'Word'.
        remove_duplicates: whether to ignore words that appear in more than
            one category.

    Returns: multi-level dictionary where:
        - first level (polarity) contains keys '+', '-', '0';
        - second level for '+' and '-' contains keys `MFT_CATEGORY_NAMES`;
        - neutral polarity ('0') contains no second level index;
        - dictionary values are lists of words.

    Words that appear in one MFD are removed from the neutral set.

    NOTE: the order of neutral words returned is important; do not change,
    since word will be filtered in that order in `filter_by_vocab` to match
    the size of the set of MFD seed words.
    '''

    res = {}
    res[NEUTRAL_POLARITY_NAME] = []
    for pol in NON_NEUTRAL_POLARITY_NAMES:
        res[pol] = {}
        for cat in MFT_CATEGORY_NAMES:
            res[pol][cat] = []

    neutral = pd.read_csv(os.path.join(dir, neutral_file))
    res[NEUTRAL_POLARITY_NAME] = list(neutral['Word'])

    mfd_words = pd.read_csv(os.path.join(dir, mfd_cleaned_file))
    for _, row in mfd_words.iterrows():
        cat_id = int(row['category'])
        word = row['word']

        if cat_id > 10: # skip general morality category from MFD v1
            assert(cat_id == 11)
            continue

        cat = MFT_CATEGORY_NAMES[(cat_id-1) // 2]
        pol = NON_NEUTRAL_POLARITY_NAMES[(cat_id-1) % 2]

        res[pol][cat].append(word)

    mfd_counts = {}
    for pol in NON_NEUTRAL_POLARITY_NAMES:
        for cat in MFT_CATEGORY_NAMES:
            for word in res[pol][cat]:
                mfd_counts[word] = mfd_counts.get(word, 0) + 1

    if remove_duplicates:
        for pol in NON_NEUTRAL_POLARITY_NAMES:
            for cat in MFT_CATEGORY_NAMES:
                res[pol][cat] = [w for w in res[pol][cat]
                        if mfd_counts[w] == 1]

    res[NEUTRAL_POLARITY_NAME] = [w for w in res[NEUTRAL_POLARITY_NAME]
            if w not in mfd_counts]

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
    Filters a seed word structure to those present in the vocabulary, and
    matches the number of neutral seed words to moral seed words.

    Args:
        seeds: seed words structure as produced by `load`.
        vocab: list of words in the vocabulary.

    Returns: similar structure to `seeds` where each word list is intersected
    with `vocab` and there are no more neutral seed words than pos+neg moral
    seed words.
    '''

    vocab = set(vocab)

    res = {}
    num_moral_seeds = 0
    for pol in NON_NEUTRAL_POLARITY_NAMES:
        res[pol] = {}
        for cat in MFT_CATEGORY_NAMES:
            res[pol][cat] = []
            for word in seeds[pol][cat]:
                if word in vocab:
                    res[pol][cat].append(word)
                    num_moral_seeds += 1

    res[NEUTRAL_POLARITY_NAME] = []
    num_neutral_seeds = 0
    for word in seeds[NEUTRAL_POLARITY_NAME]:
        if word in vocab:
            res[NEUTRAL_POLARITY_NAME].append(word)
            num_neutral_seeds += 1
            if num_neutral_seeds >= num_moral_seeds:
                break

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

def split_11_categories(seeds):
    '''
    Given seeds from `load`, produces a list of 11 lists, each being a word
    list from a polarity-category MFT pair or neutral.

    Returns: a tuple `labels, cats`, where `labels` is a list of label names
    for identification of the categories, and `cats` is a list of 10 word
    lists.
    '''

    words = {} # keys are concatention of multi-level keys, e.g. '0', '+care'
    for pol in NON_NEUTRAL_POLARITY_NAMES:
        for cat in MFT_CATEGORY_NAMES:
            words[pol+cat] = seeds[pol][cat].copy()

    words['neutral'] = seeds[NEUTRAL_POLARITY_NAME].copy()

    return list(words.keys()), list(words.values())
