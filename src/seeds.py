import os
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
    for _, row in mfd_words.iterrows():
        cat_id = int(row['category'])
        word = row['word']

        cat = MFT_CATEGORY_NAMES[(cat_id-1) // 2]
        pol = NON_NEUTRAL_POLARITY_NAMES[(cat_id-1) % 2]

        res[pol][cat].append(word)

    return res
