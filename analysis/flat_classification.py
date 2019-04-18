import argparse
import os
import pickle

import pandas as pd

import seeds
import embeddings
import categorization

parser = argparse.ArgumentParser()
parser.add_argument('--out', help='File for CSV output')
parser.add_argument('--device', help="'cuda' or 'cpu'")
parser.add_argument('--model', help="'exemplar', 'centroid', 'knn', 'gnb'",
        default='exemplar')
parser.add_argument('--seeds',
        help="'fixed' (through time) or 'varying' (as available in time)")
parser.add_argument('--fda', action='store_true',
        help="Perform FDA on embedding space.")
parser.add_argument('--remove-duplicates', action='store_true',
        help="Remove seed words that belong to more than one category.")
parser.add_argument('--mfd-file', help="Name of MFD file in seeds dir.")
parser.add_argument('--widths', help="File with pre-trained kernel widths.")
parser.add_argument('--corpus', help="one of 'ngrams', 'coha', 'nyt'")
parser.add_argument('--year', type=int, help="Analysis in specific year")
parser.add_argument('--intersect', action='store_true',
        help="Use intersection vocabulary across n-grams, coha, nyt corpora.")
parser.add_argument('--k', type=int, help="k parameter for kNN classifier.")
args = parser.parse_args()

out_filename = args.out
assert(out_filename)
if not out_filename.startswith('/dev/'):
    assert(not os.path.exists(out_filename))

if args.device:
    categorization.DEVICE = args.device

MODEL = args.model
assert(MODEL in ['centroid', 'exemplar', 'knn', 'gnb'])

SEEDS = args.seeds
assert(SEEDS in ['fixed', 'varying'])

FDA = args.fda
REMOVE_DUPLICATES = args.remove_duplicates

MFD_FILE = args.mfd_file
assert(MFD_FILE)

WIDTHS = args.widths

K = args.k
if MODEL == 'knn': assert(K)

CORPUS = args.corpus
assert(CORPUS in ['ngrams', 'coha', 'nyt'])

YEAR = args.year
INTERSECT = args.intersect

def get_years(default):
    return [YEAR] if YEAR else default

print("Loading historical embeddings.")
if CORPUS == 'ngrams':
    hist_embs, vocab = embeddings.load_all(
            dir   = 'data/hamilton-historical-embeddings/sgns/',
            years = get_years(list(range(1800, 1991, 10))))
elif CORPUS == 'coha':
    hist_embs, vocab = embeddings.load_all(
            dir   = 'data/coha-historical-embeddings/sgns/',
            years = get_years(list(range(1810, 2001, 10))))
elif CORPUS == 'nyt':
    hist_embs, vocab = embeddings.load_all_nyt(
            dir   = 'data/nyt/',
            years = get_years(list(range(1990, 2010))))

if INTERSECT:
    assert(YEAR == 1990) # only shared year
    with open('local-data/1990-vocab-intersection.pkl', 'rb') as f:
        vocab = pickle.load(f)

print("Loading and filtering seed words.")
seed_words = seeds.load(
        mfd_cleaned_file=MFD_FILE,
        remove_duplicates=REMOVE_DUPLICATES)
if SEEDS == 'fixed':
    seed_words = seeds.filter_by_vocab(seed_words, vocab)
elif SEEDS == 'varying':
    pass
else:
    assert(False)

if WIDTHS:
    kernel_widths = pd.read_csv(WIDTHS)
    kernel_widths = kernel_widths.set_index('test', drop=True)
    print("Pre-trained kernel widths:")
    for test in ['categorization', 'null_test', 'polarity']:
        print("{}: {}".format(test, kernel_widths.loc[test, 'kernel_width']))

def predictions_df(word_lists_per_class, embs, seeds_for_fda, kernel_width):
    emb_mats = [
            embeddings.convert_words_to_embedding_matrix(words, embs)
            for words in word_lists_per_class
            ]

    if MODEL == 'exemplar':
        return categorization.kernel_loo_classification(
                emb_mats, word_lists_per_class,
                seeds_for_fda=seeds_for_fda,
                embs = (embs if seeds_for_fda else None),
                kernel_width = kernel_width)
    elif MODEL == 'centroid':
        return categorization.centroid_loo_classification(
                emb_mats, word_lists_per_class,
                seeds_for_fda=seeds_for_fda,
                embs = (embs if seeds_for_fda else None))
    elif MODEL == 'knn':
        return categorization.knn_loo_classification(
                emb_mats, word_lists_per_class, k=K)
    elif MODEL == 'gnb':
        return categorization.gaussian_nb_loo_classification(
                emb_mats, word_lists_per_class)
    else:
        assert(False)

def apply_class_labels(df, class_labels):
    col_names = ['true_class', 'predicted_class']
    for col in col_names:
        for i, label in enumerate(class_labels):
            df.loc[df[col] == i, col] = label

def labelled_seed_dataset_from_split(seed_words, embs):
    X = []
    y = []
    for i, words in enumerate(seed_words):
        emb_mat = embeddings.convert_words_to_embedding_matrix(words, embs)
        for vec in emb_mat:
            X.append(vec)
            y.append(i)
    return X, y

results_df = pd.DataFrame()
years = reversed(sorted(list(hist_embs.keys()))) # later years first

for year in years:
    print("\n---------------------------------")
    print("Running analysis for year {}".format(year))

    if SEEDS == 'fixed':
        curr_seeds = seed_words
    elif SEEDS == 'varying':
        curr_seeds = seeds.filter_by_vocab(
                seed_words, list(hist_embs[year].keys()))
    else:
        assert(False)

    if FDA:
        #_, all_seeds = seeds.split_11_categories(curr_seeds)
        _, all_seeds = seeds.split_10_categories(curr_seeds)

    tests = [ { 'name'     : 'categorization',
                'split_fn' : seeds.split_10_categories },
              { 'name'     : 'null_test', # 'null' is NaN for Pandas, headache.
                'split_fn' : seeds.split_neutral_moral },
              { 'name'     : 'polarity',
                'split_fn' : seeds.split_pos_neg } ]

    if FDA:
        tests = [ { 'name'     : 'categorization',
                    'split_fn' : seeds.split_10_categories }
                    ]

    for test in tests:
        print("Running test '%s'" % test['name'])
        class_labels, words_per_class = test['split_fn'](curr_seeds)
        df = predictions_df(words_per_class, hist_embs[year],
                seeds_for_fda = (all_seeds if FDA else None),
                kernel_width = (
                    kernel_widths.loc[test['name'], 'kernel_width']
                    if WIDTHS
                    else None))

        apply_class_labels(df, class_labels)
        df['test'] = test['name']
        df['year'] = year

        results_df = results_df.append(df, ignore_index=True)

    # Save every decade to see partial results early.
    cached = results_df.set_index('year')
    cached.to_csv(out_filename)
