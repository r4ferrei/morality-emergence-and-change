import argparse
import os

import pandas as pd

import seeds
import embeddings
import categorization

parser = argparse.ArgumentParser()
parser.add_argument('--out', help='File for CSV output')
parser.add_argument('--device', help="'cuda' or 'cpu'")
parser.add_argument('--model', help="'exemplar' or 'centroid'",
        default='exemplar')
parser.add_argument('--seeds',
        help="'fixed' (through time) or 'varying' (as available in time)")
parser.add_argument('--fda', action='store_true',
        help="Perform FDA on embedding space.")
parser.add_argument('--remove-duplicates', action='store_true',
        help="Remove seed words that belong to more than one category.")
args = parser.parse_args()

out_filename = args.out
assert(out_filename)
if not out_filename.startswith('/dev/'):
    assert(not os.path.exists(out_filename))

if args.device:
    categorization.DEVICE = args.device

MODEL = args.model
assert(MODEL in ['centroid', 'exemplar'])

SEEDS = args.seeds
assert(SEEDS in ['fixed', 'varying'])

FDA = args.fda
REMOVE_DUPLICATES = args.remove_duplicates

print("Loading historical embeddings.")
hist_embs, vocab = embeddings.load_all()

print("Loading and filtering seed words.")
seed_words = seeds.load(remove_duplicates=REMOVE_DUPLICATES)
if SEEDS == 'fixed':
    seed_words = seeds.filter_by_vocab(seed_words, vocab)
elif SEEDS == 'varying':
    pass
else:
    assert(False)

def predictions_df(word_lists_per_class, embs, seeds_for_fda):
    emb_mats = [
            embeddings.convert_words_to_embedding_matrix(words, embs)
            for words in word_lists_per_class
            ]

    if MODEL == 'exemplar':
        return categorization.kernel_loo_classification(
                emb_mats, word_lists_per_class,
                seeds_for_fda=seeds_for_fda,
                embs = (embs if seeds_for_fda else None))
    elif MODEL == 'centroid':
        return categorization.centroid_loo_classification(
                emb_mats, word_lists_per_class,
                seeds_for_fda=seeds_for_fda,
                embs = (embs if seeds_for_fda else None))
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
        _, all_seeds = seeds.split_11_categories(curr_seeds)

    tests = [ { 'name'     : 'categorization',
                'split_fn' : seeds.split_10_categories },
              { 'name'     : 'null_test', # 'null' is NaN for Pandas, headache.
                'split_fn' : seeds.split_neutral_moral },
              { 'name'     : 'polarity',
                'split_fn' : seeds.split_pos_neg } ]

    for test in tests:
        print("Running test '%s'" % test['name'])
        class_labels, words_per_class = test['split_fn'](curr_seeds)
        df = predictions_df(words_per_class, hist_embs[year],
                seeds_for_fda = (all_seeds if FDA else None))

        apply_class_labels(df, class_labels)
        df['test'] = test['name']
        df['year'] = year

        results_df = results_df.append(df, ignore_index=True)

    # Save every decade to see partial results early.
    cached = results_df.set_index('year')
    cached.to_csv(out_filename)
