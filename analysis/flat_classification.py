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
args = parser.parse_args()

out_filename = args.out
assert(out_filename)
if not out_filename.startswith('/dev/'):
    assert(not os.path.exists(out_filename))

if args.device:
    categorization.DEVICE = args.device

MODEL = args.model
assert(MODEL in ['centroid', 'exemplar'])

print("Loading historical embeddings.")
hist_embs, vocab = embeddings.load_all()

print("Loading and filtering seed words.")
#seed_words = seeds.filter_by_vocab(seeds.load(), vocab)
seed_words = seeds.load()

def predictions_df(word_lists_per_class, embs):
    emb_mats = [
            embeddings.convert_words_to_embedding_matrix(words, embs)
            for words in word_lists_per_class
            ]
    if MODEL == 'exemplar':
        return categorization.kernel_loo_classification(
                emb_mats, word_lists_per_class)
    elif MODEL == 'centroid':
        return categorization.centroid_loo_classification(
                emb_mats, word_lists_per_class)
    else:
        assert(False)

def apply_class_labels(df, class_labels):
    col_names = ['true_class', 'predicted_class']
    for col in col_names:
        for i, label in enumerate(class_labels):
            df.loc[df[col] == i, col] = label

results_df = pd.DataFrame()
years = reversed(sorted(list(hist_embs.keys()))) # later years first

for year in years:
    print("\n---------------------------------")
    print("Running analysis for year {}".format(year))

    curr_seeds = seeds.filter_by_vocab(seed_words, list(hist_embs[year].keys()))

    tests = [ { 'name'     : 'categorization',
                'split_fn' : seeds.split_10_categories },
              { 'name'     : 'null_test', # 'null' is NaN for Pandas, headache.
                'split_fn' : seeds.split_neutral_moral },
              { 'name'     : 'polarity',
                'split_fn' : seeds.split_pos_neg } ]

    for test in tests:
        print("Running test '%s'" % test['name'])
        class_labels, words_per_class = test['split_fn'](curr_seeds)
        df = predictions_df(words_per_class, hist_embs[year])

        apply_class_labels(df, class_labels)
        df['test'] = test['name']
        df['year'] = year

        results_df = results_df.append(df, ignore_index=True)

results_df = results_df.set_index('year')
results_df.to_csv(out_filename)
