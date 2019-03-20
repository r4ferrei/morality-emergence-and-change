import argparse
import os

import pandas as pd

import seeds
import embeddings
import categorization

parser = argparse.ArgumentParser()
parser.add_argument('--out', help='File for CSV output')
parser.add_argument('--device', help="'cuda' or 'cpu'")
args = parser.parse_args()

out_filename = args.out
assert(out_filename)
if not out_filename.startswith('/dev/'):
    assert(not os.path.exists(out_filename))

if args.device:
    categorization.DEVICE = args.device

print("Loading historical embeddings.")
hist_embs, vocab = embeddings.load_all()

print("Loading and filtering seed words.")
seed_words = seeds.load(
        mfd_cleaned_file='mfd_v1.csv',
        remove_duplicates=False)
seed_words = seeds.filter_by_vocab(seed_words, vocab)

def predictions_df(
        words_per_class_10, class_labels_10,
        words_per_class_2,  class_labels_2,
        embs):
    emb_mats_10 = [
            embeddings.convert_words_to_embedding_matrix(words, embs)
            for words in words_per_class_10
            ]

    pos_cats = []
    neg_cats = []

    for i in range(len(words_per_class_10)):
        if class_labels_10[i][0] == '+':
            pos_cats.append(i)
        elif class_labels_10[i][0] == '-':
            neg_cats.append(i)
        else:
            assert(False)

    return categorization.centroid_tiered(
            emb_mats        = emb_mats_10,
            words_per_class = words_per_class_10,
            pos_cats        = pos_cats,
            neg_cats        = neg_cats)

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

    curr_seeds = seed_words

    class_labels_10, words_per_class_10 = seeds.split_10_categories(curr_seeds)
    class_labels_2,  words_per_class_2  = seeds.split_pos_neg(curr_seeds)

    df = predictions_df(
            words_per_class_10 = words_per_class_10,
            class_labels_10    = class_labels_10,
            words_per_class_2  = words_per_class_2,
            class_labels_2     = class_labels_2,
            embs               = hist_embs[year])
    df['year'] = year
    df['test'] = 'categorization'
    apply_class_labels(df, class_labels_10)

    results_df = results_df.append(df, ignore_index=True)

    # Save every decade to see partial results early.
    cached = results_df.set_index('year')
    cached.to_csv(out_filename)
