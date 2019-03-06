import argparse
import os

import pandas as pd

import seeds
import embeddings
import categorization

parser = argparse.ArgumentParser()
parser.add_argument('--out', help='File for CSV output')
parser.add_argument('--device', help="'cuda' or 'cpu'")
parser.add_argument('--metric', help="'l2' or 'cosine'", default='l2')
parser.add_argument('--model', help="'exemplar' or 'centroid'",
        default='exemplar')
args = parser.parse_args()

out_filename = args.out
assert(out_filename)
if not out_filename.startswith('/dev/'):
    assert(not os.path.exists(out_filename))

if args.device:
    categorization.DEVICE = args.device

METRIC = args.metric

MODEL = args.model
assert(MODEL in ['centroid', 'exemplar'])
if MODEL == 'centroid':
    assert(METRIC == 'l2')

print("Loading historical embeddings.")
hist_embs, vocab = embeddings.load_all()

print("Loading and filtering seed words.")
#seed_words = seeds.filter_by_vocab(seeds.load(), vocab)
seed_words = seeds.load()

def accuracy(word_lists_per_class, embs):
    emb_mats = [
            embeddings.convert_words_to_embedding_matrix(words, embs)
            for words in word_lists_per_class
            ]
    if MODEL == 'exemplar':
        return categorization.true_loo_accuracy(emb_mats, metric=METRIC)
    else:
        return categorization.centroid_loo_accuracy(emb_mats)

results_df = pd.DataFrame()
years = reversed(sorted(list(hist_embs.keys()))) # later years first
for year in years:
    print("\n---------------------------------")
    print("Running analysis for year {}".format(year))

    curr_seeds = seeds.filter_by_vocab(seed_words, list(hist_embs[year].keys()))

    print("\nCategorization test")
    mft_acc = accuracy(
            seeds.split_10_categories(curr_seeds)[1], hist_embs[year])

    print("\nNull test")
    neutral_acc = accuracy(
            seeds.split_neutral_moral(curr_seeds), hist_embs[year])

    print("\nPolarity test")
    pol_acc = accuracy(
            seeds.split_pos_neg(curr_seeds), hist_embs[year])

    this_results = {
            'year'           : year,
            'categorization' : mft_acc,
            'null'           : neutral_acc,
            'polarity'       : pol_acc,
            }
    print("\nResult: {}".format(this_results))

    results_df = results_df.append(this_results, ignore_index=True)

results_df = results_df.set_index('year')
results_df.to_csv(out_filename, columns=['categorization', 'null', 'polarity'])
