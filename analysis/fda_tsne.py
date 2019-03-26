import argparse

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import seeds
import embeddings

parser = argparse.ArgumentParser()
parser.add_argument('--save', help="Filename to save plot.")
parser.add_argument('--query', help="Probe word to query out of training.")
args = parser.parse_args()

QUERY = args.query

COLORS = [
        'red', 'green', 'blue', 'orange', 'brown',
        'purple', 'black', 'pink', 'yellow', 'grey',
        ]

embs, vocab = embeddings.load_all(years=[1990])
seed_words = seeds.load(
        mfd_cleaned_file='mfd_v1.csv',
        remove_duplicates=True)
seed_words = seeds.filter_by_vocab(seed_words, vocab)

labels, words_per_class = seeds.split_10_categories(seed_words)

emb_mats = [
        embeddings.convert_words_to_embedding_matrix(words, embs[1990])
        for words in words_per_class
        ]

all_words = []
X = []
y = []
for i in range(len(words_per_class)):
    for w, e in zip(words_per_class[i], emb_mats[i]):
        all_words.append(w)
        X.append(e)
        y.append(i)

X = np.array(X)
y = np.array(y)

if QUERY:
    assert(QUERY in all_words)
    query_index = all_words.index(QUERY)

    def np_drop(A, i):
        return np.concatenate([A[:i], A[(i+1):]])

    X_q = np_drop(X, query_index)
    y_q = np_drop(y, query_index)
else:
    X_q = X
    y_q = y
    query_index = -1

fda = LinearDiscriminantAnalysis(solver='eigen', shrinkage=.5,
        n_components=9)
fda.fit(X_q, y_q)
X_fda = fda.transform(X)

tsne = TSNE(2)
X_tsne = tsne.fit_transform(X_fda)

fig, ax = plt.subplots()
ax.scatter(X_tsne[:,0], X_tsne[:,1],
        c = [COLORS[i] for i in y])

for i, word in enumerate(all_words):
    if i % 5 == 0 or i == query_index:
        label = word.upper() if i == query_index else word
        ax.annotate(label, (X_tsne[i,0], X_tsne[i,1]))

if args.save:
    plt.savefig(args.save)

plt.show()
