import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import seeds
import embeddings

plt.ion()

EMBEDDINGS_DIR = 'data/hamilton-historical-embeddings/sgns/'
MFD_FILE       = 'mfd_v1.csv'

YEARS          = [1800, 1900, 1990]

WORDS          = [
        'slavery',
        'feminism',
        'gay',
        'gender',
        'war',
        'abortion',
        'homosexual',
        'wage',
        'democracy',
        'propaganda',
        'humiliation',
        'individuality',
        'pollution',
        'religion',
        ]

print("Loading historical embeddings.")
embs, vocab = embeddings.load_all(
        dir   = EMBEDDINGS_DIR,
        years = YEARS)

print("Loading and filtering seed words.")
seed_words = seeds.load(mfd_cleaned_file=MFD_FILE, remove_duplicates=False)
seed_words = seeds.filter_by_vocab(seed_words, vocab)

query_words = list(set(WORDS) & set(vocab))

seed_labels, all_seeds = seeds.split_11_categories(seed_words)

seeds_10_labels, seeds_10_classes = seeds.split_10_categories(seed_words)

_, seeds_3_classes = seeds.split_pos_neg_neutral(seed_words)

_, seeds_neutral_moral = seeds.split_neutral_moral(seed_words)

# Learn FDA transform for most recent year.

all_seed_embs = {}
seeds_3_classes_embs = {}
seeds_neutral_moral_embs = {}
seeds_10_classes_embs = {}
for year in YEARS:
    def doit(s):
        return [
                np.array([embs[year][w] for w in cat])
                for cat in s
                ]

    all_seed_embs[year] = doit(all_seeds)
    seeds_3_classes_embs[year] = doit(seeds_3_classes)
    seeds_neutral_moral_embs[year] = doit(seeds_neutral_moral)
    seeds_10_classes_embs[year] = doit(seeds_10_classes)

def get_fda(all_seed_embs, year):
    X = np.concatenate(all_seed_embs[year], axis=0)
    y = [i for i, cat in enumerate(all_seed_embs[year]) for _ in cat]
    fda = LinearDiscriminantAnalysis(
            solver='eigen', shrinkage=.5, n_components=2)
    fda.fit(X, y)
    return fda

print("Computing FDA transform.")
#fda = get_fda(seeds_3_classes_embs, YEARS[-1])
fdas = {}
for year in YEARS:
    fdas[year] = get_fda(seeds_3_classes_embs, year)
    #fdas[year] = get_fda(all_seed_embs, year)

# Compute centroids for each decade.

all_centroids = {}
neutral_moral_centroids = {}
ten_classes_centroids = {}
for year in YEARS:
    def doit(s):
        return [
                np.mean(cat_embs, axis=0)
                for cat_embs in s[year]
                ]

    all_centroids[year] = doit(all_seed_embs)
    neutral_moral_centroids[year] = doit(seeds_neutral_moral_embs)
    ten_classes_centroids[year] = doit(seeds_10_classes_embs)

# Transform centroids and queries for every decade using FDA.

transformed_centroids = {}
transformed_covs      = {}
transformed_queries   = {}

print("Transforming data.")
for year in YEARS:
    fda = fdas[year]
    transformed_centroids[year] = fda.transform(all_centroids[year])
    transformed_queries[year] = fda.transform(
            [embs[year][w] for w in query_words]
            )

    transformed_seeds = [fda.transform(s) for s in all_seed_embs[year]]
    transformed_covs[year] = [np.cov(ts.T) for ts in transformed_seeds]

# Make plots.

def make_confidence_ellipse(x, y, cov, nstd, color, alpha):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    ell = matplotlib.patches.Ellipse(xy=(np.mean(x), np.mean(y)),
                  width=w, height=h,
                  angle=theta, color=color, alpha=alpha)

    return ell

def classify(vec, centroids):
    dists = [np.linalg.norm(vec - c) for c in centroids]
    return np.argmin(dists)

fig, axes = plt.subplots(1, len(YEARS), figsize=(20, 8))

print("Plotting.")
for step, year in enumerate(YEARS):
    ax = axes[step]

    #plt.figure()
    ax.set_title("%d" % year)

    ax.set_xticks([], [])
    ax.set_yticks([], [])

    for i, centroids in enumerate(transformed_centroids[year]):
        label = seed_labels[i]
        if label == 'neutral':
            color = 'grey'
        elif label[0] == '+':
            color = 'green'
        elif label[0] == '-':
            color = 'red'
        else:
            assert(False)

        ax.plot([centroids[0]], [centroids[1]],
                marker='o',
                linestyle='',
                color=color)

        ell = make_confidence_ellipse(
                x = centroids[0], y = centroids[1],
                cov = transformed_covs[year][i],
                nstd = 1,
                color = color,
                alpha = .2)
        ax.add_artist(ell)

    #for i, label in enumerate(seed_labels):
    #    ax.annotate(
    #            label,
    #            xy = (
    #                transformed_centroids[year][i][0],
    #                transformed_centroids[year][i][1]
    #                ),
    #            xytext = (-20, 20),
    #            textcoords = 'offset points'
    #            )

    ax.plot(
            transformed_queries[year][:,0],
            transformed_queries[year][:,1],
            marker='o',
            linestyle='',
            color='black')

    for i, query in enumerate(query_words):
        neutral_moral_class = classify(
                embs[year][query], neutral_moral_centroids[year])
        if neutral_moral_class == 0:
            class_ = 'neutral'
        else:
            moral_class = classify(
                    embs[year][query], ten_classes_centroids[year])
            class_ = seeds_10_labels[moral_class]

        ax.annotate(
                "%s (%s)" % (query, class_),
                xy = (
                    transformed_queries[year][i][0],
                    transformed_queries[year][i][1]
                    ),
                xytext = (-30, 10),
                textcoords = 'offset points'
                )

    def expandrange(ab, r):
        a, b = ab
        l = b-a
        return (a - l*r, b + l*r)

    ax.set_xlim(*expandrange(ax.get_xlim(), .1))
    ax.set_ylim(*expandrange(ax.get_ylim(), .1))

plt.savefig('moral_map.pdf')
