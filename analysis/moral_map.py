import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import seeds
import embeddings

plt.ion()

matplotlib.rc('font', size=9)
matplotlib.rc('text', usetex=True)

EMBEDDINGS_DIR = 'data/hamilton-historical-embeddings/sgns/'
MFD_FILE       = 'mfd_v1.csv'

YEARS          = [1800, 1900, 1990]

WORDS          = [
        'slavery',
        'gay',
        'democracy',
        'attack',
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
fda = get_fda(seeds_3_classes_embs, YEARS[-1])
#fdas = {}
#for year in YEARS:
#    fdas[year] = get_fda(seeds_3_classes_embs, year)
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
    #fda = fdas[year]
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

# for max prob class
def classify_prob(vec, centroids):
    dists = [np.linalg.norm(vec - c) for c in centroids]
    dists = np.array(dists)
    return np.exp(-np.min(dists)) / np.sum(np.exp(-dists))

def format_class_name(c):
    mapfn = {
            '+care'      : 'care+',
            '+fairness'  : 'fairness+',
            '+loyalty'   : 'loyalty+',
            '+authority' : 'authority+',
            '+sanctity'  : 'sanctity+',
            '-care'      : 'harm-',
            '-fairness'  : 'cheating-',
            '-loyalty'   : 'betrayal-',
            '-authority' : 'subversion-',
            '-sanctity'  : 'degradation-',
            'neutral'    : 'irrelevant',
            }
    return mapfn[c]

fig, axes = plt.subplots(1, len(YEARS), figsize=(6.3, 2.2))

xlim = None
ylim = None

print("Plotting.")
for step, year in enumerate(YEARS):
    ax = axes[step]

    #plt.figure()
    ax.set_title("%ds" % year)

    # Annotate moral pos, neg, neutral.
    ax.annotate("\\textbf{Moral irrelevance}",
            xy=(.02, .93), xycoords='axes fraction')
    ax.annotate("\\textbf{Moral virtue}",
            xy=(.02, .03), xycoords='axes fraction')
    ax.annotate("\\textbf{Moral vice}",
            xy=(.69, .93), xycoords='axes fraction')

    ax.set_xticks([], [])
    ax.set_yticks([], [])

    # Only plot most recent seed words.
    for i, centroids in enumerate(transformed_centroids[YEARS[-1]]):
        label = seed_labels[i]
        if label == 'neutral':
            color = 'grey'
        elif label[0] == '+':
            color = 'green'
        elif label[0] == '-':
            color = 'red'
        else:
            assert(False)

        ax.scatter([centroids[0]], [centroids[1]],
                marker='x',
                s=7,
                c='dimgrey')

        ell = make_confidence_ellipse(
                x = centroids[0], y = centroids[1],
                # Only plot most recent seed words.
                cov = transformed_covs[YEARS[-1]][i],
                nstd = 1,
                color = color,
                alpha = .2)
        #ax.add_artist(ell)

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

    for i, query in enumerate(query_words):
        # Classify based on recent seed words.
        neutral_moral_class = classify(
                embs[year][query], neutral_moral_centroids[YEARS[-1]])
        if neutral_moral_class == 0:
            class_ = 'neutral'
        else:
            # Classify based on recent seed words.
            moral_class = classify(
                    embs[year][query], ten_classes_centroids[YEARS[-1]])
            class_ = seeds_10_labels[moral_class]

        marker = {
                'slavery'   : 'o',
                'gay'       : '^',
                'democracy' : 's',
                'attack'    : '1',
                }

        ax.scatter(
                transformed_queries[year][i,0],
                transformed_queries[year][i,1],
                marker=marker[query],
                s=12,
                c='black')

        class_ = format_class_name(class_)

        xytext = ({
            1800: {
                'slavery'   : (-25, -10),
                'gay'       : (-12, -10),
                'democracy' : (-33, 5),
                'attack'    : (1, 3),
                },
            1900: {
                'slavery'   : (-30, 6),
                'gay'       : (-10, 5),
                'democracy' : (-15, -9),
                'attack'    : (-42, 5),
                },
            1990: {
                'slavery'   : (-41, 7),
                'gay'       : (-10, 5),
                'democracy' : (-10, -10),
                'attack'    : (-59, 1),
                }
            })

        ax.annotate(
                "%s (%s)" % (query, class_),
                xy = (
                    transformed_queries[year][i][0],
                    transformed_queries[year][i][1]
                    ),
                xytext = xytext[year][query],
                textcoords = 'offset points'
                )

    def expandrange(ab, r):
        a, b = ab
        l = b-a
        return (a - l*r, b + l*r)

    if not xlim:
        xlim = (-0.11, 0.4)
        ylim = (-0.28, 0.17)

        #xlim = expandrange(ax.get_xlim(), .4)
        #ylim = expandrange(ax.get_ylim(), .4)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Crude control toggle for slow colour background.
    if True:

        # Color regions according to decision boundary.
        h = 0.002
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        xx, yy = np.meshgrid(
                np.arange(x_min, x_max, h),
                np.arange(y_min, y_max, h))

        transformed_3_centroids = [
                fda.transform(s) for s in seeds_3_classes_embs[YEARS[-1]]]

        Z = [classify(v, transformed_3_centroids) for v in
                np.c_[xx.ravel(), yy.ravel()]]
        Z = np.array(Z).reshape(xx.shape)

        Z2 = [classify_prob(v, transformed_3_centroids) for v in
                np.c_[xx.ravel(), yy.ravel()]]
        Z2 = np.array(Z2).reshape(xx.shape)

        assert(len(Z.shape) == 2)
        rgba = np.zeros(list(Z.shape) + [4])
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                green = np.array([21, 145, 0]) / 255
                red   = np.array([232, 6, 6]) / 255
                grey  = np.array([124, 124, 124]) / 255
                cols = [green, red, grey]

                col = cols[Z[i][j]]
                high = .8
                power = 1.7
                alpha = np.interp([Z2[i][j]],
                        [Z2.min(), Z2.max()],
                        [0, high])
                alpha = np.interp(alpha**power,
                        [0, high**power],
                        [0, high])

                rgba[i][j] = np.concatenate([list(col), alpha])

        cmap = matplotlib.colors.ListedColormap(['green', 'red', 'grey'])
        #ax.contourf(xx, yy, Z, alpha=.3)
        ax.imshow(
                rgba,
                #Z,
                #interpolation='none',
                #cmap=cmap, alpha=.3,
                extent=(x_min, x_max, y_min,y_max), origin='lower')

    #colours = ['green', 'red', 'grey']
    #for j in range(3):
    #    pts = fda.transform(seeds_3_classes_embs[YEARS[-1]][j])
    #    ax.plot(pts[:,0], pts[:,1], color=colours[j], marker='o',
    #            linestyle='')

plt.tight_layout(0)
plt.subplots_adjust(wspace=.05)

plt.savefig('moral_map.pdf', dpi=1000, bbox_inches='tight')
