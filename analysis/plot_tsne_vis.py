import embeddings
import constant
import models
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re
from sklearn.neighbors import KernelDensity
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sns; sns.set(color_codes=True)

REP_COLOUR = 200

def use_matplot(mfd_dict):
    cmap = plt.get_cmap('tab10')
    base_cats = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']
    all_points = []
    all_labels = []
    for cat in mfd_dict[constant.CATEGORY].unique():
        word_df = mfd_dict[mfd_dict[constant.CATEGORY] == cat]
        base_cat = re.sub('[\+\-]+', '', cat)
        x, y, color = word_df['x'].values.tolist(), word_df['y'].values.tolist(), cmap(base_cats.index(base_cat))
        marker = 'v' if '-' in cat else 'o'
        points = plt.scatter(x=x, y=y, marker=marker, color=color, alpha=0.7, s=100)
        word_ind = 5
        word = word_df[constant.WORD].values.tolist()[word_ind]
        bbox_props = dict(boxstyle="square,pad=0.3", fc=color, alpha=0.2)
        plt.text(x[word_ind],y[word_ind],word,size=15,bbox=bbox_props)
        all_points.append(points)
        all_labels.append(cat)
    plt.legend(all_points, all_labels)

load = False

emb_dict_all = None
if load:
    emb_dict_all,_ = embeddings.load_all(dir=constant.SGNS_DIR)

dim_reducer = TSNE(n_components=2,perplexity=10)
# dim_reducer = PCA(n_components=2)
mfd_dict = models.load_mfd_df(emb_dict=emb_dict_all,reload=load)
mfd_dict = mfd_dict[mfd_dict[constant.YEAR] == 1800]
clf = LinearDiscriminantAnalysis(shrinkage='auto',solver='eigen')
# mfd_dict[constant.VECTOR] = [np.array(x) for x in clf.fit_transform(mfd_dict[constant.VECTOR].values.tolist(),mfd_dict[constant.CATEGORY].values.tolist())]
tsne_results = dim_reducer.fit_transform(list(mfd_dict[constant.VECTOR].values))
mfd_dict['x'] = [pair[0] for pair in tsne_results]
mfd_dict['y'] = [pair[1] for pair in tsne_results]
color_dict = {'care':'o','fairness':'v','loyalty':'s', 'authority':'p','sanctity':'D'}

label_patches = []
for cat in mfd_dict[constant.CATEGORY].unique():
    alpha = 0.2
    word_df = mfd_dict[mfd_dict[constant.CATEGORY] == cat]
    base_cat = re.sub('[\+\-]+', '', cat)
    cmap_name = 'Reds' if '+' in cat else 'Blues'
    marker = color_dict[base_cat]
    cmap = cm.get_cmap(cmap_name)
    cmap = ListedColormap(cmap(np.linspace(0.2, 1, 256)))
    sns.kdeplot(word_df['x'].values.tolist(), word_df['y'].values.tolist(), shade_lowest=False,
                cmap=cmap, alpha=alpha,shade=True, legend=True)
for cat in mfd_dict[constant.CATEGORY].unique():
    word_df = mfd_dict[mfd_dict[constant.CATEGORY] == cat]
    base_cat = re.sub('[\+\-]+', '', cat)
    cmap_name = 'Reds' if '+' in cat else 'Blues'
    cmap = cm.get_cmap(cmap_name)
    marker = color_dict[base_cat]
    edge = 256/(list(color_dict.keys()).index(base_cat)+1)
    print(edge)
    sns.scatterplot(word_df['x'].values.tolist(), word_df['y'].values.tolist(), label=cat, color=cmap(0.6 + 0.4/(list(color_dict.keys()).index(base_cat)+1)), marker=marker,s=80)
plt.legend()
plt.show()