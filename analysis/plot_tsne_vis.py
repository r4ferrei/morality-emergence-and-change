import embeddings
import constant
import models
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

load = False

emb_dict_all = None
if load:
    emb_dict_all,_ = embeddings.load_all(dir=constant.SGNS_DIR)

dim_reducer = TSNE(n_components=3,perplexity=30)
# dim_reducer = PCA(n_components=2)
mfd_dict = models.load_mfd_df(emb_dict=emb_dict_all,reload=load)
mfd_dict = mfd_dict[mfd_dict[constant.YEAR] == 1800]
# mfd_dict[constant.VECTOR] = [np.array(x) for x in clf.fit_transform(mfd_dict[constant.VECTOR].values.tolist(),
#                                               mfd_dict[constant.CATEGORY].values.tolist())]
tsne_results = dim_reducer.fit_transform(list(mfd_dict[constant.VECTOR].values))
mfd_dict['x'] = [pair[0] for pair in tsne_results]
mfd_dict['y'] = [pair[1] for pair in tsne_results]
mfd_dict['z'] = [pair[2] for pair in tsne_results]

cmap = plt.get_cmap('tab10')
base_cats = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']
all_points = []
all_labels = []
a3 = Axes3D(plt.figure())
for cat in mfd_dict[constant.CATEGORY].unique():
    word_df = mfd_dict[mfd_dict[constant.CATEGORY] == cat]
    base_cat = re.sub('[\+\-]+', '', cat)
    x, y, z, color = word_df['x'].values.tolist(), word_df['y'].values.tolist(), word_df['z'].values.tolist(), cmap(base_cats.index(base_cat))
    marker = 'v' if '-' in cat else 'o'
    points = a3.scatter(x, y, z, s=100, marker=marker, color=color, alpha=0.7)
    word_ind = 5
    word = word_df[constant.WORD].values.tolist()[word_ind]
    bbox_props = dict(boxstyle="square,pad=0.3", fc=color, alpha=0.2)
    a3.text(x[word_ind],y[word_ind],z[word_ind], word,size=15,bbox=bbox_props)
    all_points.append(points)
    all_labels.append(cat)
plt.legend(all_points, all_labels)
plt.show()