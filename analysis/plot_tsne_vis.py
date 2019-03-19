import embeddings
import constant
import models
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

load = False

emb_dict_all = None
if load:
    emb_dict_all,_ = embeddings.load_all(dir=constant.SGNS_DIR)

clf = LinearDiscriminantAnalysis(n_components=9, shrinkage=0.5, solver='eigen')
tsne = TSNE(n_components=2)
mfd_dict = models.load_mfd_df(emb_dict=emb_dict_all,reload=load)
mfd_dict = mfd_dict[mfd_dict[constant.YEAR] == 1800]
mfd_dict[constant.VECTOR] = [np.array(x) for x in clf.fit_transform(mfd_dict[constant.VECTOR].values.tolist(),
                                              mfd_dict[constant.CATEGORY].values.tolist())]
tsne_results = tsne.fit_transform(list(mfd_dict[constant.VECTOR].values))
mfd_dict['x'] = [pair[0] for pair in tsne_results]
mfd_dict['y'] = [pair[1] for pair in tsne_results]

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
    # plt.text(x[word_ind],y[word_ind],word,size=15,bbox=bbox_props)
    all_points.append(points)
    all_labels.append(cat)
plt.legend(all_points, all_labels)
plt.show()