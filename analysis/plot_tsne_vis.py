import embeddings
import constant
import models
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re

load = False

emb_dict_all = None
if load:
    emb_dict_all,_ = embeddings.load_all(dir=constant.SGNS_DIR)

tsne = TSNE()
mfd_dict = models.load_mfd_df(emb_dict=emb_dict_all,reload=load)
mfd_dict = mfd_dict[mfd_dict[constant.YEAR] == 1800]
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
    if '-' in cat:
        points = plt.scatter(word_df['x'].values, word_df['y'].values, marker='v', color=cmap(base_cats.index(base_cat)))
    else:
        points = plt.scatter(word_df['x'].values, word_df['y'].values, marker='o', color=cmap(base_cats.index(base_cat)))
    all_points.append(points)
    all_labels.append(cat)
plt.legend(all_points, all_labels)
plt.show()