import pickle
import constant
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


tsne = TSNE(n_components=2)
mfd_dict = pickle.load(open(constant.MFD_DF, 'rb'))
mfd_dict = mfd_dict[mfd_dict[constant.YEAR] == 1800]
tsne_results = tsne.fit_transform(list(mfd_dict[constant.VECTOR].values))
mfd_dict['x'] = [pair[0] for pair in tsne_results]
mfd_dict['y'] = [pair[1] for pair in tsne_results]

all_points = []
all_labels = []
for cat in mfd_dict[constant.CATEGORY].unique():
    word_df = mfd_dict[mfd_dict[constant.CATEGORY] == cat]
    if '-' in cat:
        points = plt.scatter(word_df['x'].values, word_df['y'].values, marker='o')
    else:
        points = plt.scatter(word_df['x'].values, word_df['y'].values, label='v')
    all_points.append(points)
    all_labels.append(cat)
plt.legend(all_points, all_labels)
plt.show()