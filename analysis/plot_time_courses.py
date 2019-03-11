import pickle
import constant
import embeddings
import os
import pandas as pd
from models import CentroidModel,TwoTierCentroidModel
import tabulate
import matplotlib.pyplot as plt

test_words = ['slavery', 'feminism', 'racism', 'abortion', 'automation', 'computer', 'diversity', 'education', 'electric', 'engineer', 'environment', 'feminism', 'immigration', 'information', 'machine', 'mechanic', 'nazism', 'phone', 'privacy', 'racism', 'religion', 'robotics', 'sexism', 'technology']
years = constant.ALL_YEARS

# test_df = []
# emb_dict,_ = embeddings.load_all(years=years, dir=constant.SGNS_DIR)
# for year in years:
#     yr_emb_dict = emb_dict[year]
#     for i,word in enumerate(test_words):
#         if word in yr_emb_dict:
#             test_df.append({constant.YEAR:year,constant.WORD:word,constant.VECTOR:yr_emb_dict[word]})
# pickle.dump(pd.DataFrame(test_df),open(os.path.join(constant.DATA_DIR, 'words.pkl'), 'wb'))

test_df = pickle.load(open(os.path.join(constant.DATA_DIR, 'words.pkl'), 'rb'))
c = TwoTierCentroidModel()
mfd_dict = pickle.load(open(constant.MFD_DF, 'rb'))
reduced_mfd_dict = mfd_dict[mfd_dict[constant.YEAR] == 1990]
c.fit(reduced_mfd_dict)

for word in test_words:
    word_df = test_df[test_df[constant.WORD] == word]
    all_predictions = c.predict_proba(word_df[constant.VECTOR])
    for cat in mfd_dict[constant.CATEGORY].unique():
        cat_prediction = [x[cat] for x in all_predictions]
        if '-' in cat:
            plt.plot(word_df[constant.YEAR].values, cat_prediction, label=cat, ls='--')
        else:
            plt.plot(word_df[constant.YEAR].values, cat_prediction, label=cat)
    plt.title(word)
    plt.ylabel('Category Proba')
    plt.xlabel('Years')
    plt.legend()
    plt.savefig(os.path.join(constant.DATA_DIR,word+'.png'))
    plt.clf()

    