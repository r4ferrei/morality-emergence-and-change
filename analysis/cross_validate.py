from sklearn.model_selection import LeaveOneOut
from statistics import mean
import models
import constant
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import embeddings

loo = LeaveOneOut()
# emb_dict_all,vocab = embeddings.load_all(dir='E:/sgns')
# mfd_dict = models.load_mfd_df(emb_dict_all)
mfd_dict = pickle.load(open(constant.MFD_DF, 'rb'))
all_models = [models.TSNECentroidModel()]

for model in all_models:
    all_years = []
    preds = []
    truth = []
    # for year in mfd_dict[constant.YEAR].unique():
    for year in [1990]:
        acc = 0
        mfd_dict_red = mfd_dict[mfd_dict[constant.YEAR] == year]
        mfd_dict_red = mfd_dict_red.reset_index(drop=True)
        for train,test in loo.split(mfd_dict_red):
            train_df, test_df = mfd_dict_red.loc[train], mfd_dict_red.loc[test]
            model.fit(train_df)
            pred = model.predict([test_df[constant.VECTOR][test[0]]])
            if pred[0] == test_df[constant.CATEGORY][test[0]]:
                acc += 1
            preds.append(pred)
            truth.append(test_df[constant.CATEGORY][test[0]])
        all_years.append(acc/len(mfd_dict_red))
    print(model.name, mean(all_years))
