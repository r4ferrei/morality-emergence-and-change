from sklearn.model_selection import LeaveOneOut
from statistics import mean
import models
import constant
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import embeddings


load = False
loo = LeaveOneOut()
emb_dict_all = None
if load:
    emb_dict_all,_ = embeddings.load_all(dir=constant.SGNS_DIR)
mfd_dict_all = {
    # 'Null': models.load_mfd_df_neutral(emb_dict=emb_dict_all,reload=load),
    'Seed': models.load_mfd_df(emb_dict=emb_dict_all,reload=load),
    'Binary': models.load_mfd_df_binary(emb_dict=emb_dict_all,reload=load)
}
all_models = [models.MixtureModel2()]
# all_models=[models.KDEModel(h=0.2640425317597489)]
years = mfd_dict_all['Seed'][constant.YEAR].unique()

df_list = []
for model in all_models:
    for test_name, mfd_dict in mfd_dict_all.items():
        # mfd_dict.drop(columns=[constant.VECTOR]).to_csv(test_name+'.csv')
        all_years = []
        for year in years:
            acc = 0
            mfd_dict_red = mfd_dict[mfd_dict[constant.YEAR] == year]
            mfd_dict_red = mfd_dict_red.reset_index(drop=True)
            for train,test in loo.split(mfd_dict_red):
                train_df, test_df = mfd_dict_red.loc[train], mfd_dict_red.loc[test]
                model.fit(train_df)
                pred = model.predict([test_df[constant.VECTOR][test[0]]])
                if pred[0] == test_df[constant.CATEGORY][test[0]]:
                    acc += 1
            all_years.append(acc/len(mfd_dict_red))
            print(acc/len(mfd_dict_red))
        print(model.name, mean(all_years), test_name)
        df_list.append({'Model':model.name, 'Score': mean(all_years), 'Test':test_name})

pd.DataFrame(df_list).to_csv('something.csv')