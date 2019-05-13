import pandas as pd

null_df = pd.read_csv('NGRAM_null.csv')\
    .rename(index=str, columns={"pred": "1_pred", "orig_data": "1_origdata"})
binary_df = pd.read_csv('NGRAM_binary.csv')
null_df['+_pred'] = binary_df['pred'].values
null_df['+_origdata'] = binary_df['orig_data'].values

print(null_df.head())

