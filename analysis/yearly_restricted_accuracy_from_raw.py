import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input', help="Results CSV.")
parser.add_argument('--restricted',
        help="Results with vocabulary to restrict output to.")
parser.add_argument('--out', help="CSV with average performance.")
args = parser.parse_args()

INPUT = args.input
RESTRICTED = args.restricted
OUT = args.out
assert(INPUT)
assert(RESTRICTED)
assert(OUT)

df = pd.read_csv(INPUT)
results = []

restricted = pd.read_csv(RESTRICTED)
vocab = set(restricted['instance'])

df['available'] = False
for i, row in df.iterrows():
    if row['instance'] in vocab:
        df.loc[i, 'available'] = True
df = df[df['available']]

tests = set(df['test'])
years = set(df['year'])

for test in tests:
    df_test = df[df['test'] == test]
    for year in years:
        df_test_year = df_test[df_test['year'] == year]
        acc = (
                df_test_year['true_class'] ==
                df_test_year['predicted_class']).mean()
        results.append({
            'year'             : year,
            'test'             : test,
            'average_accuracy' : acc,
            })

results_df = pd.DataFrame(results).set_index('test')
results_df.to_csv(OUT)
