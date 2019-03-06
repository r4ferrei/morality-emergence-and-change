import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input', help="Results CSV.")
parser.add_argument('--out', help="CSV with average performance.")
args = parser.parse_args()

INPUT = args.input
OUT = args.out
assert(INPUT)
assert(OUT)

df = pd.read_csv(INPUT)
results = []

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
