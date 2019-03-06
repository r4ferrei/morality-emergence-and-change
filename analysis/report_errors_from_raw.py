import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--file', help="Raw predictions CSV file.")
parser.add_argument('--year', type=int, help="Year to extract from.")
parser.add_argument('--out', help="Output CSV file.")
args = parser.parse_args()

FILE = args.file
YEAR = args.year
OUT = args.out
assert(FILE)
assert(YEAR)
assert(OUT)

preds = pd.read_csv(FILE)
preds = preds[preds['year'] == YEAR]
preds = preds[preds['true_class'] != preds['predicted_class']]
preds = preds.sort_values(by=['test', 'true_class', 'predicted_class',
    'instance'])
preds.to_csv(OUT, index=False,
        columns=['year', 'test', 'instance',
            'true_class', 'predicted_class'])
