import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--input', help="Input dic file.")
parser.add_argument('--out', help="Output csv file.")
args = parser.parse_args()

res = []

with open(args.input, 'r') as f:
    comments = 0
    for line in f:
        if comments < 2:
            if '%' == line.strip():
                comments += 1
            continue
        else:
            word, cat = line.strip().split('\t')
            word = word.strip()
            cat = int(cat.strip())
            assert(1 <= cat <= 10)
            res.append({
                'category': cat,
                'word': word
                })

df = pd.DataFrame(res)
df.to_csv(args.out, index=False)
