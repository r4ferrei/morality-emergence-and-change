import argparse
import pandas as pd
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--files-dict',
        help="String representation of dictionary model -> results CSV file.")
parser.add_argument('--out', help="Output file.")
args = parser.parse_args()

FILES_DICT = args.files_dict
assert(FILES_DICT)

OUT = args.out
assert(OUT)

files_dict = ast.literal_eval(FILES_DICT)

results = []
for model_name, filename in files_dict.items():
    this_df = pd.read_csv(filename)

    accs = {}
    for _, row in this_df.iterrows():
        accs[row['test']] = row['average_accuracy']

    results.append({
        'Model Name'          : model_name,
        'Categorization Test' : accs['categorization'],
        'Null Test'           : accs['null_test'],
        'Polarity Test'       : accs['polarity'],
        })

results_df = pd.DataFrame(results)
results_df.to_csv(OUT, index=False)
