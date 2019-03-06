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
    year_accs = {}
    for _, row in this_df.iterrows():
        if row['year'] not in year_accs:
            year_accs[row['year']] = {}
        year_accs[row['year']][row['test']] = row['average_accuracy']
    for year, res_dict in year_accs.items():
        results.append({
            'Year'                : year,
            'Model Name'          : model_name,
            'Categorization Test' : res_dict['categorization'],
            'Null Test'           : res_dict['null_test'],
            'Polarity Test'       : res_dict['polarity'],
            })

results_df = pd.DataFrame(results).sort_values(by='Year')
results_df.to_csv(OUT, index=False)
