import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', help="Results CSV.")
parser.add_argument('--end-year', help="CSV with final-year performance.")
parser.add_argument('--average', help="CSV with average performance.")
args = parser.parse_args()

INPUT = args.input
END_YEAR = args.end_year
AVERAGE = args.average
assert(INPUT)
assert(END_YEAR)
assert(AVERAGE)

df = pd.read_csv(INPUT)

end_year = max(df['Year'])
df_end_year = df[df['Year'] == end_year]
df_average = df.groupby(df['Model Name']).mean()
del df_average['Year']

df_end_year.to_csv(END_YEAR, index=False)
df_average.to_csv(AVERAGE, index=True) # model name
