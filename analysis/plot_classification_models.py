import argparse

import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--results-file', help="Path to resulst CSV file.")
parser.add_argument('--save-plots', action='store_true')
args = parser.parse_args()

RESULTS_FILE = args.results_file
assert(RESULTS_FILE)

SAVE_PLOTS = args.save_plots

plt.style.use('seaborn-white')
plt.ion()

df = pd.read_csv(RESULTS_FILE)

models = list(set(df['Model Name']))

df = df.set_index('Model Name')

tests = ['Categorization Test', 'Null Test', 'Polarity Test']
for test in tests:
    plt.figure()
    plt.title(test)
    plt.xlabel("Year")
    plt.ylabel("Accuracy")

    for model in models:
        results = df.loc[model, ['Year', test]]
        plt.plot(results['Year'], results[test])

    plt.legend(models)
    plt.show()

    if SAVE_PLOTS:
        plot_name = test.replace(' ', '_').lower()
        plt.savefig("results_%s.pdf" % plot_name)
