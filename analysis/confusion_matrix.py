import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import plotting

parser = argparse.ArgumentParser()
parser.add_argument('--file', help="Raw classification results file.")
parser.add_argument('--yearly', action='store_true',
        help="Plots by decade in addition to aggregate.")
parser.add_argument('--save', action='store_true', help="Save plots.")
args = parser.parse_args()

FILE = args.file
SAVE = args.save
YEARLY = args.yearly
assert(FILE)

plt.ion()

df = pd.read_csv(FILE)

tests = set(df['test'])
years = set(df['year'])

def confusion_matrix(df, name):
    classes = list(set(df['predicted_class']) | set(df['true_class']))
    for class_id, class_ in enumerate(classes):
        for col in ['predicted_class', 'true_class']:
            df.loc[df[col] == class_, col] = class_id

    plotting.plot_confusion_matrix(
            y_true    = df['true_class'],
            y_pred    = df['predicted_class'],
            classes   = np.array(classes),
            normalize = True,
            title     = name)

    if SAVE:
        plt.savefig("%s.pdf" % name)

for test in tests:
    general_name = "confusion_matrix_%s" % test

    df_test = df[df['test'] == test].copy()
    confusion_matrix(df_test, general_name)

    if YEARLY:
        for year in years:
            df_test_year = df_test[df_test['year'] == year].copy()
            confusion_matrix(df_test_year, "%s_%d" % (general_name, year))
