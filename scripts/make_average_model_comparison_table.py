import subprocess
import os

import pandas as pd
import numpy as np

dir = 'results-3'

CORPUS = 'ngrams'

models  = ['centroid', 'nb', 'knn_1', 'knn_5', 'knn_10', 'knn_15', 'kde']
tests   = ['categorization', 'polarity', 'null_test']

results = []

for model in models:
    df = pd.read_csv(os.path.join(
        dir, "%s_%s_classification.csv" % (model, CORPUS)))

    test_results = {}
    for test in tests:
        filtered = df[df['test'] == test]
        assert(len(set(filtered['test'].values)) == 1)

        years = set(filtered['year'].values)
        accs = []
        for year in years:
            now = filtered[filtered['year'] == year]
            acc = (now['predicted_class'] == now['true_class']).mean()
            accs.append(acc)

        test_results[test] = np.array(accs).mean()

    test_results['model']  = model
    results.append(test_results)

df = pd.DataFrame(results)[['model'] + tests].round(2)
