import subprocess
import os

import pandas as pd

dir = 'results-1990-intersection'

corpora = ['ngrams', 'coha', 'nyt']
models  = ['centroid', 'nb', 'knn_1', 'knn_5', 'knn_10', 'knn_15', 'kde']
tests   = ['categorization', 'polarity', 'null_test']

results = []

for corpus in corpora:
    for model in models:
        df = pd.read_csv(os.path.join(
            dir, "%s_%s.csv" % (model, corpus)))

        test_results = {}
        for test in tests:
            filtered = df[df['test'] == test]
            assert(len(set(filtered['test'].values)) == 1)
            assert(len(set(filtered['year'].values)) == 1)

            pred  = filtered['predicted_class']
            true_ = filtered['true_class']

            accuracy = (pred == true_).mean()
            test_results[test] = accuracy

        test_results['corpus'] = corpus
        test_results['model']  = model
        results.append(test_results)

df = pd.DataFrame(results)[['corpus', 'model'] + tests].round(2)
