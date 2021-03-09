import csv
import pandas as pd
import os
import sys
from sklearn.metrics import classification_report, f1_score

# load csv files with results from different models
results = {}
directory = './results'
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        results[os.path.splitext(filename)[0]] = pd.read_csv(os.path.join(directory, filename))
        continue
    else:
        continue

# write predictions to series
pred_labels={}
test_labels={}

original_stdout = sys.stdout # Save a reference to the original standard output

# write output to file
with open('results_summary.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.

    for model in results:

        test_labels[model] = results[model].label
        pred_labels[model] = results[model].prediction

        print('_____________________________________________')
        print(model.upper())
        print(classification_report(test_labels[model],pred_labels[model], digits = 4))
        for average in ['micro', 'macro', 'weighted']:
            print('{} F1 score: {:.2%}'.format(average, f1_score(test_labels[model], pred_labels[model], average=average)))
        print('\nDistribution of predictions')
        print(pred_labels[model].value_counts())
        print()
        
    sys.stdout = original_stdout # Reset the standard output to its original value

    print('wrote summary of results to results_summary.txt')