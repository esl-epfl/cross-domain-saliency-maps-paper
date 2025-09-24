import pickle
import numpy as np

with open('./results/ica_ig_insertion_deletion_results.pickle', 'rb') as handle:
    results = pickle.load(handle)

predictions = results['predictions']
prediction_insertions = results['prediction_insertions']
prediction_deletions = results['prediction_deletions']

prediction_random_insertions = results['prediction_random_insertions']
prediction_random_deletions = results['prediction_random_deletions']

print("Insertion: ", (predictions - prediction_insertions).mean() )
print("Deletion: ",(predictions - prediction_deletions).mean() )

print("Random Insertion: ",(predictions - prediction_random_insertions).mean() )
print("Random Deletion: ",(predictions - prediction_random_deletions).mean() )
