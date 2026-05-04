import pickle
import numpy as np

import pickle
import numpy as np
from collections import defaultdict

aggregated_results = defaultdict(lambda: defaultdict(list))

for test_subject_id in range(1, 16):

    with open(f'./results/time_perturbation_test/S{test_subject_id}.pickle', 'rb') as handle:
        results = pickle.load(handle)
    
    for noise_level in results.keys():
        for key in results[noise_level].keys():
            values = np.array(results[noise_level][key])
            aggregated_results[noise_level][key].append(values.mean())  # store per-subject mean

# Report aggregated results
for noise_level in aggregated_results.keys():
    print(f"== {noise_level} ==")
    for key in aggregated_results[noise_level].keys():
        subject_means = np.array(aggregated_results[noise_level][key])
        print(f"  {key}: {subject_means.mean():.4f} (+/- {subject_means.std():.4f})")