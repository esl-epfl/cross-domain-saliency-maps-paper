import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme()

cm = 1 / 2.54

save_figure = False
fontsize = 11

fig_size = (7 * cm, 5.5 * cm)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

plt.rc('font', size = fontsize)          # controls default text sizes
plt.rc('axes', titlesize = fontsize)     # fontsize of the axes title
plt.rc('axes', labelsize = fontsize)    # fontsize of the x and y labels
plt.rc('xtick', labelsize = fontsize)    # fontsize of the tick labels
plt.rc('ytick', labelsize = fontsize)    # fontsize of the tick labels
plt.rc('legend', fontsize = fontsize)    # legend fontsize
plt.rc('figure', titlesize = fontsize)  # fontsize of the figure title

os.makedirs('./figures/insertion_deletion/', exist_ok=True)

change_del = np.zeros(3)
change_ins = np.zeros(3)
change_time_del = np.zeros(3)
change_time_ins = np.zeros(3)
change_rand_del = np.zeros(3)
change_rand_ins = np.zeros(3)

for i, test_subject_id in enumerate(range(1, 16)):
    y_pred_deletion = []
    y_pred_insertion = []

    y_pred_time_deletion = []
    y_pred_time_insertion = []

    y_pred_random_deletion = []
    y_pred_random_insertion = []

    for n_features in [4, 32, 64]:
        with open(f'./results/insertion_deletion/S{test_subject_id}_{n_features}_features.pickle', 'rb') as handle:
            results = pickle.load(handle)

        y_pred_deletion_tmp = results['y_pred_deletion'].flatten()
        y_pred_insertion_tmp = results['y_pred_insertion'].flatten()

        y_pred_time_deletion_tmp = results['y_pred_time_deletion'].flatten()
        y_pred_time_insertion_tmp = results['y_pred_time_insertion'].flatten()

        y_pred_random_deletion_tmp = results['y_pred_random_deletion'].flatten()
        y_pred_random_insertion_tmp = results['y_pred_random_insertion'].flatten()

        y_pred_deletion.append(y_pred_deletion_tmp)
        y_pred_insertion.append(y_pred_insertion_tmp)

        y_pred_time_deletion.append(y_pred_time_deletion_tmp)
        y_pred_time_insertion.append(y_pred_time_insertion_tmp)

        y_pred_random_deletion.append(y_pred_random_deletion_tmp)
        y_pred_random_insertion.append(y_pred_random_insertion_tmp)
            
        pred_baseline = results['pred_baseline'].flatten()

        y_pred = results['y_pred'].flatten()
        y_test = results['y_test'].flatten()

        baseline = np.abs(pred_baseline - y_pred) + 1e-3

    y_pred_deletion = np.stack(y_pred_deletion, axis = 0)
    y_pred_insertion = np.stack(y_pred_insertion, axis = 0)

    y_pred_time_deletion = np.stack(y_pred_time_deletion, axis = 0)
    y_pred_time_insertion = np.stack(y_pred_time_insertion, axis = 0)

    y_pred_random_deletion = np.stack(y_pred_random_deletion, axis = 0)
    y_pred_random_insertion = np.stack(y_pred_random_insertion, axis = 0)

    change_del += np.abs(y_pred_deletion - y_pred[None, :]).mean(axis = 1)
    change_ins += np.abs(y_pred_insertion - y_pred[None, :]).mean(axis = 1)

    change_time_del += np.abs(y_pred_time_deletion - y_pred[None, :]).mean(axis = 1)
    change_time_ins += np.abs(y_pred_time_insertion - y_pred[None, :]).mean(axis = 1)

    change_rand_del += np.abs(y_pred_random_deletion - y_pred[None, :]).mean(axis = 1)
    change_rand_ins += np.abs(y_pred_random_insertion - y_pred[None, :]).mean(axis = 1)

change_del /= 3
change_ins /= 3

change_time_del /= 3
change_time_ins /= 3

change_rand_del /= 3
change_rand_ins /= 3

print("====================================")
print("Frequency IG")
print("====================================")

print("IG deletion: ", change_del)
print("IG insertion: ",change_ins)

print("====================================")
print("Time IG")
print("====================================")
print("Time IG deletion: ",change_time_del)
print("Time IG insertion: ",change_time_ins)


print("====================================")
print("Random")
print("====================================")
print("Random deletion: ",change_rand_del)
print("Random insertion: ", change_rand_ins)

figsize = (5.5 * cm, 3 * cm)

## Deletion plots
plt.figure(figsize = figsize)
plt.plot(y_pred_deletion[0, :])
plt.plot(y_pred)
plt.savefig('./figures/insertion_deletion/deletion_example.svg', bbox_inches = 'tight')

plt.figure(figsize = figsize)
plt.plot(y_pred_random_deletion[0, :])
plt.plot(y_pred)
plt.savefig('./figures/insertion_deletion/random_deletion_example.svg', bbox_inches = 'tight')

plt.figure(figsize = figsize)
plt.plot(y_pred_time_deletion[0, :])
plt.plot(y_pred)
plt.savefig('./figures/insertion_deletion/time_deletion_example.svg', bbox_inches = 'tight')

## Insertion plots
plt.figure(figsize = figsize)
plt.plot(y_pred_insertion[0, :])
plt.plot(y_pred)
plt.savefig('./figures/insertion_deletion/insertion_example.svg', bbox_inches = 'tight')

plt.figure(figsize = figsize)
plt.plot(y_pred_random_insertion[0, :])
plt.plot(y_pred)
plt.savefig('./figures/insertion_deletion/random_insertion_example.svg', bbox_inches = 'tight')

plt.figure(figsize = figsize)
plt.plot(y_pred_time_insertion[0, :])
plt.plot(y_pred)
plt.savefig('./figures/insertion_deletion/time_insertion_example.svg', bbox_inches = 'tight')