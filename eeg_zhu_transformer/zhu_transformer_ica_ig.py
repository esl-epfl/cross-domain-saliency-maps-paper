import numpy as np
import torch
from epilepsy2bids.annotations import Annotations
from epilepsy2bids.eeg import Eeg
from zhu.utils import load_model, load_thresh, get_dataloader, predict, get_predict_mask
import matplotlib.pyplot as plt
from tqdm import tqdm 

from sklearn.decomposition import FastICA

import pickle

import os

os.makedirs('./results/', exist_ok=True)

edf_root_folder = './data/eeg/'
edf_file = 'sub-00_ses-01_task-szMonitoring_run-02_eeg.edf'

eeg = Eeg.loadEdfAutoDetectMontage(edfFile = edf_root_folder + edf_file)

device = "cuda" if torch.cuda.is_available() else "cpu"

window_size_sec = 25
fs = eeg.fs
overlap_ratio = 1-1/window_size_sec
overlap_sec = window_size_sec * overlap_ratio

# Prepare model and data
model = load_model(window_size_sec, fs, device)
model.to(device)
prediction_threshold = load_thresh()

recording_duration = int(eeg.data.shape[1] / eeg.fs)

dataloader = get_dataloader(eeg.data, window_size_sec, fs)

model.eval()  
preds = []
with torch.no_grad():
    for i, data in tqdm(enumerate(dataloader)):
        data = data.float().to(device)
        outputs = model(data)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        predicted = probs[:, 1] > prediction_threshold
        preds += predicted.cpu().detach().numpy().tolist()
preds = np.array(preds)

index_of_interest = np.argwhere(preds == 1).flatten()[0] + 1
data_of_interest = dataloader.dataset[index_of_interest]

X = data_of_interest.numpy()

fastICA = FastICA(max_iter = 30_000, tol = 1e-9)
X_ica = fastICA.fit_transform(X.T)

print("Run ", fastICA.n_iter_, " iterations.")

n_iterations = 300

X_input = torch.from_numpy(X_ica).type(torch.float32).to(device)[None, ...]

zero_pads = torch.zeros((1, 19, 6400)).to(device)

coeffs = torch.from_numpy(fastICA.mixing_.T).type(torch.float32).to(device)
coeffs_baseline = torch.zeros((19, 19), dtype = torch.float32).type(torch.float32).to(device)
mean = torch.from_numpy(fastICA.mean_).type(torch.float32).to(device)

scaled_coeffs = [ coeffs_baseline + (float(i) / n_iterations) * (coeffs - coeffs_baseline) for i in range(1, n_iterations + 1)]

grad_sum = 0

for scaled_coeff in tqdm(scaled_coeffs):
    scaled_coeff.requires_grad = True
    scaled_input = torch.matmul(X_input, scaled_coeff) + mean
    scaled_input = torch.transpose(scaled_input, 1, 2)
    scaled_input = torch.cat([scaled_input, zero_pads], dim = 0)
    prediction = model(scaled_input)
    prob_prediction = torch.nn.functional.softmax(prediction, dim=1)
    prob_prediction[0, 1].backward()
    grad_sum += scaled_coeff.grad

grad_sum /= n_iterations
ig = (coeffs - coeffs_baseline) * grad_sum

results = {'X' : X,
           'X_ica' : X_ica,
           'ig' : ig.detach().cpu().numpy()}

with open('./results/ica_ig_results.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)