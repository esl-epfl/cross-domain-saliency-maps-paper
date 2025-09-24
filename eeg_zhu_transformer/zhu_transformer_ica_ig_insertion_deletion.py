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

def find_edf_files(root_dir):
    edf_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".edf"):
                edf_files.append(os.path.join(root, file))
    
    return edf_files

def isolateICComponent(eeg_signal, ica, componentIndex):
    X_ica = ica.transform(eeg_signal.T)

    componentOfInterest = X_ica[:, componentIndex]

    isolatedICA = np.zeros_like(X_ica)
    isolatedICA[:, componentIndex] = componentOfInterest
    
    isolatedComponent = ica.inverse_transform(isolatedICA)

    return isolatedComponent.T[None, ...]

def predict_on_isolated_components(X_isolated, X_deleted, model, device):
    X_isolated = torch.from_numpy(X_isolated).to(device).type(torch.float32)
    X_isolated = torch.cat([X_isolated, zero_pads], dim = 0)
    isolated_prediction = model(X_isolated)
    isolated_prediction = torch.nn.functional.softmax(isolated_prediction, dim=1)[0, 1]

    X_deleted = torch.from_numpy(X_deleted).to(device).type(torch.float32)
    X_deleted = torch.cat([X_deleted, zero_pads], dim = 0)
    deleted_prediction = model(X_deleted)
    deleted_prediction = torch.nn.functional.softmax(deleted_prediction, dim=1)[0, 1]

    X_tmp = torch.from_numpy(X[None, ...]).to(device).type(torch.float32)
    X_tmp = torch.cat([X_tmp, zero_pads], dim = 0)
    original_prediction = model(X_tmp)
    original_prediction = torch.nn.functional.softmax(original_prediction, dim=1)[0, 1]

    return isolated_prediction, deleted_prediction, original_prediction

os.makedirs('./results/', exist_ok=True)

dataset_root_folder = './data/bids/siena/'

all_files = find_edf_files(dataset_root_folder)

n_files = len(all_files)

all_predictions = np.zeros((n_files))
all_predictions_deletion = np.zeros((n_files))
all_predictions_insertion = np.zeros((n_files))

all_predictions_random_deletion = np.zeros((n_files))
all_predictions_random_insertion = np.zeros((n_files))

for i in range(n_files):
    print(f"Processing file {i} out of {n_files}...")
    edf_filepath = all_files[i]
    edf_root_folder, edf_file = os.path.split(edf_filepath)


    keywords = edf_file.split("_")
    subject = keywords[0]
    session = keywords[1]
    run = keywords[3]

    eeg = Eeg.loadEdfAutoDetectMontage(edfFile = edf_root_folder + "/" + edf_file)

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
        for j, data in tqdm(enumerate(dataloader)):
            data = data.float().to(device)
            outputs = model(data)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            predicted = probs[:, 1] > prediction_threshold
            preds += predicted.cpu().detach().numpy().tolist()
    preds = np.array(preds)

    index_of_interest = np.argwhere(preds == 1).flatten()[0] + 1
    data_of_interest = dataloader.dataset[index_of_interest]

    X = data_of_interest.numpy()

    fastICA = FastICA(max_iter = 1_000, tol = 1e-9, random_state = 42)
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

    ica_ig = np.sum(ig.detach().cpu().numpy(), axis = 1)
    maxIG = np.argmax(ica_ig)
    
    # Isolate max IG
    X_isolated = isolateICComponent(X, fastICA, maxIG)
    X_deleted = X - X_isolated
    isolated_prediction, deleted_prediction, original_prediction = predict_on_isolated_components(X_isolated, X_deleted, model, device)
    
    all_predictions[i] = original_prediction.detach().cpu().numpy()
    all_predictions_deletion[i] = deleted_prediction.detach().cpu().numpy()
    all_predictions_insertion[i] = isolated_prediction.detach().cpu().numpy()

    # Isolate random IG
    random_index = np.random.randint(0, 19)
    X_isolated = isolateICComponent(X, fastICA, random_index)
    X_deleted = X - X_isolated
    isolated_prediction, deleted_prediction, _ = predict_on_isolated_components(X_isolated, X_deleted, model, device)

    all_predictions_random_deletion[i] = deleted_prediction.detach().cpu().numpy()
    all_predictions_random_insertion[i] = isolated_prediction.detach().cpu().numpy()

results = {
    'predictions' : all_predictions,
    'prediction_deletions' : all_predictions_deletion,
    'prediction_insertions' : all_predictions_insertion,
    'prediction_random_deletions' : all_predictions_random_deletion,
    'prediction_random_insertions' : all_predictions_random_insertion
}

with open('./results/ica_ig_insertion_deletion_results.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)