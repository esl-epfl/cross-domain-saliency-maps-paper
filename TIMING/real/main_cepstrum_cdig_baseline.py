import sys
from os import path
from pathlib import Path
print(path.dirname( path.dirname( path.abspath(__file__) ) ))
sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))


import multiprocessing as mp
import numpy as np
import random
import torch as th
import torch.nn as nn
import os
from utils.tools import print_results

from attribution.gate_mask import GateMask
from attribution.gatemasknn import *
from argparse import ArgumentParser
from tqdm import tqdm
from captum.attr import DeepLift, GradientShap, IntegratedGradients, Lime, KernelShap, DeepLiftShap
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from typing import List

from tqdm import tqdm 

from tint.attr import (
    DynaMask,
    ExtremalMask,
    Fit,
    Retain,
    TemporalAugmentedOcclusion,
    TemporalOcclusion,
    Occlusion,
    FeatureAblation,
    TimeForwardTunnel,
)
from tint.attr.models import (
    ExtremalMaskNet,
    JointFeatureGeneratorNet,
    MaskNet,
    RetainNet,
)
from datasets.mimic3 import Mimic3
from datasets.PAM import PAM
from datasets.boiler import Boiler
from datasets.epilepsy import Epilepsy
from datasets.wafer import Wafer
from datasets.freezer import Freezer
from tint.metrics import (
    accuracy,
    comprehensiveness,
    cross_entropy,
    log_odds,
    sufficiency,
)

from real.cumulative_difference import cumulative_difference
from tint.models import MLP, RNN

from cross_domain_saliency_maps.torch_ig.domain_transforms import FourierDomain
from cross_domain_saliency_maps.torch_ig.domain_transforms import DomainBase

from real.classifier import MimicClassifierNet

class CrossDomainIG:
    """ Cross Domain IG base class. Defines the basic functionality 
        for cross-domain ig. 

        Attributes:
        model (torch.nn.Module): A pytorch model.
        n_iterations (int): Number of iterations for approximating IG.
        output_channel(int): The output channel of the model for which 
                             we generate the saliency map.
        dtype (dtype): The type of the target domain.
    """
    def __init__(self, model: th.nn.Module, 
                 n_iterations: int, 
                 output_channel: int, 
                 dtype = th.float32):
        """ Initializes CrossDomainIG.

        Args:
            model (torch.nn.Module): A pytorch model for which 
                saliency maps will be generated.
            n_iterations (int): Number of steps used in approximating
                the integral in the Integrated Gradients computation.
            output_channel (int): The channle of the model's output used
                for the saliency map (e.g. the class channel).
        """
        self.model = model
        self.n_iterations = n_iterations
        self.output_channel = output_channel

    def initialize_domain(self, Domain: DomainBase, **kwargs):
        """ Initializes the target domain in which the IG is 
            expressed.

            Args:
                Domain (DomainBase): The target domain.
                **kwargs: Parameters needed to initialize the Domain.
        """
        self.domain = Domain(**kwargs)
    
    def run(self, x: np.array, x_baseline: np.array):
        """ Runs the saliency map generation for input sample x
            using the x_baseline for baseline.

            Args:
                x (np.array): a single sample with shape [1, n_timesteps, n_channels]
                x_baseline (np.array): the baseline sample with shape [1, n_timesteps, n_channels]
        """
        self.domain.set_coefficients(x, x_baseline)

        grad_sum = 0

        X_in = self.domain.get_coefficients()
        X_baseline = self.domain.get_coefficient_baseline()

        X_samples = [ X_baseline + (float(i) / self.n_iterations) * (X_in - X_baseline) for i in range(1, self.n_iterations + 1)]

        for X_sample in tqdm(X_samples):
            X_sample.requires_grad = True
            x_ = self.domain.inverse_transform(X_sample)
            prediction = self.model(x_)
            prediction[0, self.output_channel].backward()
            grad_sum += th.conj(X_sample.grad)

        grad_sum /= self.n_iterations
        self.multiIG = th.real((X_in - X_baseline) * grad_sum)
        
        return self.multiIG
    
    def run_batched(self, x: np.array, x_baseline: np.array, output_channels: np.array):
        """ Runs batched saliency map generation for multiple samples.

            Args:
                x (np.array): batch of samples with shape [batch_size, n_timesteps, n_channels]
                x_baseline (np.array): batch of baselines with shape [batch_size, n_timesteps, n_channels]
                output_channels (np.array): array of output channels for each sample [batch_size]
        """
        batch_size = x.shape[0]
        self.domain.set_coefficients(x, x_baseline)

        X_in = self.domain.get_coefficients()
        X_baseline = self.domain.get_coefficient_baseline()

        # Create interpolation samples for all batch items at once
        X_samples = [X_baseline + (float(i) / self.n_iterations) * (X_in - X_baseline) 
                     for i in range(1, self.n_iterations + 1)]

        grad_sum = th.zeros_like(X_in)

        for X_sample in X_samples:
            X_sample.requires_grad = True
            x_ = self.domain.inverse_transform(X_sample)
            prediction = self.model(x_)
            
            # Compute gradients for each sample's respective output channel
            # Create a target tensor with ones at the specified channels
            target = th.zeros_like(prediction)
            target[range(batch_size), output_channels] = 1.0
            
            # Backward pass
            prediction.backward(gradient=target)
            grad_sum += th.conj(X_sample.grad)
            
            # Zero gradients for next iteration
            X_sample.grad = None

        grad_sum /= self.n_iterations
        self.multiIG = th.real((X_in - X_baseline) * grad_sum)
        
        return self.multiIG

    def getMultiIG(self):
        """ Get the generated saliency map.
        """
        return self.multiIG
    

class ComplexCepstrumDomain(DomainBase):
    """ Domain implementation for the Fourier transform, mapping 
    time-domain samples into the frequency domain.
    """
    def __init__(self, device, dtype = th.float32, time_dimension = -1, eps = 1e-7):
        """ Initialize the Fourier Domain.

        Args:
            dtype: The type of the input features. 
            time_dimension: The dimension in the input which 
                corresponds to the time-points.
        """
        super().__init__(device = device, dtype = dtype)
        self.time_dimension = time_dimension
        self.eps = eps
    
    def forward_transform(self, x: th.Tensor):
        """ Implementation of the forward transform, transforming the input
        time sample to the corresponding frequency domain sample. 

        Args:
            x (tf.Tensor): Input time-domain sample.
        """
        X = th.fft.fft(x, dim=self.time_dimension)          # complex
        Y = th.log(X + self.eps)                            # complex log
        C = th.fft.ifft(Y, dim=self.time_dimension)         # complex cepstrum
        return C
      
    def set_coefficients(self, x: np.array, x_baseline: np.array):
        """ Sets the frequency coefficients transforming the input and
        baseline samples into the frequency domain.

        Args:
            x (np.array): The input sample in time-domain.
            x_baseline (np.array): The baseline sample in time-domain.
        """
        x_tf = th.from_numpy(x).type(self.dtype).to(self.device)
        x_baseline_tf = th.from_numpy(x_baseline).type(self.dtype).to(self.device)

        self.coefficients = self.forward_transform(x_tf)
        self.baseline_coefficients = self.forward_transform(x_baseline_tf)
    
    def inverse_transform(self, x_input: th.Tensor):
        """ Inverse transform, transforming the frequency domain input
        x_input points back into the time domain.

        Args:
            x_input (tf.Tensor): The frequency domain input.
        """
        Y = th.fft.fft(x_input, dim=self.time_dimension)    # complex
        X = th.exp(Y)                                       # complex
        x_rec = th.fft.ifft(X, dim=self.time_dimension)
        return x_rec.type(th.float32)

class ComplexCepstrumIG(CrossDomainIG):
    """ Implementation of the CrossDomainIG specifically for the
    frequency target domain. 
    """
    def __init__(self, 
                 model: th.nn.Module, 
                 n_iterations: int, 
                 output_channel: int, 
                 device: th.device,
                 dtype = th.float32):
        super().__init__(model, n_iterations,
                         output_channel, dtype)
        
        self.initialize_domain(ComplexCepstrumDomain, device = device, dtype = dtype, time_dimension = 1)

class TransformModelWrapper(nn.Module):
    def __init__(self, time_model, inverse_transform, time_dim = 1):
        super().__init__()
        self.time_model = time_model
        self.inverse_transform = inverse_transform
        self.time_dim = time_dim

    def forward(self, z, *args, **kwargs):
        x_time = self.inverse_transform(z).type(th.float32)
        y = self.time_model(x_time, *args, **kwargs)
        return y
    
def Sepstrum(x):
    eps = 1e-7
    X = th.fft.fft(x, dim=1)
    Y = th.log(X + eps)
    C = th.fft.ifft(Y, dim=1)
    return C

def InverseSepstrum(z):
    Y = th.fft.fft(z, dim=1)
    X = th.exp(Y)
    x_rec = th.fft.ifft(X, dim=1)
    return x_rec


def CDIG_batched(model, sample, device, baseline_sample):
    data = sample.to(device)
    output = model(data.float())
    output_channels = np.argmax(output.detach().cpu().numpy(), axis=-1)
    
    # Create single IG instance (output_channel is unused in batched mode)
    ig = ComplexCepstrumIG(
        model=model, 
        n_iterations=20, 
        device=device, 
        output_channel=0  # Not used in batched version
    )
    
    # Prepare batched inputs
    ig_data = data.detach().cpu().numpy()
    # baseline = np.zeros_like(ig_data)
    baseline = baseline_sample.expand(data.shape[0], *baseline_sample.shape[1:])
    baseline = baseline.detach().cpu().numpy()
    
    # Run batched attribution - all samples at once!
    fourier_saliency_map = ig.run_batched(ig_data, baseline, output_channels)
    
    return fourier_saliency_map.cpu()

def main(
    explainers: List[str],
    data: str,
    areas: list,
    device: str = "cpu",
    fold: int = 0,
    seed: int = 42,
    is_train: bool = True,
    deterministic: bool = False,
    lambda_1: float = 1.0,
    lambda_2: float = 1.0,
    lambda_3: float = 1.0,
    num_segments: int = 50,
    min_seg_len: int = 1,
    max_seg_len: int = 48,
    mask_lr: float = 0.1,
    output_file: str = "results.csv",
    model_type: str = "state",
    testbs: int = 0,
    top: int = 50,
    skip_train_timex: bool = True,
    prob: float = 0.1 ,
):
    # Get accelerator and device
    accelerator = device.split(":")[0]
    device_id = 1
    if len(device.split(":")) > 1:
        device_id = [int(device.split(":")[1])]

    # Create lock
    lock = mp.Lock()

    # Load data
    if data == "mimic3":
        datamodule = Mimic3(n_folds=5, fold=fold, seed=seed)
        
        classifier = MimicClassifierNet(
            feature_size=32,
            # feature_size=31,
            n_state=2,
            n_timesteps=48,
            hidden_size=200,
            regres=True,
            loss="cross_entropy",
            lr=0.0001,
            l2=1e-3,
            model_type=model_type
        )
        num_features = 32
        num_classes = 2
        max_len = 48
        
    elif data == "PAM":
        datamodule = PAM(fold=fold, seed=seed)
        
        classifier = MimicClassifierNet(
            feature_size=17,
            n_state=8,
            n_timesteps=600,
            hidden_size=200,
            regres=True,
            loss="cross_entropy",
            lr=0.0001,
            l2=1e-3,
            model_type=model_type
        )
        num_features = 17
        num_classes = 8
        max_len = 600
        
    elif data == "boiler":
        datamodule = Boiler(fold=fold, seed=seed)
        
        classifier = MimicClassifierNet(
            feature_size=20,
            n_state=2,
            n_timesteps=36,
            hidden_size=200,
            regres=True,
            loss="cross_entropy",
            lr=0.0001,
            l2=1e-3,
            model_type=model_type
        )
        num_features = 20
        num_classes = 2
        max_len = 36
    
    elif data == "epilepsy":
        datamodule = Epilepsy(fold=fold, seed=seed)
        
        classifier = MimicClassifierNet(
            feature_size=1,
            n_state=2,
            n_timesteps=178,
            hidden_size=200,
            regres=True,
            loss="cross_entropy",
            lr=0.0001,
            l2=1e-3,
            model_type=model_type
        )
        num_features = 1
        num_classes = 2
        max_len = 178
    
    elif data == "freezer":
        datamodule = Freezer(n_folds=5, fold=fold, seed=seed)
        
        classifier = MimicClassifierNet(
            feature_size=1,
            n_state=2,
            n_timesteps=301,
            hidden_size=200,
            regres=True,
            loss="cross_entropy",
            lr=0.0001,
            l2=1e-3,
            model_type=model_type
        )
        
        num_features = 1
        num_classes = 2
        max_len = 301
    
    elif data == "wafer":
        datamodule = Wafer(n_folds=5, fold=fold, seed=seed)
        
        classifier = MimicClassifierNet(
            feature_size=1,
            n_state=2,
            n_timesteps=152,
            hidden_size=200,
            regres=True,
            loss="cross_entropy",
            lr=0.0001,
            l2=1e-3,
            model_type=model_type
        )
        num_features = 1
        num_classes = 2
        max_len = 152

    classifier.load_state_dict(th.load("./model/{}/{}_classifier_{}_{}_no_imputation".format(data, model_type, fold, seed)))
    transformed_model = TransformModelWrapper(time_model = classifier, inverse_transform = InverseSepstrum)


    # Get data for explainers
    with lock:
        x_train = datamodule.preprocess(split="train")["x"].to(device)
        x_test = datamodule.preprocess(split="test")["x"].to(device)
        y_train = datamodule.preprocess(split="train")["y"].to(device)
        y_test = datamodule.preprocess(split="test")["y"].to(device)
        mask_train = datamodule.preprocess(split="train")["mask"].to(device)
        mask_test = datamodule.preprocess(split="test")["mask"].to(device)

        # Mean sample over the test set
        test_mean_baseline = x_test.mean(dim=0, keepdim=True)   # shape: (1, ...)

        x_test_frequency = Sepstrum(x_test)
        test_mean_baseline_frequency = Sepstrum(test_mean_baseline)

    print("Data loaded.")
    # Switch to eval
    classifier.eval()

    # Set model to device
    classifier.to(device)

    # Disable cudnn if using cuda accelerator.
    # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
    # for more information.
    if accelerator == "cuda":
        th.backends.cudnn.enabled = False

    # Create dict of attributions
    attr = dict()
    
    from torch.utils.data import DataLoader, TensorDataset
    test_dataset = TensorDataset(x_test, mask_test)
    test_loader = DataLoader(test_dataset, batch_size=testbs, shuffle=False)

    frequency_test_dataset = TensorDataset(x_test_frequency, mask_test)
    frequency_test_loader = DataLoader(frequency_test_dataset, batch_size=testbs, shuffle=False)
    
    if model_type == "state":
        temporal_additional_forward_args = (False, False, False)
    else:
        temporal_additional_forward_args = (False, False, False)
    
    data_mask=mask_test
    data_len, t_len, _ = x_test.shape
        
    timesteps=(
        th.linspace(0, 1, t_len, device=x_test.device)
        .unsqueeze(0)
        .repeat(data_len, 1)
    )

    print("Running sepstrum_cdig")
    if "sepstrum_cdig" in explainers:
        attributions = []
        for sample, target in tqdm(test_loader):
            attributions.append(CDIG_batched(classifier, sample, device, test_mean_baseline_frequency))

        attr["sepstrum_cdig"] = th.cat(attributions)

    # Compute x_avg for the baseline
    x_avg = x_test.mean(1, keepdim=True).repeat(1, x_test.shape[1], 1)
    x_avg = Sepstrum(x_avg)
    
    # print

    # Dict for baselines
    baselines_dict = {0: "Average", 1: "Zeros"}
    
    # ## data_mask=mask_test.to("cpu")
    # data_mask = mask_test.to(x_test.device)
    # data_len, t_len, _ = x_test.shape
        
    timesteps=(
        th.linspace(0, 1, t_len, device=x_test.device)
        .unsqueeze(0)
        .repeat(data_len, 1)
    )

    with open(output_file, "a") as fp, lock:
        for i, baselines in enumerate([x_avg, 0.0]):
            for topk in areas:
                for k, v in attr.items():
                    print(x_test_frequency.shape)
                    print(v.shape)
                    cum_diff, AUCC, cum_50_diff, _ = cumulative_difference(
                        transformed_model,
                        x_test_frequency,
                        attributions=v,
                        baselines=baselines,
                        topk=topk,
                        top=args.top,
                        testbs=testbs,
                        additional_forward_args=(mask_test, None, False),
                    )
                    
                    
                    
                    total_acc = 0.0
                    total_comp = 0.0
                    total_ce = 0.0
                    total_lodds = 0.0
                    total_suff = 0.0
                    total_samples = 0

                    # 2. Loop over batches
                    for batch_idx, batch in enumerate(frequency_test_loader):
                        # batch = (input_tensor, data_mask, ...)
                        x_batch = batch[0].to(device)
                        data_mask_batch = batch[1].to(device)
                        batch_size = x_batch.shape[0]

                        # If timesteps is sized for the entire dataset, slice for this batch
                        # Example (adjust accordingly if needed):
                        timesteps_batch = timesteps[batch_idx * batch_size : batch_idx * batch_size + batch_size]

                        # Prepare baselines for the batch
                        # If baselines is a tensor like x_avg, slice it for the batch dimension
                        if isinstance(baselines, th.Tensor):
                            baselines_batch = baselines[batch_idx * batch_size : batch_idx * batch_size + batch_size]
                            baselines_batch = baselines_batch.to(device)
                        else:
                            # e.g., if baselines=0.0 or a scalar, you might just keep it as-is
                            # Or replicate it: baselines_batch = torch.zeros_like(x_batch)
                            baselines_batch = baselines

                        # Similarly slice the attribution tensor 'v'
                        v_batch = v[batch_idx * batch_size : batch_idx * batch_size + batch_size].to(device)

                        # 3. Compute metrics for this batch
                        acc = accuracy(
                            transformed_model,
                            x_batch,
                            attributions=v_batch,
                            baselines=baselines_batch,
                            topk=topk,
                            additional_forward_args=(data_mask_batch, timesteps_batch, False)
                        )
                        comp = comprehensiveness(
                            transformed_model,
                            x_batch,
                            attributions=v_batch,
                            baselines=baselines_batch,
                            topk=topk,
                            additional_forward_args=(data_mask_batch, timesteps_batch, False)
                        )
                        ce = cross_entropy(
                            transformed_model,
                            x_batch,
                            attributions=v_batch,
                            baselines=baselines_batch,
                            topk=topk,
                            additional_forward_args=(data_mask_batch, timesteps_batch, False)
                        )
                        l_odds = log_odds(
                            transformed_model,
                            x_batch,
                            attributions=v_batch,
                            baselines=baselines_batch,
                            topk=topk,
                            additional_forward_args=(data_mask_batch, timesteps_batch, False)
                        )
                        suff = sufficiency(
                            transformed_model,
                            x_batch,
                            attributions=v_batch,
                            baselines=baselines_batch,
                            topk=topk,
                            additional_forward_args=(data_mask_batch, timesteps_batch, False)
                        )

                        # 4. Accumulate results (multiply by batch_size if metrics are averages)
                        #    If your metric function already returns a sum, you may not need to multiply.
                        total_acc += acc * batch_size
                        total_comp += comp * batch_size
                        total_ce += ce * batch_size
                        total_lodds += l_odds * batch_size
                        total_suff += suff * batch_size
                        total_samples += batch_size
                        
                    mean_acc = total_acc / total_samples
                    mean_comp = total_comp / total_samples
                    mean_ce = total_ce / total_samples
                    mean_lodds = total_lodds / total_samples
                    mean_suff = total_suff / total_samples

                    fp.write(str(seed) + ",")
                    fp.write(str(fold) + ",")
                    fp.write(baselines_dict[i] + ",")
                    fp.write(str(topk) + ",")
                    fp.write(k + ",")
                    fp.write(str(lambda_1) + ",")
                    fp.write(str(lambda_2) + ",")
                    fp.write(str(lambda_3) + ",")
                    fp.write(f"{cum_50_diff:.4},")
                    fp.write(f"{cum_diff:.4},")
                    fp.write(f"{AUCC:.4},")
                    fp.write(f"{mean_acc:.4},")
                    fp.write(f"{mean_comp:.4},")
                    fp.write(f"{mean_ce:.4},")
                    fp.write(f"{mean_lodds:.4},")
                    fp.write(f"{mean_suff:.4}")
                    fp.write("\n")

                    print(str(seed) + ",")
                    print(str(fold) + ",")
                    print(baselines_dict[i] + ",")
                    print(str(topk) + ",")
                    print(k + ",")
                    print(str(lambda_1) + ",")
                    print(str(lambda_2) + ",")
                    print(str(lambda_3) + ",")
                    print(f"{cum_50_diff:.4},")
                    print(f"{cum_diff:.4},")
            

    if not os.path.exists("./results_our/"):
        os.makedirs("./results_our/")
    for key in attr.keys():
        result = attr[key]
        if isinstance(result, tuple): result = result[0]
        np.save('./results_our/{}_{}_{}_result_{}_{}.npy'.format(data, model_type, key, fold, seed), result.detach().cpu().numpy())
    
    print(f"{explainers} done")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--explainers",
        type=str,
        default=[
            "gate_mask"
        ],
        nargs="+",
        metavar="N",
        help="List of explainer to use.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="mimic3",
        help="real world data",
    )
    parser.add_argument(
        "--areas",
        type=float,
        default=[
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        ],
        nargs="+",
        metavar="N",
        help="List of areas to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Which device to use.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Fold of the cross-validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data generation.",
    )
    parser.add_argument(
        "--train",
        type=bool,
        default=False,
        help="Train thr rnn classifier.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Whether to make training deterministic or not.",
    )
    parser.add_argument(
        "--lambda-1",
        type=float,
        default=0.001,   # 0.01
        help="Lambda 1 hyperparameter.",
    )
    parser.add_argument(
        "--lambda-2",
        type=float,
        default=0.01,    #0.01
        help="Lambda 2 hyperparameter.",
    )
    parser.add_argument(
        "--lambda-3",
        type=float,
        default=0.01,    #0.01
        help="Lambda 2 hyperparameter.",
    )
    parser.add_argument(
        "--mask_lr",
        type=float,
        default=0.01,   
        help="learning rate for mask based method",
    )
    parser.add_argument(
        "--prob",
        type=float,
        default=0.1,   
        help="asff",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results_gate.csv",
        help="Where to save the results.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="state",
        choices=["state", "mtand", "seft", "transformer", "cnn"],
    )
    parser.add_argument(
        "--testbs",
        type=int,
        default=200
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50
    )
    parser.add_argument(
        "--num_segments",
        type=int,
        default=50
    )
    parser.add_argument(
        "--min_seg_len",
        type=int,
        default=1
    )
    parser.add_argument(
        "--max_seg_len",
        type=int,
        default=48
    )
    parser.add_argument(
        "--skip_train_timex",
        action='store_true'
    )
    return parser.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    print(f"set seed as {seed}")

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(
        explainers=args.explainers,
        data=args.data,
        areas=args.areas,
        device=args.device,
        fold=args.fold,
        seed=args.seed,
        is_train=args.train,
        deterministic=args.deterministic,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        lambda_3=args.lambda_3,
        num_segments=args.num_segments,
        min_seg_len=args.min_seg_len,
        max_seg_len=args.max_seg_len,
        mask_lr=args.mask_lr,
        output_file=args.output_file,
        model_type=args.model_type,
        testbs=args.testbs,
        top=args.top,
        skip_train_timex=args.skip_train_timex,
        prob=args.prob
    )