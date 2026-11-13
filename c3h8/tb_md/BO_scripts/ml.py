import os
from unicodedata import normalize
import torch
import warnings
import numpy as np
from typing import Optional

from botorch import fit_gpytorch_mll
from botorch.optim import optimize_acqf

from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.acquisition.objective import GenericMCObjective
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.models.transforms.input import Normalize

from botorch.sampling.normal import SobolQMCNormalSampler
# from botorch.acquisition import qLogExpectedImprovement, qLogNoisyExpectedImprovement
import dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")

# build model
def generate_model(train_x, train_obj, state_dict = None):
    model_obj = SingleTaskGP(
        train_x,
        train_obj,
        input_transform=Normalize(d=train_x.shape[-1]),
        ).to(train_x)

    model = ModelListGP(model_obj)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    print("model is generated")
    return mll, model

def opimization_observation(acq_func, model, train_x, train_obj, num_restarts=10, raw_samples=512):
    # Optimize acquisition function
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
        new_x, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack([torch.zeros(train_x.shape[-1], dtype=dtype), torch.ones(train_x.shape[-1], dtype=dtype)]),
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    new_x = new_x.detach()
    new_x_denormalized = dataset.denormalize(new_x)
    print("New x: ", new_x_denormalized)
    Ep, Es, SPE = new_x_denormalized[0].tolist()
    dataset.modify_skf(Ep, Es, SPE, skf_file=dataset.Skf_File, output_file=dataset.Skf_File)
    energy, force = dataset.test_md()
    input_data = np.array([Ep, Es, SPE])    
    output_data = np.array([energy, force])
    input_tensor_new = torch.tensor(input_data.reshape(1,-1), dtype=torch.float64)
    output_tensor_new = torch.tensor(output_data.reshape(1,-1), dtype=torch.float64)
    train_x_new = torch.cat((train_x, dataset.normalize(input_tensor_new)), dim=0)
    train_obj_new = torch.cat((train_obj, -dataset.calculate_loss(output_tensor_new)), dim=0)
    # print("New train_x: ", input_tensor_new)
    # print("New train_obj: ", dataset.calculate_loss(output_tensor_new))
    return train_x_new, train_obj_new