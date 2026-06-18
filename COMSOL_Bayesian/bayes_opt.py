import os
from typing import Optional
import torch
from botorch.test_functions.synthetic import Ackley
from botorch.models.transforms.input import Normalize
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf
import time
import warnings
from botorch import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler
from trainset import trainX, trainY, train_set

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")

# constrains
def constraint_ftn(X):
    return X[:,1]+X[:,3]-15 # if the zero point is 3mm to 12mm, moving the boundary btw magnets

# evaluate constrains
def weighted_obj(X, exact_obj_):
    """feasible 1; not feasible 0"""
    value = exact_obj_.unsqueeze(-1) if exact_obj_.ndim==1 else exact_obj_
    return value*(constraint_ftn(X) <= 0).type_as(X)

# constrained objective wrapper
def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
    return Z[..., 0]
def constraint_callable(Z):
    return Z[..., 1]

# build model
def generate_model(train_x, train_obj, train_con, train_yvar, state_dict = None):
    model_obj = SingleTaskGP(
        train_x,
        train_obj,
        train_yvar.expand_as(train_obj),
        input_transform=Normalize(d=train_x.shape[-1]),
        ).to(train_x)
    model_con = SingleTaskGP(
        train_x,
        train_con,
        train_yvar.expand_as(train_con),
    input_transform = Normalize(d=train_x.shape[-1],)
        ).to(train_x)
    model = ModelListGP(model_obj, model_con)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model

# acquisit function
def optimize_acqf_and_get_observation(acq_func, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, NOISE_SE):
    """Optimizes the acquistion function, and returns a new candidate and a noisy observation"""
    candidates, _ = optimize_acqf(
        acq_function = acq_func,
        bounds=torch.tensor([[0.1, 3.0, 0.1, 3.0, -3.0], [15.0, 9.0, 15.0, 9.0, 5.0]]),#check lower and upper bound!
        q = BATCH_SIZE,
        num_restarts = NUM_RESTARTS,
        raw_samples = RAW_SAMPLES,
        options = {"batch_limit": 5, "maxiter": 200},
        )
        
    new_x = candidates.detach()
    exact_con = constraint_ftn(new_x).unsqueeze(-1)
    new_con = exact_con + NOISE_SE * torch.randn_like(exact_con)
    return new_x, new_con

def BayesianOptimization(modelFolder):
    BATCH_SIZE = 1
    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
    RAW_SAMPLES = 512 if not SMOKE_TEST else 32
    MC_SAMPLES = 256 if not SMOKE_TEST else 32

    # generate noise    
    NOISE_SE = 0.25
    train_yvar = torch.tensor(NOISE_SE**2, device=device, dtype=dtype)

    # generate data
    train_x, exact_obj_ = train_set(modelFolder)
    exact_obj = exact_obj_.unsqueeze(-1)
    train_obj = exact_obj + NOISE_SE*torch.randn_like(exact_obj)
    exact_con = constraint_ftn(train_x).unsqueeze(-1)
    train_con = exact_con + NOISE_SE*torch.randn_like(exact_con)
    best_observed_value = weighted_obj(train_x, exact_obj).max().item()


    # call the objective
    objective = GenericMCObjective(objective= obj_callable)

    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    verbose = False

    mll_nei, model_nei = generate_model(train_x, train_obj, train_con, train_yvar)

    # best_observed_nei.append(best_observed_value_nei)

    fit_gpytorch_mll(mll_nei)

    qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

    qLogNEI = qLogNoisyExpectedImprovement(
        model=model_nei,
        X_baseline=train_x,
        sampler = qmc_sampler,
        objective=objective,
        constraints=[constraint_callable],
        )
        
    new_x_nei, new_con_nei = optimize_acqf_and_get_observation(qLogNEI, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, NOISE_SE)

    train_x = torch.cat([train_x, new_x_nei])
   
    train_con = torch.cat([train_con, new_con_nei])

    # best_value_nei = weighted_obj(train_x_nei).max().item()

    # best_observed_nei.append(best_value_nei)

    # reinitialize the models for next iteration with the current state dict to speedup fitting
    print('M1_H', 'M1_W', 'M2_H', 'M2_W', 'offset', sep=",   ")
    print(' '.join(map(str, new_x_nei.flatten().tolist())))
   
  

    # if verbose:
    #     print(
    #         f"({new_x_nei:>4.2f}),",
    #         end="",
    #         )
    # else:
    #     print(".", end="")
    

