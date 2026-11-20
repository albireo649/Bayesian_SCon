import os
from random import random
import re
import torch
import random
from ase.io import read
from ase.calculators.dftb import Dftb
import numpy as np
from scipy.stats import qmc


# get energy and force RMSE for a given set of parameters
def test_md(prefix='../corrupt_pp/'):
	frames = read('../opt.traj', index=':')

	ref_energy = []
	ref_forces = []
	
	for atoms in frames:
		ref_energy.append(atoms.get_potential_energy())
		ref_forces.append(atoms.get_forces())

	pred_energy = []
	pred_forces = []

	for i, atoms in enumerate(frames):
		calc = Dftb(label='sys',
	    	Hamiltonian_SlaterKosterFiles_Prefix=prefix,
	    	Hamiltonian_MaxAngularMomentum_='',
	    	Hamiltonian_MaxAngularMomentum_C='p',
	    	Hamiltonian_MaxAngularMomentum_H='s',
	    	kpts=(1,1,1)
	    	)
		atoms.calc = calc
		pred_energy.append(atoms.get_potential_energy())
		pred_forces.append(atoms.get_forces())

	ref_energy = np.array(ref_energy)
	pred_energy = np.array(pred_energy)
	ref_relative = ref_energy - ref_energy[0]
	pred_relative = pred_energy - pred_energy[0]

	ref_forces = np.array(ref_forces)
	pred_forces = np.array(pred_forces)

	force_rmse = np.sqrt(np.mean((ref_forces - pred_forces) ** 2))
	energy_rmse = np.sqrt(np.mean((ref_energy - pred_energy) ** 2))

	print(f"Force RMSE: {force_rmse:.6f} eV/Å")
	print(f"Energy RMSE: {energy_rmse:.6f} eV")
	return energy_rmse, force_rmse



def modify_skf(Ep, Es, SPE, skf_file, output_file):

    with open(skf_file, "r") as file:
            lines = file.readlines()
    line = lines[1]
    parameters = list(re.finditer( r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", line ))
    replacements= [f"{Ep:.10f}", f"{Es:.10f}", f"{SPE:.10f}"]

    for parameter, replacement in reversed(list(zip(parameters[1:4], replacements))): 
          start, end = parameter.span()
          line = line[:start] + replacement + line[end:]
    lines[1] = line
    print("New Random ", Ep, ", ", Es, ", ",SPE)
    with open(output_file, "w") as f:
			  f.writelines(lines)    


# generate training set with random parameters and evaluate the RMSE for each set of parameters for initial training
N_SAMPLES=10
LOWER = -0.6
UPPER = 0.0
# temporary bounds for the parameters, can be changed later
bounds = torch.tensor([[LOWER,LOWER,LOWER], [UPPER,UPPER,UPPER]], dtype=torch.float64)
def normalize(x):
    return (x - bounds[0]) / (bounds[1] - bounds[0])
def denormalize(x):
    return x * (bounds[1] - bounds[0]) + bounds[0]

scale = torch.tensor([0.5, 0.5], dtype=torch.float64)

def calculate_loss(outputs):
    # Calculate the loss
    loss = torch.sum((outputs*scale), dim=1, keepdim=True)
    return loss

Skf_File = os.path.join("../corrupt_pp", "C-C.skf")
def trainset(n_samples=N_SAMPLES, lower=LOWER, upper=UPPER, skf_file=Skf_File):
    sampler = qmc.LatinHypercube(d=3)
    l_bounds = [LOWER, LOWER, LOWER]
    u_bounds = [UPPER, UPPER, UPPER]
    input = []
    output = []
    for i in range(n_samples):
        sample = sampler.random(n=1)
        qmc.discrepancy(sample)
        sample = qmc.scale(sample, l_bounds, u_bounds)
        Ep=sample[0, 0]
        Es=sample[0, 1]
        SPE=sample[0, 2]
        modify_skf(Ep, Es, SPE, skf_file, skf_file)
        energy, force = test_md()
        input_data = np.array([Ep, Es, SPE])
        input.append(input_data)
        output_data = np.array([energy, force])
        output.append(output_data)
    input_tensor = torch.tensor(input, dtype=torch.float64)
    output_tensor = torch.tensor(output, dtype=torch.float64)
    train_x = normalize(input_tensor)
    train_y = -calculate_loss(output_tensor)
    return train_x, train_y