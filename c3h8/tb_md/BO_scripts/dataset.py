import os
from random import random
import re
import torch
import random
from ase.io import read
from ase.calculators.dftb import Dftb
import numpy as np


# get energy and force RMSE for a given set of parameters
def test_md(prefix='../corrupt_pp/'):
	frames = read('../../p6_md/md.traj', index=':')

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

Skf_File = os.path.join("../corrupt_pp", "C-C.skf")

def modify_skf(Ep, Es, SPE, skf_file=Skf_File, output_file=Skf_File):
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
bounds = torch.tensor([[LOWER, UPPER], [LOWER, UPPER], [LOWER, UPPER]], dtype=torch.float64)
def normalize(x):
    return (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
def denormalize(x):
    return x * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

scale = torch.tensor([1.0, 1.0], dtype=torch.float64)

def calculate_loss(outputs):
    # Calculate the loss
    loss = torch.sum((outputs/scale), dim=1, keepdim=True)
    return loss

def trainset(n_samples=N_SAMPLES, lower=LOWER, upper=UPPER):
	input = []
	output = []
	for i in range(n_samples):
		Ep = random.uniform(lower, upper)
		Es = random.uniform(lower, upper)
		SPE = random.uniform(lower, upper)
		modify_skf(Skf_File, Skf_File, Ep, Es, SPE)
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