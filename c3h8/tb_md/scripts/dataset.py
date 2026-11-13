from ase.io import read
from ase.calculators.dftb import Dftb
import numpy as np


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

