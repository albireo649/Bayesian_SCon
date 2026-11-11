from ase.build import molecule
from ase.optimize import BFGS
import numpy as np
from ase.calculators.dftb import Dftb
from ase.io import read, write
import os

# Read the molecule
atoms = read('sys.xyz')

# box size
atoms.set_pbc([True, True, True])
atoms.set_cell([25,25,25])
atoms.center()

calc = Dftb(label='sys',
	   #Hamiltonian_SlaterKosterFiles_Prefix = "/home/wonk/src/par/trans3d-0-1/",
	   Hamiltonian_SlaterKosterFiles_Prefix = "./corrupt_pp/",
	   Hamiltonian_MaxAngularMomentum_='',
	   Hamiltonian_MaxAngularMomentum_C='p',
	   Hamiltonian_MaxAngularMomentum_H='s',
	   kpts=(1,1,1)
	   )

atoms.set_calculator(calc)

# Pre-optimize wigh BFGS optimization module
dyn = BFGS(atoms=atoms, trajectory='opt.traj'); dyn.run(fmax=0.01)

write('opt.cif', atoms)
