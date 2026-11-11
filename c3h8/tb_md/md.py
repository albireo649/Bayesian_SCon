#!/usr/bin/python
from ase import Atoms
from ase.optimize import BFGS
from ase.io import read, write
from ase.calculators.dftb import Dftb
from ase.md.velocitydistribution import(MaxwellBoltzmannDistribution, Stationary, ZeroRotation)
from ase.md.npt import NPT
from ase.io.trajectory import Trajectory
from ase import units
import numpy as np
import time
import sys, os

# Read the molecule
atoms = read('opt.cif')

calc = Dftb(label='sys',
	    #Hamiltonian_SlaterKosterFiles_Prefix='/home/wonk/src/par/trans3d-0-1/',
	    Hamiltonian_SlaterKosterFiles_Prefix='./corrupt_pp/',
	    Hamiltonian_MaxAngularMomentum_='',
	    Hamiltonian_MaxAngularMomentum_C='p',
	    Hamiltonian_MaxAngularMomentum_H='s',
	    kpts=(1,1,1)
	    )
atoms.set_calculator(calc)

dyn=BFGS(atoms=atoms, trajectory='opt.traj')
dyn.run(fmax=0.01)

write('opt.cif',atoms)

# Temperature(Kelvin)
T=300

start_time = time.time()

# SEt the momenta corresponding to the temperature)
MaxwellBoltzmannDistribution(atoms=atoms, temperature_K=T*units.kB)
Stationary(atoms) # zero linear momentum
ZeroRotation(atoms) # zero angular momentum

dyn = NPT(atoms,  
	  timestep=1.*units.fs, 
	  temperature=units.kB*T, # Kelvin 
	  externalstress = 1, # bar or Gpa isotropic pressure
	  ttime=20.*units.fs, # Thermostat time factor, relaxation time 
	  pfactor=None # Barostat time fafctor, Bulk modulus
	  )

def printenergy(a=atoms):
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.4feV Ekin = %.4feV (T=%4.0fK)   '
	  'Etot = %.4feV, time lapse = %.4f' % (epot, ekin, ekin / (1.5*units.kB), epot+ekin, time.time()-start_time))
    sys.stdout.flush()

dyn.attach(printenergy, interval=1)

traj = Trajectory('md.traj', 'w', atoms)
dyn.attach(traj.write, interval=10)

printenergy()
dyn.run(500000)
