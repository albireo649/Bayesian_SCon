#!/usr/bin/python
from ase import Atoms
from p6_0 import P6 as p60
from p6_1 import P6 as p61
from ase.optimize import BFGS
#from ase.visualize import view
from ase.io import *
import numpy as np
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
					 Stationary, ZeroRotation)
from ase.md.npt import NPT
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase import units
from ase.io import read, write
import time
import sys,os

T = 300  # Kelvin

calc0 = p60()
calc1 = p61()



atoms = read("sys.xyz") 

atoms.set_calculator(calc0)

atoms.set_pbc([True,True,True])
atoms.set_cell([[25,0,0],[0,25,0],[0,0,25]])
atoms.center()

#print atoms.get_potential_energies()

dyn = BFGS(atoms, trajectory='opt.traj')
dyn.run(fmax=10.)

#atoms[0:1].set_masses([162.50,162.50])
#atoms[0].mass=162.50
#atoms[1].mass=162.50

atoms.set_calculator(calc1) 

start_time = time.time()

#atoms.set_pbc([True,True,True])
#atoms.set_cell([[10,0,0],[0,10,0],[0,0,10]])
#atoms.center()
# Set the momenta corresponding to T=300K
MaxwellBoltzmannDistribution(atoms, 300 * units.kB)
Stationary(atoms)  # zero linear momentum
ZeroRotation(atoms)  # zero angular momentum


#dyn = Langevin(atoms, 1 * units.fs, T * units.kB, 0.002)

dyn = NPT(atoms, timestep=1.*units.fs, temperature= units.kB * 300,
          ttime=10.*units.fs,pfactor=None,externalstress=1)


def printenergy(a=atoms):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    d = a.get_dipole_moment()
    os.system("cat p6.out >> trj_all_out.out")
    print("d:", d)
    print('Energy per atom: Epot = %.4feV  Ekin = %.4feV (T=%4.0fK)  '
          'Etot = %.4feV, time lapse = %.4f' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin, time.time()-start_time))
    sys.stdout.flush()
    
dyn.attach(printenergy, interval=1)

traj = Trajectory('md.traj', 'w', atoms)
dyn.attach(traj.write, interval=10)

# Now run the dynamics
printenergy()
dyn.run(500000)
