from ase.build import molecule
from ase.io import read, write
from ase.calculators.dftb import Dftb

# Create position of C60 without optimization
atoms = molecule('C60')
calc = Dftb(label = 'c60',
	    Hamiltonian_SlaterKosterFiles_Prefix = "/home/wonk/src/par/trans3d-0-1/",
	    Hamiltonian_MaxAngularMomentum_='',
	    Hamiltonian_MaxAngularMomentum_C='p'
	    )
atoms.set_calculator(calc)

write('sysC60.xyz', atoms)
