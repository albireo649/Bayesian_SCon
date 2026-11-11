import os

import numpy as np

from warnings import warn
from ase.atoms import Atoms
from ase.units import Hartree, Bohr
from ase.calculators.calculator import FileIOCalculator, Parameters, ReadError


class P6(FileIOCalculator):

    implemented_properties = ['energy', 'forces','dipole']
    command = './mpiexec -np 10 ./priroda p6.in > p6.out'


    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='p6', atoms=None, **kwargs):
        """Construct NWchem-calculator object."""
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)

    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()


    default_parameters = dict(
        #skin=1.0,
        #potential=None,
        header=[b'XXX\n',
                b'XXX\n',
                b'XXX\n'])
    def write_input(self, atoms, properties=None, system_changes=None):
         a=open("p6.in","w")
         a.write("""$system
memory=2000
disk=80
path=./ 
$end
$control
task=gradient
basis=./mixed.bas
print=charges
save=vec.data
$end
$guess
read=1
$end
$scf
convergence=1e-5
procedure=bfgs
iterations=2000
$end
$grid
accuracy=1e-7
$end
$molecule
charge=0
mult=1
cartesian
set=3z
""")
         b=self.atoms.get_positions()
         c=self.atoms.get_atomic_numbers()
         for i in range (b.shape[0]):
            a.write("%i   " % c[i])
            for j in range (b.shape[1]):
               a.write("% 3.8f   " % b[i,j])
            a.write("\n")
         a.write("$end")
         a.close()

    def read(self, label):
        FileIOCalculator.read(self, label)
        if not os.path.isfile(self.label + '.out'):
            raise ReadError
        self.read_results()

    def read_results(self):
        self.read_energy()
        self.read_forces()
        self.read_dipole()

         
    def read_energy(self):
        """Read Energy from nwchem output file."""
        text = open(self.label + '.out', 'r').read()
        lines = iter(text.split('\n'))

        # Energy:
        estring = 'eng> E='
        for line in lines:
            if line.find(estring) >= 0:
                energy = float(line.split("=")[-1])
                #print "here", energy
        self.results['energy'] = energy * Hartree

    def read_dipole(self):
        """Read Energy from nwchem output file."""
        text = open(self.label + '.out', 'r').read()
        lines = iter(text.split('\n'))

        # Energy:
        estring = 'eng> D='
        for line in lines:
            if line.find(estring) >= 0:
                d = line.split("D=")[-1]
                d = np.float32(d.split())
                #print "here", energy
        self.results['dipole'] = np.array(d)*1.

    def read_forces(self):
        text = open(self.label + '.out', 'r').read()
        lines = iter(text.split('\n'))
        fstrin1 = 'eng> G='
        fstrin2 = 'eng>   '
        gradients=[]
        for line in lines:
            if line.find(fstrin1) >= 0:
                b=line.split("=")[-1]
                b= np.float32(b.split())
                gradients.append(list(b))
            if line.find(fstrin2) >= 0:
                b=line.split(">   ")[-1]
                b= np.float32(b.split())
                gradients.append(list(b))

        self.results['forces'] = -np.array(gradients) * Hartree / Bohr

 
