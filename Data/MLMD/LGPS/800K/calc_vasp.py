# +
from ase.calculators.vasp import Vasp

calc = Vasp(command="mpirun -np 128 vasp_std", xc='pbe', ncore=2,
            directory='vasp', lwave=False, lcharg=False, nelm=300,
            ispin=1, encut=500, lreal='Auto', algo='Fast',
            )

