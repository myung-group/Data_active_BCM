### Introduction
Each temperature folders has same input files `POSCAR_LPGS`, `run.sh`, `train-from-tape.py`, `calc_vasp.py`, `mpi.sh`. Only the temperature setting is different in `train-from-tape.py` file. The folder `vasp` is an example of DFT setting and automatically created based on `calc_vasp.py` file contents when you running `train-from-tape.py`.

* <ins>required</ins>: `POSCAR_LPGS`, `run.sh`, `train-from-tape.py`, `calc_vasp.py`
* <ins>optional</ins>: `mpi.sh`

The file `mpi.sh` contains job script information what we used. If you are performing calculations under parallel computing, change it appropriately to your environment.

Also, adjust how much CPU to allocate in the settings of the two files below.

`calc_vasp.py`
```shell
# +
from ase.calculators.vasp import Vasp

calc = Vasp(command="mpirun -np {number_of_cpus} vasp_std", xc='pbe', ncore=2,
            directory='vasp', lwave=False, lcharg=False, nelm=300,
            ispin=1, encut=500, lreal='Auto', algo='Fast',
            )
```

`run.sh`
```shell
python -m theforce.calculator.calc_server &
sleep 1
mpirun -np {number_of_cpus} python train-from-tape.py 
```

