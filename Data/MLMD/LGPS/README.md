# NPT ML-MD simulations for LGPS

This directory contains python scripts for conducting NPT MD simulations for LGPS system of several temperatures using the Vienna Ab-initio Simulation Package(VASP) at the Perdew–Burke–Ernzerhof (PBE) functional level to train the SGPR potential.
Same input files `POSCAR_LPGS`, `run.sh`, `train-from-tape.py`, `calc_vasp.py` are containing each folders. Only the temperature is different in `train-from-tape.py`. The folder `vasp` is an example of VASP(DFT) setting and automatically generated based on `calc_vasp.py` parameters when you execute `run.sh`. To build ML models via NPT MD simulations, execute the following commands:
```
./run.sh
```

- Before execute `./run.sh` commands, please modify `calc_vasp.py` and `run.sh` scripts how many CPUs to allocate.

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
mpirun -np {number_of_cpus} python train-from-tape.py 
```

- When ML-MD is performed for each temperature, DFT calculated data is stored every 500 steps at `active_FP.traj`. You can reproduce `LGPS_train.traj` and `LGPS_test.traj` by execute notebook code.

