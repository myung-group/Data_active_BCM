import sys 

import torch
from ase.calculators.socketio import SocketClient
# +
from theforce.calculator.socketcalc import SocketCalculator
from theforce.calculator.active import ActiveCalculator, FilterDeltas
from theforce.util.parallel import mpi_init
from theforce.util.aseutil import init_velocities
from ase.build import bulk
from ase.md.npt import NPT 
from ase import units
from ase.calculators.vasp import Vasp
from ase.calculators.emt import EMT 
from ase.io import read, write
#from ase.io.trajectory import Trajectory

process_group = mpi_init()
common = dict(ediff=0.1, fdiff=0.1, process_group=process_group)

_calc_1 = SocketCalculator(script='calc_vasp.py')
calc_1 = ActiveCalculator(calculator=_calc_1,
                        logfile='active.log',
                        pckl='model.pckl/',
                        tape='model.sgpr',test=500,
                        **common)

atoms = read('POSCAR_LGPS')
atoms.calc = calc_1

npt = True
tem = 1100.0
stress = 0.000101325
dt = 1.
ttime = 200*units.fs
ptime = 2000*units.fs
bulk_modulus = 20.43 # no-VDW
pfactor = (ptime**2)*bulk_modulus*units.GPa
init_velocities(atoms, tem)
# make_cell_upper_triangular(atoms)
dyn = NPT(atoms, dt*units.fs, temperature_K=tem, externalstress=stress*units.GPa,
        ttime=ttime, pfactor=pfactor if npt else None, mask=None, trajectory='md.traj',
        append_trajectory=True, loginterval=1)

dyn.run(100000)
