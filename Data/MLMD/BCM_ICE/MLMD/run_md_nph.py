# +
import sys 
import json 

from ase import units

from theforce.calculator.active_bcm import BCMActiveCalculator
#from theforce.calculator.socketcalc import SocketCalculator
#from theforce.similarity.sesoap import SeSoapKernel
#from theforce.util.aseutil import init_velocities  # , make_cell_upper_triangular
from theforce.util.parallel import mpi_init

from ase_md_npt import NPT3
from ase_md_logger import MDLogger3

from ase.io import read 
import numpy as np 

fname_json = 'input.json'
if len(sys.argv) > 1:
    fname_json = sys.argv[1]

fd = open (fname_json)
json_data = json.load (fd)

# The NPH MD simulation should be run with the none-active MLModel
mbx_calc =  None
#if json_data['active_learning']:
#    mbx_calc = SocketCalculator (script="mbx_calc.py")


# (2) the kernel and the active learning calculator
#     We will build the kernel-based Model via an active NVT MD simulation

kern = 'pckl'
#if json_data['model_init']:
#    lmax, nmax, exponent, cutoff = 3, 3, 4, 6.0
#    kern = SeSoapKernel(lmax, nmax, exponent, cutoff)  # <- universal (all atoms types)

calc = BCMActiveCalculator(
    covariance=kern,
    calculator=mbx_calc,  
    logfile=json_data['fname_active_logfile'],
    ediff=json_data['ediff'], # 0.08
    fdiff=json_data['fdiff'],  # control params for accuracy/speed tradeoff
    process_group=mpi_init(),  # for mpi parallelism
    pckl=json_data['model_pckl'],  # for continuous pickling (saving) the model
    tape=json_data['model_tape'],  # for saving the training data step by step
    test=json_data['test_nstep'],  # test 100 steps after last dft (None for no test)
)


# (3). define the system and set its calculator


atoms = read (json_data['fname_atoms'], index=-1)
atoms.set_calculator(calc)


json_md = json_data['md']
dt_fs = json_md['dt_fs']*units.fs
ttime = 25.0*units.fs
ptime = 100.0*units.fs
bulk_modulus = 137.0
pfactor = (ptime**2)*bulk_modulus * units.GPa
temperature_K = json_md['temperature_K']
temperature = temperature_K * units.kB
external_stress = json_md['press_GPa'] * units.GPa 

#if json_md['vel_init']:
#    init_velocities (atoms, temperature_K)


#if not json_md['fixed_temperature']:
ttime=None

#if not json_md['fixed_pressure']:
#    pfactor = None 



dyn = NPT3 (atoms,
            dt_fs,
            temperature=temperature, 
            externalstress=external_stress,
            ttime=ttime,
            pfactor=pfactor,
            anisotropic=json_md['anisotropic'],
            trajectory=json_md['fname_md_traj'],
            logfile=None,
            append_trajectory=True,
            loginterval=json_md['md_nsave_traj'])


logger = MDLogger3 (dyn=dyn, atoms=atoms, 
                    logfile=json_md['fname_md_logfile'], stress=True)
dyn.attach (logger, json_md['md_nprint_logfile'])
dyn.run (json_md['md_nstep'])

