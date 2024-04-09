import numpy as np 
import json 
from ase.io import read 
from ase import units
from ase.md import velocitydistribution as vd

from ase_md_npt import NPT3 
from ase_md_logger import MDLogger3
from ase_mbx import MBX 

from active import ActiveCalculator
from socket_client import SocketCalculator


l_vel_init = True


fname_json = 'input.json'
with open (fname_json) as f:
    json_data = json.load (f)

    atoms = read (json_data['atoms'])
    mbx_calc = SocketCalculator (script=json_data['socket']['scirpt'],
                                 ip=json_data['socket']['ip'],
                                 port=json_data['socket']['port']) #MBX()
    atoms.calc = ActiveCalculator(json_data=json_data,
                            calculator=mbx_calc)
    
    dt_fs = 0.5*units.fs
    ttime = 25.0*units.fs
    ptime = 100.0*units.fs
    bulk_modulus = 137.0
    pfactor = (ptime**2)*bulk_modulus * units.GPa
    temperature_K = 255.0
    temperature = temperature_K * units.kB
    external_stress = 0.4 * units.GPa 

    if l_vel_init:
        vd.MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
        vd.Stationary(atoms)
        vd.ZeroRotation(atoms)

    fixed_temperature = True
    fixed_pressure = True

    if not fixed_temperature:
        ttime=None

    if not fixed_pressure:
        pfactor = None 

    anisotropic = False

    dyn = NPT3 (atoms,
            dt_fs,
            temperature=temperature, 
            externalstress=external_stress,
            ttime=ttime,
            pfactor=pfactor,
            anisotropic=anisotropic,
            trajectory='md_npt.traj',
            logfile=None,
            append_trajectory=True,
            loginterval=200)


    logger = MDLogger3 (dyn=dyn, atoms=atoms, logfile='md_npt.dat', stress=True)
    dyn.attach (logger, 2)
    # upto 100 ps
    dyn.run (20000)

