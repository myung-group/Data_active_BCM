# This directory contains Python scripts for conducting active NVT and NPT MD simulations, as well as non-active NPH MD simulations.
# Included are two input files (input_nvt.json and input_npt.json) and one initial configuration file each for Ice2Liq, Ice3Liq, Ice5Liq and Liq directories. To build ML models via active NVT and NPT MD simulations, execute the following commands:
```
./run_nvt.sh
./run_npt.sh
```
- Please modify 'run_nvt.sh' and 'run_npt.sh' scripts to build an ML model for Ice2Liq, Ice3Liq, and Ice5Liq.

# Once ML model experts for ice2/liq, ice3/liq, and ice5/liq coexisting systems are built, non-active NPH MD simulations with BCM machine learning potential can be executed using following command:
 ```
 mpirun -np 6 python run_md_nph.py BCM_Ice2Liq/input_nph.json
 ```
- To run NPH MD simulations at different external pressures, please adjust the value of 'press_GPa' in the 'BCM_Ice2Liq/input_nph.json' file accordingly.