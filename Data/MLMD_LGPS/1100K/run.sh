python -m theforce.calculator.calc_server &
sleep 1
mpirun -np 128 python train-from-tape.py 
