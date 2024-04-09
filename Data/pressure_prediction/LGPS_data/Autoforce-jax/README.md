# AutoForce-jax
You can reproduce `test_kv.dat`, `test_old.dat` files. Follow the instructions below and execute command.

1) Build a Model with a virial Kernel implemented.
```shell
python main.py TEST/input_train_kv.json
```
   
2) Test this Model. Test requires `model_train_kv.pckl` and `model_train_kv.sgpr`.
```shell
python main.py TEST/input_test_kv.json
```
   
3) Build a Model without a virial Kernel implemented.
```shell
python main.py TEST/input_train_old.json
```
   
4) Test this Model. Test requires `model_old_kv.pckl` and `model_old_kv.sgpr`.
```shell
python main.py TEST/input_test_old.json
```

