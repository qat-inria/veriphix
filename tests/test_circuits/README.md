These circuits have been generated with the following veriphix commit hash: b7e7af05b119dfdbd370810e60e7c75b503e7769

To reproduce these samples, you may run the following command:
```
git clone https://github.com/qat-inria/veriphix.git
git checkout b7e7af05b119dfdbd370810e60e7c75b503e7769
python -m veriphix.sampling_circuits.sampling_circuits --ncircuits 10 --nqubits 4 --depth 5 --p-gate 0.5 --p-cnot 0.25 --p-cnot-flip 0.5 --p-rx 0.5 --seed 1729 --target circuits
```
