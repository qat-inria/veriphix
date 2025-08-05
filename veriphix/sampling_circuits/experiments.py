from pathlib import Path

import numpy as np
from graphix import Circuit

from veriphix.sampling_circuits.sampling_circuits import (
    sample_circuits,
    sample_truncated_circuit,
)

ncircuits = 1000
nqubits = 5
depth = 30
p_gate = 0.5
p_cnot = 0.25
p_cnot_flip = 0.5
p_rx = 0.5
seed = 1729


def get_circuit(n: int) -> Circuit:
    """Return a circuit given its number, with the experiment parameters."""
    sequence = np.random.SeedSequence(entropy=seed)
    circuit_seed = sequence.spawn(ncircuits)[n]
    return sample_truncated_circuit(
        nqubits=nqubits,
        depth=depth,
        rng=np.random.default_rng(circuit_seed),
        p_gate=p_gate,
        p_cnot=p_cnot,
        p_cnot_flip=p_cnot_flip,
        p_rx=p_rx,
    )


def run_sample_circuits(target: Path) -> None:
    sample_circuits(
        ncircuits=ncircuits,
        nqubits=nqubits,
        depth=depth,
        p_gate=p_gate,
        p_cnot=p_cnot,
        p_cnot_flip=p_cnot_flip,
        p_rx=p_rx,
        seed=seed,
        target=target,
    )


if __name__ == "__main__":
    run_sample_circuits(Path("circuits/"))
