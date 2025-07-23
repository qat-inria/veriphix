from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import git
import matplotlib.pyplot as plt
import numpy as np
import qiskit
import qiskit.qasm2
import typer
from graphix import Circuit
from graphix.instruction import Instruction, InstructionKind
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Pauli, Statevector  # type: ignore[attr-defined]
from qiskit_aer.primitives import SamplerV2  # type: ignore[attr-defined]
from tqdm import tqdm

from veriphix.sampling_circuits.brickwork_state_transpiler import (
    XZ,
    Brick,
    Layer,
    identity,
    layers_to_circuit,
    transpile_to_layers,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


def sample_angle(rng: np.random.Generator) -> float:
    # return rng.random() * math.pi * 2
    return (1 + rng.integers(7)) * np.pi / 4


def sample_circuit(
    nqubits: int,
    depth: int,
    rng: np.random.Generator,
    p_gate: float = 0.5,
    p_cnot: float = 0.5,
    p_cnot_flip: float = 0.5,
    p_rx: float = 0.5,
) -> Circuit:
    """Generates a quantum circuit with the given number of qubits and depth.

    At each layer (depth), the circuit iterates through the
    qubits. For each qubit,

    - a gate is applied with probability `p_gate` (otherwise, the
      qubit is skipped);

    - in the case a gate is applied, if there is room (i.e. not on the
      last qubit) and with probability `p_cnot`, a controlled-NOT (CX)
      gate is applied between the current qubit and the next one
      (there will be no other gate applied to the next qubit for this
      layer);

    - in the CNOT case: if the previous instruction applied on this
      qubit pair was a CNOT, the reversed CNOT where control and
      target are flipped is applied; otherwise, with probability
      `p_cnot_flip`, the target is the current qubit and the control
      the next one; otherwise the control is the current qubit and the
      target the next one;

    - in the case a gate is applied but not a CNOT, a rotation gate
      with a random angle is applied to the current qubit: if the
      previous instruction applied on this qubit was a rotation (RX or
      RZ), then a rotation on the other axis is applied (RZ or RX);
      otherwise with probability `p_rx`, the rotation is RX, otherwise
      RZ.

    The circuit is then completed by `complete_circuit` to ensure that
    each gate can affect the measured qubit 0 by introducing
    additional CNOTs.
    """
    circuit = Circuit(nqubits)
    # Last gate for each qubit ("rx", "rz", "control", or "target")
    last_gate_by_qubit: dict[int, str] = {}
    for _ in range(depth):
        qubit = 0
        while qubit < nqubits:
            if rng.random() < p_gate:
                last = last_gate_by_qubit.get(qubit)
                # Check if there's room for a CX gate and with probability p_cnot, apply CX.
                if qubit < nqubits - 1 and rng.random() < p_cnot:
                    last_next = last_gate_by_qubit.get(qubit + 1, None)
                    if (last, last_next) == ("control", "target") or (
                        (last, last_next) != ("target", "control") and rng.random() < p_cnot_flip
                    ):
                        control = qubit + 1
                        target = qubit
                    else:
                        control = qubit
                        target = qubit + 1
                    circuit.cnot(control, target)
                    last_gate_by_qubit[control] = "control"
                    last_gate_by_qubit[target] = "target"
                    qubit += 2  # Skip the next qubit since it's already involved in CX
                else:
                    angle = sample_angle(rng)
                    # With probability p_rx, apply RX; otherwise, apply RZ.
                    if last == "rz" or (last != "rx" and rng.random() < p_rx):
                        circuit.rx(qubit, angle)
                        last_gate_by_qubit[qubit] = "rx"
                    else:
                        circuit.rz(qubit, angle)
                        last_gate_by_qubit[qubit] = "rz"
                    qubit += 1
            else:
                # No gate applied; move to the next qubit.
                qubit += 1
    complete_circuit(circuit, p_cnot_flip, rng)
    return circuit


def sample_truncated_circuit(
    nqubits: int,
    depth: int,
    rng: np.random.Generator,
    p_gate: float = 0.5,
    p_cnot: float = 0.5,
    p_cnot_flip: float = 0.5,
    p_rx: float = 0.5,
) -> Circuit:
    # return sample_circuit(nqubits, depth, rng, p_gate, p_cnot, p_cnot_flip, p_rx)
    while True:
        while True:
            circuit = sample_circuit(nqubits, 2 * depth, rng, p_gate, p_cnot, p_cnot_flip, p_rx)
            layers = transpile_to_layers(circuit)
            if len(layers) >= depth:
                break
        if layers[-1].odd == bool(depth % 2):
            truncated_layers = layers[-depth + 1 :]
            last_layer_odd = not bool(depth % 2)
            last_layer_len = nqubits // 2 - 1 if last_layer_odd else nqubits // 2
            first_brick = identity()
            first_brick.top.add(XZ.X, math.pi / 4)
            bricks: list[Brick] = [first_brick]
            bricks.extend(identity() for _ in range(last_layer_len - 1))
            truncated_layers.append(
                Layer(
                    odd=last_layer_odd,
                    bricks=bricks,
                )
            )
        else:
            truncated_layers = layers[-depth:]
        assert not truncated_layers[0].odd
        circuit = layers_to_circuit(truncated_layers)
        rotated = set()
        for instr in circuit.instruction:
            # Use of `==` here for mypy
            if instr.kind == InstructionKind.RX or instr.kind == InstructionKind.RZ:
                rotated.add(instr.target)
        # if rotated != set(range(nqubits)):
        #    continue
        if len(transpile_to_layers(circuit)) == depth:
            break
    return circuit


def complete_circuit(circuit: Circuit, p_cnot_flip: float, rng: np.random.Generator) -> None:
    """
    Complete circuit with CNOTs so that all gates can affect qubit 0.
    """
    reachable = 0  # tracks the highest qubit index that can affect qubit 0

    def add_cnots(target: int) -> None:
        for i in range(target, reachable, -1):
            if rng.random() < p_cnot_flip:
                circuit.cnot(i, i - 1)
            else:
                circuit.cnot(i - 1, i)

    for instr in reversed(circuit.instruction):
        # Use of `if` instead of `match` here for mypy
        if instr.kind == InstructionKind.CNOT:
            min_qubit = min(instr.control, instr.target)
            if min_qubit < reachable:
                continue
            add_cnots(min_qubit)
            reachable = min_qubit + 1
        # Use of `==` here for mypy
        elif instr.kind == InstructionKind.RX or instr.kind == InstructionKind.RZ:
            if instr.target <= reachable:
                continue
            add_cnots(instr.target)
            reachable = instr.target


def strip_circuit(circuit: Circuit) -> None:
    """
    Strip circuit so that gates are kept only if they can affect qubit 0.

    Rotations or CNOT on other qubits that are not followed with a CNOT
    connecting them to qubit 0 are removed.

    This method is deprecated in favor of `complete_circuit`.
    """
    # Initialize an empty list for the instructions that remain after stripping.
    new_instructions: list[Instruction] = []
    # 'reachable' tracks the index of the last qubit that can affect qubit 0.
    reachable = 0
    for instr in reversed(circuit.instruction):
        # Use of `if` instead of `match` here for mypy
        if instr.kind == InstructionKind.CNOT:
            min_qubit = min(instr.control, instr.target)
            # If the control qubit is beyond the current reachable range,
            # the gate cannot affect qubit 0 and is removed.
            if min_qubit > reachable:
                continue
            # If the control qubit is exactly at the reachable boundary,
            # this CX gate extends the influence to the next qubit.
            if min_qubit == reachable:
                reachable += 1
            # Keep the instruction.
            new_instructions.append(instr)
        # Use of `==` here for mypy
        elif instr.kind == InstructionKind.RX or instr.kind == InstructionKind.RZ:
            # Keep the rotation only if it can affect a qubit within the reachable range.
            if instr.target <= reachable:
                new_instructions.append(instr)
    # The instructions were collected in reverse order; reverse them to restore original order.
    new_instructions.reverse()
    # Replace the original instruction list with the new, stripped list.
    circuit.instruction = new_instructions


def add_hadamard_on_inputs(qc: QuantumCircuit) -> None:
    for qubit in range(qc.num_qubits):
        qc.h(qubit)


def circuit_to_qiskit(c: Circuit, hadamard_on_inputs: bool = False) -> QuantumCircuit:
    """
    Convert a Graphix circuit to a Qiskit QuantumCircuit.

    Parameters:
        c (Circuit): Graphix circuit

    Returns:
        QuantumCircuit: A Qiskit QuantumCircuit representing the custom circuit.

    Raises:
        ValueError: If an instruction type is not supported.
    """
    qc = QuantumCircuit(QuantumRegister(c.width))
    if hadamard_on_inputs:
        add_hadamard_on_inputs(qc)
    for instr in c.instruction:
        # Use of `if` instead of `match` here for mypy
        if instr.kind == InstructionKind.CNOT:
            # Qiskit's cx method expects (control, target).
            qc.cx(instr.control, instr.target)
        elif instr.kind == InstructionKind.RX:
            qc.rx(instr.angle, instr.target)
        elif instr.kind == InstructionKind.RZ:
            qc.rz(instr.angle, instr.target)
        else:
            raise ValueError(f"Unsupported instruction: {instr.kind}")
    return qc


def copy_qiskit_circuit_with_hamadard_on_inputs(qc: QuantumCircuit) -> QuantumCircuit:
    qc_copy = QuantumCircuit(QuantumRegister(qc.num_qubits), ClassicalRegister(1))
    add_hadamard_on_inputs(qc_copy)
    qc_copy.append(qc.to_instruction(), range(qc.num_qubits))
    return qc_copy.decompose()


def estimate_circuit_by_sampling(qc: QuantumCircuit, seed: int | None = None) -> float:
    """
    Estimate the probability of measuring the '1' outcome on the first qubit.

    This is an alternative method for estimating probability by sampling.
    This method is deprecated in favor of `estimate_circuit_expectation_value`,
    which is more accurate, deterministic and faster.
    """
    # Copy the circuit before adding a measure to qubit 0
    qc_copy = copy_qiskit_circuit_with_hamadard_on_inputs(qc)
    qc_copy.h(0)
    qc_copy.measure(0, 0)
    nb_shots = 2 << 12
    sampler = SamplerV2(seed=seed)
    job = sampler.run([qc_copy], shots=nb_shots)
    job_result = job.result()
    nb_one_outcomes = sum(next(iter(job_result[0].data.values())).bitcount())
    assert nb_one_outcomes.is_integer()
    return int(nb_one_outcomes) / nb_shots


def estimate_circuit_by_expectation_value(qc: QuantumCircuit) -> float:
    """
    Estimate the probability of measuring the '1' outcome on the first qubit.

    The observable is chosen as Z on the first qubit (and I on all others),
    so that the expectation value <Z> is computed on the first qubit.
    Given that for a qubit in state |ψ⟩:
        <Z> = p(0) - p(1)
    the probability of outcome '1' is computed as:
        p(1) = (1 - <Z>) / 2
    """
    qc_copy = copy_qiskit_circuit_with_hamadard_on_inputs(qc)
    # Get the statevector for the circuit
    sv = Statevector.from_instruction(qc_copy)
    # Compute the expectation value of the observable
    exp_val = sv.expectation_value(Pauli("X"), [0])
    assert np.imag(exp_val) == 0
    # p(1) = (1 - <Z>)/2
    return (1 - np.real(exp_val)) / 2


def estimate_circuits(
    circuits: Iterable[QuantumCircuit],
) -> list[tuple[QuantumCircuit, float]]:
    return [(circuit, estimate_circuit_by_expectation_value(circuit)) for circuit in tqdm(list(circuits))]


def save_circuits(circuits: list[tuple[QuantumCircuit, float]], path: Path) -> None:
    table = {}
    maxlen = int(np.log10(len(circuits) - 1) + 1)
    for i, (circuit, p) in enumerate(circuits):
        filename = f"circuit{str(i).zfill(maxlen)}.qasm"
        with (path / filename).open("w") as f:
            qiskit.qasm2.dump(circuit, f)
        table[filename] = p
    with (path / "table.json").open("w") as f:
        json.dump(table, f)


def plot_distribution(circuits: list[tuple[QuantumCircuit, float]], filename: Path) -> None:
    samples = [p for _circuit, p in circuits]
    plt.hist(samples, bins=30, edgecolor="black", alpha=0.7)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Random Samples (0 to 1)")
    plt.savefig(filename)


def sample_circuits(
    ncircuits: Annotated[int, typer.Option(..., help="Number of circuits")],
    nqubits: Annotated[int, typer.Option(..., help="Number of qubits")],
    depth: Annotated[int, typer.Option(..., help="Circuit depth")],
    p_gate: Annotated[float, typer.Option(..., help="Probability of applying a gate")],
    p_cnot: Annotated[float, typer.Option(..., help="Probability of applying a CNOT gate")],
    p_cnot_flip: Annotated[float, typer.Option(..., help="Probability of flipping a CNOT gate")],
    p_rx: Annotated[float, typer.Option(..., help="Probability of applying an RX gate")],
    seed: Annotated[int, typer.Option(..., help="Random seed")],
    target: Annotated[Path, typer.Option(..., help="Target directory")],
) -> None:
    params = locals()
    sequence = np.random.SeedSequence(entropy=seed)
    target.mkdir()
    circuits = [
        sample_truncated_circuit(
            nqubits=nqubits,
            depth=depth,
            rng=np.random.default_rng(seed),
            p_gate=p_gate,
            p_cnot=p_cnot,
            p_cnot_flip=p_cnot_flip,
            p_rx=p_rx,
        )
        for seed in sequence.spawn(ncircuits)
    ]
    qiskit_circuits = map(circuit_to_qiskit, circuits)
    estimated_circuits = estimate_circuits(qiskit_circuits)
    save_circuits(estimated_circuits, target)
    plot_distribution(estimated_circuits, target / "distribution.svg")
    arg_str = " ".join(f"--{key.replace('_', '-')} {value}" for key, value in params.items())
    command_line = f"python -m veriphix.sampling_circuits.sampling_circuits {arg_str}"
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    with (target / "README.md").open("w") as f:
        f.write(
            f"""These circuits have been generated with the following veriphix commit hash: {sha}

To reproduce these samples, you may run the following command:
```
git clone https://github.com/qat-inria/veriphix.git
git checkout {sha}
{command_line}
```
"""
        )


if __name__ == "__main__":
    typer.run(sample_circuits)
