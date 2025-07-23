import json
from pathlib import Path

import typer

from veriphix.sampling_circuits import (
    circuit_to_qiskit,
    estimate_circuit_by_expectation_value,
    read_qasm,
)


def regenerate_table(path: Path) -> None:
    table = {}
    for circuit_path in sorted(Path(path).glob("*.qasm")):
        with circuit_path.open() as f:
            circuit = read_qasm(f)
        qc = circuit_to_qiskit(circuit)
        table[circuit_path.name] = estimate_circuit_by_expectation_value(qc)
    print(json.dumps(table))


if __name__ == "__main__":
    typer.run(regenerate_table)
