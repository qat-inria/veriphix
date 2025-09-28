from __future__ import annotations

import csv
import json
import random
import re
from os.path import exists
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from graphix.noise_models import DepolarisingNoiseModel
from graphix.sim.density_matrix import DensityMatrixBackend

import veriphix.sampling_circuits.brickwork_state_transpiler
from veriphix.client import Client, TrappifiedSchemeParameters
from veriphix.malicious_noise_model import MaliciousNoiseModel
from veriphix.protocols import FK12, Dummyless
from veriphix.sampling_circuits.qasm_parser import read_qasm

if TYPE_CHECKING:
    from graphix.pattern import Pattern

# === Helper functions ===


def find_correct_value(circuit_name: str) -> int:
    with Path("circuits/table.json").open() as f:
        table = json.load(f)
        return round(table[circuit_name])


def load_pattern_from_circuit(circuit_label: str) -> Pattern:
    with Path(f"circuits/{circuit_label}").open() as f:
        circuit = read_qasm(f)
        pattern = veriphix.sampling_circuits.brickwork_state_transpiler.transpile(circuit)
        pattern.minimize_space()
    return pattern


# === Load sampled circuits ===

sampled_circuits = []
with Path("demo/sampled_circuits.txt").open() as f:
    sampled_circuits = [line.strip() for line in f if line.strip()]

# === Define noise models ===


depol_param_sweep = [1e-4, 5e-4, 1e-3, 2.7e-3, 5e-3, 1e-2, 5e-2]
depol = {f"depolarising-{p}": DepolarisingNoiseModel(entanglement_error_prob=p) for p in depol_param_sweep}

default_pattern = load_pattern_from_circuit(random.choice(sampled_circuits))
colors = veriphix.sampling_circuits.brickwork_state_transpiler.get_bipartite_coloring(default_pattern)
output_node = default_pattern.output_nodes[0]

malicious_global_param_sweep = np.linspace(0, 1, 3)
malicious_global = {
    f"malicious-{p}": MaliciousNoiseModel(nodes=[output_node], prob=p) for p in malicious_global_param_sweep
}

combined_noise_models = {
    **malicious_global,
    # **depol
}

# === Setup protocol parameters ===

parameters = TrappifiedSchemeParameters(comp_rounds=100, test_rounds=100, threshold=None)

# === CSV Setup ===

csv_path = "demo/results.csv"
csv_fields = [
    "protocol",
    "noise_model",
    "parameter",
    "circuit_label",
    "n_failed_test_rounds",
    "decoded_output",
    "correct_value",
    "match",
    "computation_outcomes_count",
]

csv_exists = exists(csv_path)

with open(csv_path, "a", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
    if not csv_exists:
        writer.writeheader()

    """
    TODO: La boucle doit être homogène. Ici, FK12 doit etre une fonction de: "coloriage ou None" vers une instance de FK12 ;
    Dummyless doit être une lambda qui ignore son argument et renvoie une instance de Dummyless

    TODO: ou eventuellement, réordonner les boucles
    """
    for protocol in [FK12(manual_colouring=colors), Dummyless()]:
        print("PROTOCOL:", type(protocol).__name__)

        for noise_model_label, noise_model in combined_noise_models.items():
            print("NOISE:", noise_model_label)

            for i, circuit in enumerate(sampled_circuits, 1):
                print(f"CIRCUIT {i}/{len(sampled_circuits)}")

                try:
                    # Prepare pattern and client
                    pattern = load_pattern_from_circuit(circuit)
                    client = Client(pattern=pattern, protocol=protocol, parameters=parameters)
                    client.trappifiedScheme.params.threshold = client.trappifiedScheme.params.test_rounds / (
                        2 * len(client.test_runs)
                    )

                    # Simulate
                    canvas = client.sample_canvas()
                    outcomes = client.delegate_canvas(
                        canvas=canvas, backend_cls=DensityMatrixBackend, noise_model=noise_model
                    )
                    decision, result, result_analysis = client.analyze_outcomes(canvas=canvas, outcomes=outcomes)
                    result = int(result)
                    print(decision, result, result_analysis)
                    correct_value = find_correct_value(circuit)

                    # Parse noise label
                    match = re.match(r"(depolarising|malicious)-([0-9.]+)", noise_model_label)
                    noise_type, param = match.groups() if match else ("unknown", "unknown")

                    match_flag = "✓" if result == correct_value else "✗"

                    # Write result row
                    writer.writerow(
                        {
                            "protocol": type(protocol).__name__,
                            "noise_model": noise_type,
                            "parameter": param,
                            "circuit_label": circuit,
                            "n_failed_test_rounds": result_analysis.nr_failed_test_rounds,
                            "decoded_output": result,
                            "correct_value": correct_value,
                            "match": match_flag,
                            "computation_outcomes_count": result_analysis.computation_count,
                        }
                    )

                    csvfile.flush()

                except Exception as e:
                    print(f"❌ Error processing circuit {circuit}: {e}")
                    continue  # Optional: you might want to log these errors too

print("✅ All results saved progressively to", csv_path)
