from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass
import dataclasses
from os.path import exists
from pathlib import Path
from typing import TYPE_CHECKING


import dask.distributed
import numpy as np
import typer
from dask_jobqueue import SLURMCluster
from graphix.noise_models import DepolarisingNoiseModel, NoiseModel
from graphix.sim.density_matrix import DensityMatrixBackend

import veriphix.sampling_circuits.brickwork_state_transpiler
from veriphix.client import Client, TrappifiedSchemeParameters
from veriphix.malicious_noise_model import MaliciousNoiseModel
from veriphix.protocols import VerificationProtocol, FK12, Dummyless
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


class Report:
    pass


@dataclass(frozen=True)
class Result(Report):
    protocol: str
    noise_model: str
    parameter: str
    circuit_label: str
    n_failed_test_rounds: int
    decoded_output: int
    correct_value: int
    match: str
    computation_outcomes_count: int


@dataclass(frozen=True)
class Failure(Report):
    circuit_label: str
    error: str
    

@dataclass(frozen=True)
class Run:
    protocol: VerificationProtocol
    noise_model_label: str
    noise_model: NoiseModel
    index: int
    circuit_label: str

    def run(self, parameters: TrappifiedSchemeParameters) -> Report:
        try:
            # Prepare pattern and client
            pattern = load_pattern_from_circuit(self.circuit_label)
            client = Client(pattern=pattern, protocol=self.protocol, parameters=parameters)
            client.trappifiedScheme.params.threshold = client.trappifiedScheme.params.test_rounds / (
                2 * len(client.test_runs)
            )
    
            # Simulate
            canvas = client.sample_canvas()
            outcomes = client.delegate_canvas(
                canvas=canvas, backend_cls=DensityMatrixBackend, noise_model=self.noise_model
            )
            decision, result, result_analysis = client.analyze_outcomes(canvas=canvas, outcomes=outcomes)
            result = int(result)
            correct_value = find_correct_value(self.circuit_label)
    
            # Parse noise label
            match = re.match(r"(depolarising|malicious)-([0-9.]+)", self.noise_model_label)
            noise_type, param = match.groups() if match else ("unknown", "unknown")
    
            match_flag = "✓" if result == correct_value else "✗"
            return Result(
                protocol=type(self.protocol).__name__,
                noise_model=noise_type,
                parameter=param,
                circuit_label=self.circuit_label,
                n_failed_test_rounds=result_analysis.nr_failed_test_rounds,
                decoded_output=result,
                correct_value=correct_value,
                match=match_flag,
                computation_outcomes_count=result_analysis.computation_count,
            )
        except Exception as e:
            return Failure(self.circuit_label, str(e))


def get_cluster(
    walltime: int | None = None,
    memory: int | None = None,
    cores: int | None = None,
    port: int | None = None,
    scale: int | None = None,
) -> dask.distributed.deploy.cluster.Cluster:
    if walltime is None and memory is None and cores is None and port is None:
        cluster: dask.distributed.deploy.cluster.Cluster = (
            dask.distributed.LocalCluster()  # type: ignore[no-untyped-call]
        )
    else:
        if walltime is None:
            raise ValueError("--walltime <hours> is required for running on cleps")
        if memory is None:
            raise ValueError("--memory <GB> is required for running on cleps")
        if cores is None:
            raise ValueError("--cores <N> is required for running on cleps")
        if port is None:
            raise ValueError("--port <N> is required for running on cleps")
        if scale is None:
            raise ValueError("--scale <N> is required for running on cleps")
        cluster = SLURMCluster(
            account="inria",
            queue="cpu_devel",
            cores=cores,
            memory=f"{memory}GB",
            walltime=f"{walltime}:00:00",
            scheduler_options={"dashboard_address": f":{port}"},
        )
    if scale is not None:
        cluster.scale(scale)
    return cluster


def main(
    walltime: int | None = None,
    memory: int | None = None,
    cores: int | None = None,
    port: int | None = None,
    scale: int | None = None,
):

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
    csv_fields = [field.name for field in dataclasses.fields(Result)]

    runs = [
        Run(protocol, noise_model_label, noise_model, index, circuit_label)
        for protocol in [FK12(manual_colouring=colors), Dummyless()]
        for noise_model_label, noise_model in combined_noise_models.items()
        for index, circuit_label in enumerate(sampled_circuits, 1)
    ]

    csv_exists = exists(csv_path)
    
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        if not csv_exists:
            writer.writeheader()

        cluster = get_cluster(walltime, memory, cores, port, scale)
        dask_client = dask.distributed.Client(cluster)
        futures = []
        for run in runs:
            fut = dask_client.submit(lambda run, parameters: run.run(parameters), run, parameters, pure=False)
            futures.append(fut)

        for fut in dask.distributed.as_completed(futures):
            try:
                report = fut.result()
            except Exception as e:
                print(f"❌ Error retrieving result: {e}")
                continue
            print(report)
            if isinstance(report, Result):
                writer.writerow(dataclasses.asdict(report))
                csvfile.flush()
            if isinstance(report, Failure):
                print(f"❌ Error processing circuit {report.circuit_label}: {report.error}")

    print("✅ All results saved progressively to", csv_path)


if __name__ == "__main__":
    typer.run(main)
