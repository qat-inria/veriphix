from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pytest
import stim
from graphix._linalg import MatGF2
from graphix.random_objects import rand_circuit
from graphix.sim.statevec import StatevectorBackend
from graphix_qasm_parser import OpenQASMParser

from veriphix.blinding import Secrets
from veriphix.client import Client
from veriphix.protocols import (
    FK12,
    Dummyless,
    RandomTraps,
    VerificationProtocol,
)
from veriphix.verifying import TestRun, build_stabilizer

if TYPE_CHECKING:
    from graphix import Pattern
    from numpy.random import Generator



class TestProtocols:
    @pytest.mark.parametrize("protocol_class", (FK12, RandomTraps))
    def test_noiseless_all_protocols(
        self, fx_rng: np.random.Generator, protocol_class: type[VerificationProtocol]
    ) -> None:
        nqubits = 3
        depth = 5
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        protocol = protocol_class()
        client = Client(pattern=pattern, protocol=protocol, rng=fx_rng)
        canvas = client.sample_canvas(rng=fx_rng)
        run_results = client.delegate_canvas(canvas=canvas, backend_cls=StatevectorBackend, rng=fx_rng)
        decision, _, result_analysis = client.analyze_outcomes(canvas=canvas, outcomes=run_results)
        assert decision
        assert result_analysis.nr_failed_test_rounds == 0

    @pytest.mark.parametrize("manual", (True, False))
    def test_FK(self, fx_rng: np.random.Generator, manual: bool) -> None:
        """
        Tests that for a given circuit, we can indeed generate test runs from the graph coloring approach of FK
        """
        parser = OpenQASMParser()

        def load_pattern_from_circuit(circuit_label: str) -> Pattern:
            circuit = parser.parse_file(Path("tests/test_circuits") / circuit_label)
            pattern = circuit.transpile().pattern
            pattern.minimize_space()
            return pattern

        with Path("tests/test_circuits/table.json").open() as f:
            table = json.load(f)
            circuits = list(table.keys())
        pattern = load_pattern_from_circuit(circuit_label=circuits[0])
        # colors = veriphix.sampling_circuits.brickwork_state_transpiler.get_bipartite_coloring(pattern=pattern)

        # fk_protocol = FK12(manual_colouring=colors) if manual else FK12()
        fk_protocol = FK12()
        client = Client(pattern=pattern, protocol=fk_protocol, rng=fx_rng)
        assert client.test_runs != []

    def test_create_test_run_manual_fail(self, fx_rng: Generator) -> None:
        """testing not all qubits in the manual colouring"""

        # generate random circuit
        nqubits = 2
        depth = 1
        circuit = rand_circuit(nqubits, depth, fx_rng)
        # transpile to pattern
        pattern = circuit.transpile().pattern
        pattern.standardize()

        # initialise client
        protocol = FK12(manual_colouring=(set([0]), set()))
        client = Client(pattern=pattern, protocol=protocol, autogen=False, rng=fx_rng)
        client.preprocess_pattern()
        client.create_blind_patterns(rng=fx_rng)
        with pytest.raises(ValueError):  # trivially duplicate a node
            protocol.create_test_runs(graph=client.graph)

    def test_create_test_run_manual_fail_improper(self, fx_rng: Generator) -> None:
        """testing manual colouring not proper"""

        # generate random circuit
        nqubits = 2
        depth = 1
        circuit = rand_circuit(nqubits, depth, fx_rng)
        # transpile to pattern
        pattern = circuit.transpile().pattern
        pattern.standardize()

        nodes = pattern.extract_nodes()

        with pytest.raises(ValueError):  # trivially bad colouring
            FK12(manual_colouring=(set(nodes), set([next(iter(nodes))])))

    def test_random_traps(self, fx_rng: np.random.Generator) -> None:
        """
        Nothing is done more than in 'test_noiseless_all_protocols'
        """
        nqubits = 3
        depth = 5
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        secrets = Secrets(r=True, a=True, theta=True)
        protocol = RandomTraps()
        client = Client(pattern=pattern, secrets=secrets, protocol=protocol, rng=fx_rng)
        canvas = client.sample_canvas(rng=fx_rng)
        run_results = client.delegate_canvas(canvas=canvas, backend_cls=StatevectorBackend, rng=fx_rng)
        decision, _, result_analysis = client.analyze_outcomes(canvas=canvas, outcomes=run_results)
        assert decision
        assert result_analysis.nr_failed_test_rounds == 0

    @pytest.mark.parametrize("protocol_cls", (FK12, Dummyless, RandomTraps))
    def test_average_detection_rate(
        self, fx_rng: np.random.Generator, protocol_cls: type[VerificationProtocol]
    ) -> None:
        nqubits = 2
        depth = 2
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        protocol = protocol_cls()
        graph = pattern.extract_graph()
        trs = protocol.create_test_runs(graph=graph, rng=fx_rng)

        nodes = list(graph.nodes)
        n = len(nodes)

        n_dev = 10
        n_test_runs = 100
        detections = 0
        for _ in range(n_dev):
            # Fixed arbitrary Pauli deviation support:
            # these are the qubits where the Pauli is X or Y after twirling.
            error_size = int(fx_rng.integers(1, n + 1))
            error_support = frozenset(fx_rng.choice(nodes, size=error_size, replace=False).tolist())

            detections = 0
            for __ in range(n_test_runs):
                test_run = protocol.sample_test_run(
                    graph=graph,
                    test_runs=trs,
                    rng=fx_rng,
                )
                detected = sum([(len(error_support & trap) % 2) == 1 for trap in test_run.traps]) > 0
                detections += int(detected)

            detection_rate = detections / (n_test_runs)
            expected = protocol.detection_rate
            eps = 0.15
            # With 100 samples, allow statistical slack.
            assert expected - eps <= detection_rate, (
                f"Expected ≈{expected} detection rate, got {detection_rate:.3f}, support: {error_support}"
            )

    def test_dummyless(self, fx_rng: np.random.Generator) -> None:
        nqubits = 2
        depth = 1
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        secrets = Secrets(r=True, a=True, theta=True)
        protocol = Dummyless()
        client = Client(pattern=pattern, secrets=secrets, protocol=protocol, rng=fx_rng)

        assert client.test_runs, "no test runs generated"

        stabilizers = [run.stabilizer for run in client.test_runs]
        assert_no_z(stabilizers)
        assert_linearly_independent(stabilizers, client.graph)
        assert_from_canonical_basis(client.test_runs, client.graph, len(client.graph))

def assert_no_z(stabilizers: list[stim.PauliString]) -> None:
    for stab in stabilizers:
        assert stab.pauli_indices("Z") == [], f"stabilizer contains Z: {stab}"


def assert_linearly_independent(stabilizers: list[stim.PauliString], graph: nx.Graph) -> None:
    n = len(stabilizers[0])
    rows = np.array(
        [
            [int(stab[i] in (1, 2)) for i in range(n)] + [int(stab[i] in (2, 3)) for i in range(n)]
            for stab in stabilizers
        ],
        dtype=np.uint8,
    )
    rank = MatGF2(rows).compute_rank()
    n_components = nx.number_connected_components(graph)
    expected_rank = len(graph.nodes) - n_components
    assert rank == expected_rank, f"expected rank {expected_rank}, got {rank}"


def assert_from_canonical_basis(test_runs: list[TestRun], graph: nx.Graph, n_qubits: int) -> None:
    """Each stabilizer must equal the product of per-node canonical stabilizers for its trap."""
    canonical: dict[int, stim.PauliString] = {
        node: build_stabilizer(graph, n_qubits, frozenset({frozenset({node})}))
        for node in graph.nodes
    }
    for run in test_runs:
        (trap,) = run.traps
        expected = stim.PauliString(n_qubits)
        for v in trap:
            expected *= canonical[v]
        assert run.stabilizer == expected, (
            f"stabilizer for trap {set(trap)} does not match product of canonical stabilizers:\n"
            f"  got      {run.stabilizer}\n"
            f"  expected {expected}"
        )
