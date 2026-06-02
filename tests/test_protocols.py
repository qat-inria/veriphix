from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pytest
from graphix._linalg import MatGF2
from graphix.random_objects import rand_circuit
from graphix.sim.statevec import StatevectorBackend
from graphix_qasm_parser import OpenQASMParser

from veriphix.blinding import Secrets
from veriphix.client import Client
from veriphix.protocols import (
    FK12,
    Dummyless,
    OddPairGeneratorFn,
    RandomTraps,
    VerificationProtocol,
    odd_pair_generators_bfs,
    odd_pair_generators_exhaustive,
)

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
    def test_average_detection_rate(self, fx_rng: np.random.Generator, protocol_cls:type[VerificationProtocol]) -> None:
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

    @pytest.mark.parametrize(
        "odd_pair_generator",
        [
            odd_pair_generators_bfs,
            odd_pair_generators_exhaustive,
        ],
        ids=["bfs", "exhaustive"],
    )
    def test_dummyless(self, fx_rng: np.random.Generator, odd_pair_generator: OddPairGeneratorFn) -> None:
        nqubits = 2
        depth = 1
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        secrets = Secrets(r=True, a=True, theta=True)
        protocol = Dummyless(odd_pair_generator=odd_pair_generator)
        client = Client(pattern=pattern, secrets=secrets, protocol=protocol, rng=fx_rng)

        stabilizers = [run.stabilizer for run in client.test_runs]
        assert stabilizers, "no test runs generated"

        # Each stabilizer must be a tensor product of I, X, Y only (no Z)
        for stab in stabilizers:
            assert stab.pauli_indices("Z") == [], f"stabilizer contains Z: {stab}"

        # Linear independence over F2: represent each stabilizer as a 2n binary row vector
        # (X-component || Z-component), stack into MatGF2, check rank == |V|-1.
        # stim encodes paulis as: 0=I, 1=X, 2=Y, 3=Z
        n = len(stabilizers[0])
        rows = np.array(
            [
                [int(stab[i] in (1, 2)) for i in range(n)]  # X part
                + [int(stab[i] in (2, 3)) for i in range(n)]  # Z part
                for stab in stabilizers
            ],
            dtype=np.uint8,
        )
        mat = MatGF2(rows)
        rank = mat.compute_rank()

        # Each connected component contributes one degree of freedom (its own "logical qubit"),
        # so the generators must span a space of dimension |V| - n_components.
        n_components = nx.number_connected_components(client.graph)
        assert rank == len(client.graph.nodes) - n_components
