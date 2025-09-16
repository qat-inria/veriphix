from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
from graphix.random_objects import rand_circuit
from graphix.sim.statevec import StatevectorBackend
from stim import PauliString

from veriphix.client import Client, Secrets
from veriphix.protocols import (
    FK12,
    Dummyless,
    RandomTraps,
    VerificationProtocol,
    generate_graph_stabilizers,
    gf2_solve,
    pauli_to_symplectic,
)


class TestProtocols:
    @pytest.mark.parametrize("protocol_class", (FK12, RandomTraps, Dummyless))
    def test_noiseless_all_protocols(self, fx_rng: np.random.Generator, protocol_class: type[VerificationProtocol]):
        nqubits = 3
        depth = 5
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        protocol = protocol_class()
        client = Client(pattern=pattern, protocol=protocol)
        canvas = client.sample_canvas()
        run_results = client.delegate_canvas(canvas=canvas, backend_cls=StatevectorBackend)
        decision, _, result_analysis = client.analyze_outcomes(canvas=canvas, outcomes=run_results)
        assert decision
        assert result_analysis.nr_failed_test_rounds == 0

    @pytest.mark.parametrize("manual", (True, False))
    def test_FK(self, fx_rng: np.random.Generator, manual: bool):
        import veriphix.sampling_circuits.brickwork_state_transpiler
        from veriphix.sampling_circuits.qasm_parser import read_qasm

        def load_pattern_from_circuit(circuit_label: str):
            with Path(f"circuits/{circuit_label}").open() as f:
                circuit = read_qasm(f)
                pattern = veriphix.sampling_circuits.brickwork_state_transpiler.transpile(circuit)

                pattern.minimize_space()
            return pattern

        with Path("circuits/table.json").open() as f:
            table = json.load(f)
            circuits = list(table.keys())
        random_circuit_label = random.choice(circuits)
        pattern = load_pattern_from_circuit(circuit_label=random_circuit_label)
        colors = veriphix.sampling_circuits.brickwork_state_transpiler.get_bipartite_coloring(pattern=pattern)

        fk_protocol = FK12(manual_colouring=colors) if manual else FK12()
        client = Client(pattern=pattern, protocol=fk_protocol)
        assert client.test_runs != []

    def test_create_test_run_manual_fail(self, fx_rng):
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
        client = Client(pattern=pattern, protocol=protocol, autogen=False)
        client.preprocess_pattern()
        client.create_blind_patterns()
        with pytest.raises(ValueError):  # trivially duplicate a node
            protocol.create_test_runs(client=client)

    def test_create_test_run_manual_fail_improper(self, fx_rng):
        """testing manual colouring not proper"""

        # generate random circuit
        nqubits = 2
        depth = 1
        circuit = rand_circuit(nqubits, depth, fx_rng)
        # transpile to pattern
        pattern = circuit.transpile().pattern
        pattern.standardize()

        nodes, _ = pattern.get_graph()

        # initialise client
        protocol = FK12(manual_colouring=(set(nodes), set([nodes[0]])))
        client = Client(pattern=pattern, autogen=False)
        client.preprocess_pattern()
        client.create_blind_patterns()
        with pytest.raises(ValueError):  # trivially duplicate a node
            protocol.create_test_runs(client=client)

    def test_random_traps(self, fx_rng: np.random.Generator):
        """
        Nothing is done more than in 'test_noiseless_all_protocols'
        """
        nqubits = 3
        depth = 5
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        secrets = Secrets(r=True, a=True, theta=True)
        protocol = RandomTraps()
        client = Client(pattern=pattern, secrets=secrets, protocol=protocol)
        canvas = client.sample_canvas()
        run_results = client.delegate_canvas(canvas=canvas, backend_cls=StatevectorBackend)
        decision, _, result_analysis = client.analyze_outcomes(canvas=canvas, outcomes=run_results)
        assert decision
        assert result_analysis.nr_failed_test_rounds == 0

    def test_dummyless(self, fx_rng: np.random.Generator):
        nqubits = 3
        depth = 5
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        protocol = Dummyless()
        client = Client(pattern=pattern, protocol=protocol)
        nodes = list(client.graph.nodes)
        idx_map = {v: i for i, v in enumerate(nodes)}
        n = len(nodes)
        # Check that they are linearly independent
        # By construction, the rank of the symplectic matrix (built from symplectic vectors stacked as rows) of the Pauli Strings is $n-1$
        # so the $n-1$ strings are linearly independent.

        for run in client.test_runs:
            # Check that there are no Z
            trap_stabilizer = run.stabilizer
            assert trap_stabilizer.pauli_indices("Z") == []

            # Check that they are product of generators
            trap_stabilizer_bin = pauli_to_symplectic(trap_stabilizer)
            graph_stabilizers = generate_graph_stabilizers(client.graph)
            # Construct symplectic matrix: 2n x n
            graph_stabilizers_bin = np.array([pauli_to_symplectic(p) for p in graph_stabilizers]).T

            coeffs = gf2_solve(graph_stabilizers_bin, trap_stabilizer_bin)
            support = [nodes[j] for j in range(n) if coeffs[j] == 1]
            reconstructed = PauliString("I" * n)
            for v in support:
                reconstructed *= graph_stabilizers[idx_map[v]]

            assert reconstructed == trap_stabilizer
