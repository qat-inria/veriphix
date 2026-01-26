from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from graphix.random_objects import rand_circuit
from graphix.sim.statevec import StatevectorBackend

from veriphix.client import Client, Secrets
from veriphix.protocols import (
    FK12,
    RandomTraps,
    VerificationProtocol,
)

if TYPE_CHECKING:
    import numpy as np


class TestProtocols:
    @pytest.mark.parametrize("protocol_class", (FK12, RandomTraps))
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
