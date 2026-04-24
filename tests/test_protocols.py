from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from graphix.random_objects import rand_circuit
from graphix.sim.statevec import StatevectorBackend
from graphix.transpiler import transpile_swaps
from graphix_qasm_parser import OpenQASMParser

from veriphix.blinding import Secrets
from veriphix.client import Client
from veriphix.protocols import (
    FK12,
    RandomTraps,
    VerificationProtocol,
)

if TYPE_CHECKING:
    import numpy as np
    from graphix import Pattern
    from numpy.random import Generator


class TestProtocols:
    @pytest.mark.parametrize("protocol_class", (FK12, RandomTraps))
    def test_noiseless_all_protocols(
        self, fx_rng: np.random.Generator, protocol_class: type[VerificationProtocol]
    ) -> None:
        nqubits = 3
        depth = 5
        circuit = transpile_swaps(rand_circuit(nqubits, depth, fx_rng)).circuit
        pattern = circuit.transpile().pattern
        pattern.minimize_space()

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
            circuit = transpile_swaps(parser.parse_file(Path("tests/test_circuits") / circuit_label)).circuit
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
        circuit = transpile_swaps(rand_circuit(nqubits, depth, fx_rng)).circuit
        # transpile to pattern
        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern.minimize_space()

        # initialise client
        protocol = FK12(manual_colouring=(set([0]), set()))
        client = Client(pattern=pattern, protocol=protocol, autogen=False, rng=fx_rng)
        client.preprocess_pattern()
        client.create_blind_patterns(rng=fx_rng)
        with pytest.raises(ValueError):  # trivially duplicate a node
            protocol.create_test_runs(client=client)

    def test_create_test_run_manual_fail_improper(self, fx_rng: Generator) -> None:
        """testing manual colouring not proper"""

        # generate random circuit
        nqubits = 2
        depth = 1
        circuit = transpile_swaps(rand_circuit(nqubits, depth, fx_rng)).circuit

        # transpile to pattern
        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern.minimize_space()

        nodes = pattern.extract_nodes()

        # initialise client
        protocol = FK12(manual_colouring=(set(nodes), set([next(iter(nodes))])))
        client = Client(pattern=pattern, autogen=False, rng=fx_rng)
        client.preprocess_pattern()
        client.create_blind_patterns(rng=fx_rng)
        with pytest.raises(ValueError):  # trivially duplicate a node
            protocol.create_test_runs(client=client)

    def test_random_traps(self, fx_rng: np.random.Generator) -> None:
        """
        Nothing is done more than in 'test_noiseless_all_protocols'
        """
        nqubits = 3
        depth = 5
        circuit = transpile_swaps(rand_circuit(nqubits, depth, fx_rng)).circuit
        pattern = circuit.transpile().pattern
        pattern.minimize_space()

        secrets = Secrets(r=True, a=True, theta=True)
        protocol = RandomTraps()
        client = Client(pattern=pattern, secrets=secrets, protocol=protocol, rng=fx_rng)
        canvas = client.sample_canvas(rng=fx_rng)
        run_results = client.delegate_canvas(canvas=canvas, backend_cls=StatevectorBackend, rng=fx_rng)
        decision, _, result_analysis = client.analyze_outcomes(canvas=canvas, outcomes=run_results)
        assert decision
        assert result_analysis.nr_failed_test_rounds == 0
