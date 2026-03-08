from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from graphix.noise_models import DepolarisingNoiseModel
from graphix.random_objects import rand_circuit
from graphix.sim.density_matrix import DensityMatrixBackend
from graphix.sim.statevec import StatevectorBackend
from graphix.states import BasicStates

from tests.qasm_parser import read_qasm
from veriphix.blinding import Secrets
from veriphix.client import Client
from veriphix.verifying import QuantumComputationResult, TrappifiedSchemeParameters

if TYPE_CHECKING:
    from graphix.measurements import Outcome
    from graphix.pattern import Pattern
    from numpy.random import Generator


def load_pattern_from_circuit(circuit_label: str) -> Pattern:
    with Path(f"tests/test_circuits/{circuit_label}").open() as f:
        circuit = read_qasm(f)
        pattern = circuit.transpile().pattern

        pattern.minimize_space()
    return pattern


class TestVBQC:
    @pytest.mark.parametrize("blind", (False, True))
    def test_trap_delegated(self, fx_rng: np.random.Generator, blind: bool) -> None:
        nqubits = 3
        depth = 5
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        secrets = Secrets(r=blind, a=blind, theta=blind)
        client = Client(pattern=pattern, secrets=secrets, rng=fx_rng)
        for test_run in client.test_runs:
            backend = StatevectorBackend()
            trap_outcomes = test_run.delegate(backend=backend, rng=fx_rng).trap_outcomes
            assert sum(trap_outcomes.values()) == 0

    def test_sample_canvas(self, fx_rng: Generator) -> None:
        nqubits = 3
        depth = 5
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        client = Client(pattern=pattern, rng=fx_rng)

        assert client.sample_canvas(rng=fx_rng)
        # Just tests that it runs

    def test_delegate_canvas(self, fx_rng: Generator) -> None:
        nqubits = 3
        depth = 5
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        svbackend = StatevectorBackend()
        simulated_pattern_output = pattern.simulate_pattern(backend=svbackend, rng=fx_rng)
        simulated_circuit_output = circuit.simulate_statevector().statevec

        parameters = TrappifiedSchemeParameters(comp_rounds=10, test_rounds=10, threshold=0)
        client = Client(pattern=pattern, parameters=parameters, classical_output=False, rng=fx_rng)

        canvas = client.sample_canvas(rng=fx_rng)
        outcomes = client.delegate_canvas(canvas=canvas, backend_cls=StatevectorBackend, rng=fx_rng)
        for result in outcomes.values():
            if isinstance(result, QuantumComputationResult):
                np.testing.assert_almost_equal(
                    np.abs(
                        np.dot(result.output_state.psi.flatten().conjugate(), simulated_pattern_output.psi.flatten())
                    ),
                    1,
                )
                np.testing.assert_almost_equal(
                    np.abs(
                        np.dot(result.output_state.psi.flatten().conjugate(), simulated_circuit_output.psi.flatten())
                    ),
                    1,
                )
        # Just tests that it runs
        """
        TODO, in the tests:
        - Noiseless, quantum outputs: check evolution of the state for all the comp. runs, and check for no trap failures
        """

    @pytest.mark.parametrize("blind", (False, True))
    def test_analyze_outcomes(self, fx_rng: Generator, blind: bool) -> None:
        nqubits = 3
        depth = 3
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        secrets = Secrets(r=blind, a=blind, theta=blind)

        parameters = TrappifiedSchemeParameters(comp_rounds=50, test_rounds=50, threshold=10)
        client = Client(pattern=pattern, secrets=secrets, parameters=parameters, rng=fx_rng)

        canvas = client.sample_canvas(rng=fx_rng)
        outcomes = client.delegate_canvas(canvas=canvas, backend_cls=StatevectorBackend, rng=fx_rng)

        # only for BQP
        assert client.analyze_outcomes(canvas, outcomes)

    @pytest.mark.parametrize("blind", (False, True))
    @pytest.mark.xfail(reason="Incorrect value")
    def test_BQP_circuit(self, fx_rng: Generator, blind: bool) -> None:
        with Path("tests/test_circuits/table.json").open() as f:
            table = json.load(f)
            circuits = [name for name, prob in table.items()]
        for circuit_label in circuits:
            pattern = load_pattern_from_circuit(circuit_label=circuit_label)

            secrets = Secrets(r=blind, a=blind, theta=blind)

            parameters = TrappifiedSchemeParameters(comp_rounds=18, test_rounds=18, threshold=5)
            client = Client(pattern=pattern, secrets=secrets, parameters=parameters, rng=fx_rng)

            canvas = client.sample_canvas(rng=fx_rng)
            outcomes = client.delegate_canvas(canvas=canvas, backend_cls=StatevectorBackend, rng=fx_rng)
            decision, result, _ = client.analyze_outcomes(canvas, outcomes)
            assert decision
            assert result is not None
            assert int(result) == find_correct_value(circuit_label)

    @pytest.mark.parametrize("blind", (False, True))
    def test_noiseless(self, fx_rng: Generator, blind: bool) -> None:
        nqubits = 3
        depth = 3
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        states = [BasicStates.PLUS for _ in pattern.input_nodes]

        secrets = Secrets(a=blind, r=blind, theta=blind)

        client = Client(pattern=pattern, input_state=states, secrets=secrets, rng=fx_rng)
        noise_model = DepolarisingNoiseModel(
            measure_error_prob=0,
            entanglement_error_prob=0,
            x_error_prob=0,
            z_error_prob=0,
            measure_channel_prob=0,
        )
        for test_run in client.test_runs:
            client.refresh_randomness(rng=fx_rng)
            backend = DensityMatrixBackend()
            trap_outcomes = test_run.delegate(backend=backend, noise_model=noise_model, rng=fx_rng).trap_outcomes
            assert sum(trap_outcomes.values()) == 0

    def test_noisy(self, fx_rng: Generator) -> None:
        nqubits = 3
        depth = 3
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        states = [BasicStates.PLUS for _ in pattern.input_nodes]

        secrets = Secrets(a=True, r=True, theta=True)

        client = Client(pattern=pattern, input_state=states, secrets=secrets, rng=fx_rng)
        noise_model = DepolarisingNoiseModel(
            measure_error_prob=1,
            entanglement_error_prob=1,
            x_error_prob=1,
            z_error_prob=1,
            measure_channel_prob=1,
        )

        for test_run in client.test_runs:
            backend = DensityMatrixBackend()
            client.refresh_randomness(rng=fx_rng)
            trap_outcomes = test_run.delegate(backend=backend, noise_model=noise_model, rng=fx_rng).trap_outcomes
            assert sum(trap_outcomes.values()) > 0


def find_correct_value(circuit_name: str) -> Outcome:
    with Path("tests/test_circuits/table.json").open() as f:
        table = json.load(f)
        # return 1 if yes instance
        # return 0 else (no instance, as circuits are already filtered)
        # print(table[circuit_name])
        return round(table[circuit_name])
