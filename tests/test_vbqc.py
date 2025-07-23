import graphix.command
from graphix.pattern import Pattern

import numpy as np
from numpy.random import Generator
from graphix.noise_models import DepolarisingNoiseModel
from graphix.random_objects import Circuit, rand_circuit
from graphix.sim.density_matrix import DensityMatrixBackend
from graphix.sim.statevec import StatevectorBackend
from graphix.states import BasicStates
from pathlib import Path

from veriphix.qasm_parser import read_qasm
import veriphix.brickwork_state_transpiler

from veriphix.client import Client, Secrets, TrappifiedSchemeParameters
from veriphix.run import TestRun, ComputationRun
import json
import random
import pytest


def load_pattern_from_circuit(circuit_label: str) -> tuple[Pattern, list[int]]:
    with Path(f"tests/circuits/{circuit_label}").open() as f:
        circuit = read_qasm(f)
        pattern = veriphix.brickwork_state_transpiler.transpile(circuit)

        
        pattern.minimize_space()
    return pattern


class TestVBQC:
    def test_trap_delegated(self, fx_rng: np.random.Generator):
        nqubits = 3
        depth = 5
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        # pattern.standardize()
        # don't forget to add in the output nodes that are not initially measured!
        for onode in pattern.output_nodes:
            pattern.add(graphix.command.M(node=onode))

        states = [BasicStates.PLUS for _ in pattern.input_nodes]
        secrets = Secrets(r=True, a=True, theta=True)
        client = Client(pattern=pattern, input_state=states, secrets=secrets)
        for test_run in client.test_runs:
            backend = StatevectorBackend()
            trap_outcomes = test_run.delegate(backend=backend)
            assert sum(trap_outcomes.values())==0


    def test_sample_canvas(self, fx_rng: Generator):
        nqubits = 3
        depth = 5
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        states = [BasicStates.PLUS for _ in pattern.input_nodes]
        secrets = Secrets(r=True, a=True, theta=True)
        client = Client(pattern=pattern, input_state=states, secrets=secrets)

        canvas = client.sample_canvas()
        # Just tests that it runs


    def test_delegate_canvas(self, fx_rng: Generator):
        nqubits = 3
        depth = 5
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        svbackend = StatevectorBackend()
        simulated_pattern_output = pattern.simulate_pattern(backend=svbackend)
        simulated_circuit_output = circuit.simulate_statevector().statevec

        states = [BasicStates.PLUS for _ in pattern.input_nodes]
        secrets = Secrets(r=True, a=True, theta=True)
        
        parameters = TrappifiedSchemeParameters(comp_rounds=10, test_rounds=10, threshold=0)
        client = Client(pattern=pattern, input_state=states, secrets=secrets, parameters=parameters, classical_output=False)

        backend = StatevectorBackend()

        canvas = client.sample_canvas()
        outcomes = client.delegate_canvas(canvas=canvas, backend=backend)
        for round in canvas:
            if isinstance(canvas[round], ComputationRun):
                np.testing.assert_almost_equal(
                    np.abs(np.dot(outcomes[round].psi.flatten().conjugate(), simulated_pattern_output.psi.flatten())), 1
                )
                np.testing.assert_almost_equal(
                    np.abs(np.dot(outcomes[round].psi.flatten().conjugate(), simulated_circuit_output.psi.flatten())), 1
                )
        # Just tests that it runs
        """
        TODO, in the tests:
        - Noiseless, quantum outputs: check evolution of the state for all the comp. runs, and check for no trap failures
        """

    def test_analyze_outcomes(self, fx_rng: Generator):
        nqubits = 3
        depth = 3
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        states = [BasicStates.PLUS for _ in pattern.input_nodes]
        secrets = Secrets(r=True, a=True, theta=True)
        
        parameters = TrappifiedSchemeParameters(comp_rounds=50, test_rounds=50, threshold=10)
        client = Client(pattern=pattern, input_state=states, secrets=secrets, parameters=parameters)

        backend = StatevectorBackend()

        canvas = client.sample_canvas()
        outcomes = client.delegate_canvas(canvas=canvas, backend=backend)

        # only for BQP
        decision, result = client.analyze_outcomes(canvas, outcomes)



    @pytest.mark.parametrize('blind', (False, True))
    def test_BQP_circuit(self, fx_rng: Generator, blind:bool):
        bqp_error = 0.01
        with Path("tests/circuits/table.json").open() as f:
            table = json.load(f)
            circuits = [name for name, prob in table.items() if prob < bqp_error or prob > 1 - bqp_error]
        random_circuit_label = random.choice(circuits)
        # Example of deterministic circuit with output 0
        random_circuit_label = "circuit677.qasm"
        pattern = load_pattern_from_circuit(circuit_label=random_circuit_label)

        states = [BasicStates.PLUS for _ in pattern.input_nodes]
        secrets = Secrets(r=blind, a=blind, theta=blind)
        
        parameters = TrappifiedSchemeParameters(comp_rounds=10, test_rounds=10, threshold=5)
        # QCircuit, we keep the first output only
        desired_outputs=[0]
        client = Client(pattern=pattern, input_state=states, secrets=secrets, parameters=parameters, desired_outputs=desired_outputs)
        backend = StatevectorBackend()

        canvas = client.sample_canvas()
        outcomes = client.delegate_canvas(canvas=canvas, backend=backend)
        decision, result = client.analyze_outcomes(canvas, outcomes)
        assert decision == True
        assert result != "Abort"
        assert int(result) == find_correct_value(random_circuit_label)

    @pytest.mark.parametrize('blind', (False, True))
    def test_noiseless(self, fx_rng: Generator, blind:bool):
        nqubits = 3
        depth = 3
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        states = [BasicStates.PLUS for _ in pattern.input_nodes]

        secrets = Secrets(a=blind, r=blind, theta=blind)

        client = Client(pattern=pattern, input_state=states, secrets=secrets)
        noise_model = DepolarisingNoiseModel(
            measure_error_prob=0, entanglement_error_prob=0, x_error_prob=0, z_error_prob=0, measure_channel_prob=0
        )
        for test_run in client.test_runs:
            client.refresh_randomness()
            backend = DensityMatrixBackend(rng=fx_rng)
            trap_outcomes = test_run.delegate(backend=backend, noise_model=noise_model)
            assert sum(trap_outcomes.values()) == 0

    def test_noisy(self, fx_rng: Generator):
        nqubits = 3
        depth = 3
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern


        states = [BasicStates.PLUS for _ in pattern.input_nodes]

        secrets = Secrets(a=True, r=True, theta=True)

        client = Client(pattern=pattern, input_state=states, secrets=secrets)
        noise_model = DepolarisingNoiseModel(
            measure_error_prob=1, entanglement_error_prob=1, x_error_prob=1, z_error_prob=1, measure_channel_prob=1
        )
        for test_run in client.test_runs:
            backend = DensityMatrixBackend(rng=fx_rng)
            client.refresh_randomness()
            trap_outcomes = test_run.delegate(backend=backend, noise_model=noise_model)
            assert sum(trap_outcomes.values()) > 0


def find_correct_value(circuit_name):
    with Path("tests/circuits/table.json").open() as f:
        table = json.load(f)
        # return 1 if yes instance
        # return 0 else (no instance, as circuits are already filtered)
        # print(table[circuit_name])
        return round(table[circuit_name])