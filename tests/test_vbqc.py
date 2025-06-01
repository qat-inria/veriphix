import graphix.command
import numpy as np
from numpy.random import Generator
from graphix.noise_models import DepolarisingNoiseModel
from graphix.random_objects import Circuit, rand_circuit
from graphix.sim.density_matrix import DensityMatrixBackend
from graphix.sim.statevec import StatevectorBackend
from graphix.states import BasicStates

from veriphix.client import Client, Secrets


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
        test_runs = client.create_test_runs()
        for test_run in test_runs:
            backend = StatevectorBackend()
            trap_outcomes = test_run.delegate(backend=backend)
            assert sum(trap_outcomes.values())==0



    def test_noiseless(self, fx_rng: Generator):
        nqubits = 3
        depth = 3
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        # pattern.standardize()
        for o in pattern.output_nodes:
            pattern.add(graphix.command.M(node=o))

        states = [BasicStates.PLUS for _ in pattern.input_nodes]

        secrets = Secrets(a=True, r=True, theta=True)

        client = Client(pattern=pattern, input_state=states, secrets=secrets)
        test_runs = client.create_test_runs()
        noise_model = DepolarisingNoiseModel(
            measure_error_prob=0, entanglement_error_prob=0, x_error_prob=0, z_error_prob=0, measure_channel_prob=0
        )
        for test_run in test_runs:
            client.refresh_randomness()
            backend = DensityMatrixBackend(rng=fx_rng)
            trap_outcomes = test_run.delegate(backend=backend, noise_model=noise_model)
            assert sum(trap_outcomes.values()) == 0

    def test_noisy(self, fx_rng: Generator):
        nqubits = 3
        depth = 3
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        for o in pattern.output_nodes:
            pattern.add(graphix.command.M(node=o))

        states = [BasicStates.PLUS for _ in pattern.input_nodes]

        secrets = Secrets(a=True, r=True, theta=True)

        client = Client(pattern=pattern, input_state=states, secrets=secrets)
        test_runs = client.create_test_runs()
        noise_model = DepolarisingNoiseModel(
            measure_error_prob=1, entanglement_error_prob=1, x_error_prob=1, z_error_prob=1, measure_channel_prob=1
        )
        for test_run in test_runs:
            backend = DensityMatrixBackend(rng=fx_rng)
            client.refresh_randomness()
            trap_outcomes = test_run.delegate(backend=backend, noise_model=noise_model)
            assert sum(trap_outcomes.values()) > 0
