import graphix.command
import numpy as np
from numpy.random import Generator
from graphix.noise_models import DepolarisingNoiseModel
from graphix.random_objects import Circuit, rand_circuit
from graphix.sim.density_matrix import DensityMatrixBackend
from graphix.sim.statevec import StatevectorBackend
from graphix.states import BasicStates

from veriphix.client import Client, Secrets
from veriphix.trappifiedCanvas import TrappifiedCanvas


class TestVBQC:
    def test_trap_delegated(self, fx_rng: np.random.Generator):
        nqubits = 2
        depth = 2
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        # pattern.standardize()

        # don't forget to add in the output nodes that are not initially measured!
        for onode in pattern.output_nodes:
            pattern.add(graphix.command.M(node=onode))

        states = [BasicStates.PLUS for _ in pattern.input_nodes]
        secrets = Secrets(r=True, a=True, theta=True)
        client = Client(pattern=pattern, input_state=states, secrets=secrets)
        test_stabs = client.create_test_runs()
        for stab in test_stabs:
            backend = StatevectorBackend()
            canvas = TrappifiedCanvas(stab)
            trap_outcomes = client.delegate_test_run(backend=backend, run=canvas)
            assert trap_outcomes == [0 for _ in stab.traps_list]

    def test_noiseless(self, fx_rng: Generator):
        circuit = Circuit(3)
        circuit.rz(1, np.pi / 4)
        circuit.cnot(0, 2)
        pattern = circuit.transpile().pattern

        print(f"{pattern.n_node} nodes")
        # pattern.standardize()
        for o in pattern.output_nodes:
            pattern.add(graphix.command.M(node=o))

        states = [BasicStates.PLUS for _ in pattern.input_nodes]

        secrets = Secrets(a=True, r=True, theta=True)

        client = Client(pattern=pattern, input_state=states, secrets=secrets)
        test_stabs = client.create_test_runs()
        noise_model = DepolarisingNoiseModel(
            measure_error_prob=0, entanglement_error_prob=0, x_error_prob=0, z_error_prob=0, measure_channel_prob=0
        )
        for stab in test_stabs:
            backend = DensityMatrixBackend(rng=fx_rng)
            client.refresh_randomness()
            canvas = TrappifiedCanvas(stab, rng=fx_rng)
            trap_outcomes = client.delegate_test_run(backend=backend, run=canvas, noise_model=noise_model)
            assert sum(trap_outcomes) == 0

    def test_noisy(self, fx_rng: Generator):
        circuit = Circuit(3)
        circuit.rz(1, np.pi / 4)
        circuit.cnot(0, 2)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        for o in pattern.output_nodes:
            pattern.add(graphix.command.M(node=o))

        states = [BasicStates.PLUS for _ in pattern.input_nodes]

        secrets = Secrets(a=True, r=True, theta=True)

        client = Client(pattern=pattern, input_state=states, secrets=secrets)
        test_stabs = client.create_test_runs()
        noise_model = DepolarisingNoiseModel(
            measure_error_prob=1, entanglement_error_prob=1, x_error_prob=1, z_error_prob=1, measure_channel_prob=1
        )
        total_trap_failures = 0
        for stab in test_stabs:
            backend = DensityMatrixBackend(rng=fx_rng)
            client.refresh_randomness()
            canvas = TrappifiedCanvas(stab, rng=fx_rng)
            trap_outcomes = client.delegate_test_run(backend=backend, run=canvas, noise_model=noise_model)
            total_trap_failures += sum(trap_outcomes)
        assert total_trap_failures > 0
