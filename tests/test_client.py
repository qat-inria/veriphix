import unittest

import numpy as np
from graphix.random_objects import rand_circuit
from graphix.sim.statevec import StatevectorBackend
from graphix.states import BasicStates
from numpy.random import Generator

from veriphix.client import Client, ClientMeasureMethod, Secrets


class TestClient:
    def test_client_input(self, fx_rng: Generator):
        # Generate random pattern
        nqubits = 2
        depth = 1
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        pattern.standardize(method="global")

        secrets = Secrets(theta=True)

        # Create a |+> state for each input node
        states = [BasicStates.PLUS for node in pattern.input_nodes]

        # Create the client with the input state
        _client = Client(pattern=pattern, input_state=states, secrets=secrets)

        # Assert something...
        # Todo ?

    def test_r_secret_simulation(self, fx_rng: Generator):
        # Generate and standardize pattern
        nqubits = 2
        depth = 1
        for _i in range(10):
            circuit = rand_circuit(nqubits, depth, fx_rng)
            pattern = circuit.transpile().pattern
            pattern.standardize(method="global")

            state = circuit.simulate_statevector().statevec

            backend = StatevectorBackend()
            # Initialize the client
            secrets = Secrets(r=True)
            # Giving it empty will create a random secret
            client = Client(pattern=pattern, secrets=secrets)
            client.delegate_pattern(backend)
            state_mbqc = backend.state
            np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.psi.flatten().conjugate(), state.psi.flatten())), 1)

    def test_theta_secret_simulation(self, fx_rng: Generator):
        # Generate random pattern
        nqubits = 2
        depth = 1
        for _i in range(10):
            circuit = rand_circuit(nqubits, depth, fx_rng)
            pattern = circuit.transpile().pattern
            pattern.standardize(method="global")

            secrets = Secrets(theta=True)

            # Create a |+> state for each input node
            states = [BasicStates.PLUS for node in pattern.input_nodes]

            # Create the client with the input state
            client = Client(pattern=pattern, input_state=states, secrets=secrets)
            backend = StatevectorBackend()
            # Blinded simulation, between the client and the server
            client.delegate_pattern(backend)
            blinded_simulation = backend.state

            # Clear simulation = no secret, just simulate the circuit defined above
            clear_simulation = circuit.simulate_statevector().statevec

            np.testing.assert_almost_equal(
                np.abs(np.dot(blinded_simulation.psi.flatten().conjugate(), clear_simulation.psi.flatten())), 1
            )

    def test_a_secret_simulation(self, fx_rng: Generator):
        # Generate random pattern
        nqubits = 2
        depth = 1
        for _ in range(10):
            circuit = rand_circuit(nqubits, depth, fx_rng)
            pattern = circuit.transpile().pattern
            pattern.standardize(method="global")

            secrets = Secrets(a=True)

            # Create a |+> state for each input node
            states = [BasicStates.PLUS for __ in pattern.input_nodes]

            # Create the client with the input state
            client = Client(pattern=pattern, input_state=states, secrets=secrets)
            backend = StatevectorBackend()
            # Blinded simulation, between the client and the server
            client.delegate_pattern(backend)
            blinded_simulation = backend.state

            # Clear simulation = no secret, just simulate the circuit defined above
            clear_simulation = circuit.simulate_statevector().statevec
            np.testing.assert_almost_equal(
                np.abs(np.dot(blinded_simulation.psi.flatten().conjugate(), clear_simulation.psi.flatten())), 1
            )

    def test_r_secret_results(self, fx_rng: Generator):
        # Generate and standardize pattern
        nqubits = 2
        depth = 1
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        pattern.standardize(method="global")
        server_results = dict()

        class CacheMeasureMethod(ClientMeasureMethod):
            def set_measure_result(self, node: int, result: bool) -> None:
                nonlocal server_results
                server_results[node] = result
                super().set_measure_result(node, result)

        # Initialize the client
        secrets = Secrets(r=True)
        # Giving it empty will create a random secret
        client = Client(pattern=pattern, measure_method_cls=CacheMeasureMethod, secrets=secrets)
        backend = StatevectorBackend()
        client.delegate_pattern(backend)

        for measured_node in client.measurement_db:
            # Compare results on the client side and on the server side : should differ by r[node]
            result = client.results[measured_node]
            client_r_secret = client.secret_datas.r[measured_node]
            server_result = server_results[measured_node]
            assert result == (server_result + client_r_secret) % 2

    def test_qubits_preparation(self, fx_rng: Generator):
        nqubits = 2
        depth = 1
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        nodes = pattern.get_graph()[0]
        pattern.standardize(method="global")
        secrets = Secrets(a=True, r=True, theta=True)

        # Create a |+> state for each input node, and associate index
        states = [BasicStates.PLUS for node in pattern.input_nodes]

        # Create the client with the input state
        client = Client(pattern=pattern, input_state=states, secrets=secrets)

        backend = StatevectorBackend()
        # Blinded simulation, between the client and the server
        client.prepare_states(backend)
        assert set(backend.node_index) == set(nodes)
        client.blind_qubits(backend)
        assert set(backend.node_index) == set(nodes)

    def test_UBQC(self, fx_rng: Generator):
        # Generate random pattern
        nqubits = 2
        # TODO : work on optimization of the quantum communication
        depth = 1
        for _i in range(10):
            circuit = rand_circuit(nqubits, depth, fx_rng)
            pattern = circuit.transpile().pattern
            pattern.standardize(method="global")

            secrets = Secrets(a=True, r=True, theta=True)

            # Create a |+> state for each input node, and associate index
            states = [BasicStates.PLUS for _ in pattern.input_nodes]

            # Create the client with the input state
            client = Client(pattern=pattern, input_state=states, secrets=secrets)

            backend = StatevectorBackend()
            # Blinded simulation, between the client and the server
            client.delegate_pattern(backend)
            blinded_simulation = backend.state
            # Clear simulation = no secret, just simulate the circuit defined above
            clear_simulation = circuit.simulate_statevector().statevec
            np.testing.assert_almost_equal(
                np.abs(np.dot(blinded_simulation.psi.flatten().conjugate(), clear_simulation.psi.flatten())), 1
            )


if __name__ == "__main__":
    unittest.main()
