import unittest

import graphix.command
import numpy as np
import pytest
import stim
from graphix.fundamentals import IXYZ
from graphix.pauli import Pauli
from graphix.random_objects import rand_circuit
from graphix.sim.statevec import StatevectorBackend
from graphix.states import BasicStates
from numpy.random import Generator
from stim import PauliString

from veriphix.client import CircuitUtils, Client, ClientMeasureMethod, Secrets
from veriphix.run import ComputationRun


class TestClient:
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
        secrets = Secrets(r=True, a=True, theta=True)
        client = Client(pattern=pattern, secrets=secrets)

        with pytest.raises(ValueError):
            client.create_test_runs(manual_colouring=(set([0]), set()))

    def test_create_test_run_manual_fail_improper(self, fx_rng):
        """testing manual colouring not proper"""

        # generate random circuit
        nqubits = 2
        depth = 1
        circuit = rand_circuit(nqubits, depth, fx_rng)
        # transpile to pattern
        pattern = circuit.transpile().pattern
        pattern.standardize()

        nodes, edges = pattern.get_graph()

        # initialise client
        secrets = Secrets(r=True, a=True, theta=True)
        client = Client(pattern=pattern, secrets=secrets)

        with pytest.raises(ValueError):  # trivially duplicate a node
            client.create_test_runs(manual_colouring=(set(nodes), set([nodes[0]])))

    def test_standardize(self, fx_rng: Generator):
        """
        Test to check that the Client-Server delegation works with standardized patterns
        """
        nqubits = 2
        depth = 2
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        for o in pattern.output_nodes:
            pattern.add(graphix.command.M(node=o))

        states = [BasicStates.PLUS for _ in pattern.input_nodes]

        secrets = Secrets(a=True, r=True, theta=True)

        client = Client(pattern=pattern, input_state=states, secrets=secrets)
        ComputationRun(client).delegate(backend=StatevectorBackend())
        # No assertion needed

    def test_minimize_space(self, fx_rng: Generator):
        """
        Test to check that the Client-Server delegation works with patterns re-organized with minimize-space
        """
        nqubits = 3
        depth = 5
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        pattern.minimize_space()
        for o in pattern.output_nodes:
            pattern.add(graphix.command.M(node=o))

        states = [BasicStates.PLUS for _ in pattern.input_nodes]

        secrets = Secrets(a=True, r=True, theta=True)

        client = Client(pattern=pattern, input_state=states, secrets=secrets)
        ComputationRun(client).delegate(backend=StatevectorBackend())
        # No assertion needed

    def test_client_input(self, fx_rng: Generator):
        # Generate random pattern
        nqubits = 2
        depth = 1
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        pattern.standardize()

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
            pattern.standardize()

            state = circuit.simulate_statevector().statevec

            backend = StatevectorBackend()
            # Initialize the client
            secrets = Secrets(r=True)
            # Giving it empty will create a random secret
            client = Client(pattern=pattern, secrets=secrets, classical_output=False)
            ComputationRun(client).delegate(backend)
            state_mbqc = backend.state
            np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.psi.flatten().conjugate(), state.psi.flatten())), 1)

    def test_theta_secret_simulation(self, fx_rng: Generator):
        # Generate random pattern
        nqubits = 2
        depth = 1
        for _i in range(10):
            circuit = rand_circuit(nqubits, depth, fx_rng)
            pattern = circuit.transpile().pattern
            pattern.standardize()

            secrets = Secrets(theta=True)

            # Create a |+> state for each input node
            states = [BasicStates.PLUS for node in pattern.input_nodes]

            # Create the client with the input state
            client = Client(pattern=pattern, input_state=states, secrets=secrets, classical_output=False)
            backend = StatevectorBackend()
            # Blinded simulation, between the client and the server
            ComputationRun(client).delegate(backend)
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
            pattern.standardize()

            secrets = Secrets(a=True)

            # Create a |+> state for each input node
            states = [BasicStates.PLUS for __ in pattern.input_nodes]

            # Create the client with the input state
            client = Client(pattern=pattern, input_state=states, secrets=secrets, classical_output=False)
            backend = StatevectorBackend()
            # Blinded simulation, between the client and the server
            ComputationRun(client).delegate(backend)
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
        pattern.standardize()
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
        ComputationRun(client).delegate(backend)

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
        pattern.standardize()
        secrets = Secrets(a=True, r=True, theta=True)

        # Create a |+> state for each input node, and associate index
        states = [BasicStates.PLUS for node in pattern.input_nodes]

        # Create the client with the input state
        client = Client(pattern=pattern, input_state=states, secrets=secrets)

        backend = StatevectorBackend()
        # Blinded simulation, between the client and the server
        client.prepare_states(backend, states_dict=client.computation_states)
        assert set(backend.node_index) == set(pattern.input_nodes)

    def test_UBQC(self, fx_rng: Generator):
        # Generate random pattern
        nqubits = 2
        # TODO : work on optimization of the quantum communication
        depth = 15
        for _ in range(10):
            circuit = rand_circuit(nqubits, depth, fx_rng)
            pattern = circuit.transpile().pattern
            # pattern.minimize_space()
            # pattern.standardize(method="global")

            secrets = Secrets(a=True, r=True, theta=True)

            # Create a |+> state for each input node, and associate index
            states = [BasicStates.PLUS for _ in pattern.input_nodes]

            # Create the client with the input state
            client = Client(pattern=pattern, input_state=states, secrets=secrets, classical_output=False)

            backend = StatevectorBackend()
            # Blinded simulation, between the client and the server
            # ComputationRun(client).delegate(backend)
            computation = ComputationRun(client=client)
            computation.delegate(backend=backend)
            blinded_simulation = backend.state

            # Clear simulation = no secret, just simulate the circuit defined above
            clear_simulation = circuit.simulate_statevector().statevec
            np.testing.assert_almost_equal(
                np.abs(np.dot(blinded_simulation.psi.flatten().conjugate(), clear_simulation.psi.flatten())), 1
            )

    def test_utils(self):
        n = 8
        rd_tableau = stim.Tableau.random(n)
        pattern = CircuitUtils.tableau_to_pattern(rd_tableau)

        input_string = rd_tableau.inverse()(stim.PauliString("X" * n))
        sign_error = input_string.sign.real == -1
        input_state = [Pauli(IXYZ(pauli)).eigenstate() for pauli in input_string]

        pattern.minimize_space()
        classical_output = pattern.output_nodes
        for onode in classical_output:
            pattern.add(graphix.command.M(node=onode))

        backend = StatevectorBackend()
        pattern.simulate_pattern(backend=backend, input_state=input_state)

        assert sum([pattern.results[i] for i in classical_output]) % 2 ^ sign_error == 0

    def test_delegate_pattern(self, fx_rng: Generator):
        nqubits = 5
        depth = 10
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        client = Client(pattern=pattern)

        comp_run = ComputationRun(client=client)
        backend = StatevectorBackend()
        outcomes = comp_run.delegate(backend=backend)
        assert outcomes is not None
        # TODO: assert something ? generate BQP computation for that

    def test_graph_clifford_structure(self, fx_rng: Generator):
        nqubits = 5
        depth = 10
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        client = Client(pattern=pattern)
        for node in client.graph.nodes:
            x_string = PauliString(["X" if i == node else "I" for i in client.graph.nodes])
            conjugated_string = client.clifford_structure.inverse()(x_string)
            neighbors = set(client.graph.neighbors(node))
            expected_conjugated_string = PauliString(
                ["X" if i == node else "Z" if i in neighbors else "I" for i in client.graph.nodes]
            )
            assert conjugated_string == expected_conjugated_string


if __name__ == "__main__":
    unittest.main()
