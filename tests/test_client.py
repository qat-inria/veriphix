import unittest

import networkx as nx
import numpy as np
import pytest
from graphix import Measurement, OpenGraph
from graphix.measurements import Outcome
from graphix.random_objects import rand_circuit
from graphix.sim.statevec import StatevectorBackend
from graphix.states import BasicStates
from numpy.random import Generator
from stim import PauliString
from typing_extensions import override

from veriphix.blinding import Secrets
from veriphix.client import Client, ClientMeasureMethod
from veriphix.verifying import ComputationRun


class TestClient:
    def test_standardize(self, fx_rng: Generator) -> None:
        """
        Test to check that the Client-Server delegation works with standardized patterns
        """
        nqubits = 2
        depth = 2
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        pattern.standardize()

        states = [BasicStates.PLUS for _ in pattern.input_nodes]

        secrets = Secrets(a=True, r=True, theta=True)

        client = Client(pattern=pattern, input_state=states, secrets=secrets, classical_output=True, rng=fx_rng)
        ComputationRun(client).delegate(backend=StatevectorBackend(), rng=fx_rng)
        # No assertion needed

    def test_minimize_space(self, fx_rng: Generator) -> None:
        """
        Test to check that the Client-Server delegation works with patterns re-organized with minimize-space
        """
        nqubits = 3
        depth = 5
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        pattern.minimize_space()

        states = [BasicStates.PLUS for _ in pattern.input_nodes]

        secrets = Secrets(a=True, r=True, theta=True)

        client = Client(pattern=pattern, input_state=states, secrets=secrets, classical_output=True, rng=fx_rng)
        ComputationRun(client).delegate(backend=StatevectorBackend(), rng=fx_rng)
        # No assertion needed

    def test_client_input(self, fx_rng: Generator) -> None:
        """test that the Client can input a custom quantum state."""
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
        _client = Client(pattern=pattern, input_state=states, secrets=secrets, rng=fx_rng)

        # Assert something...
        # Todo ?

    def test_r_secret_simulation(self, fx_rng: Generator) -> None:
        """Test for equal output state when Client blinds the computation with only a 'r' secret"""
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
            secrets = Secrets(r=True, a=False, theta=False)
            # Giving it empty will create a random secret
            client = Client(pattern=pattern, secrets=secrets, classical_output=False, rng=fx_rng)
            ComputationRun(client).delegate(backend, rng=fx_rng)
            state_mbqc = backend.state
            np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.psi.flatten().conjugate(), state.psi.flatten())), 1)

    def test_theta_secret_simulation(self, fx_rng: Generator) -> None:
        """Test for equal output state when Client blinds the computation with only a 'theta' secret"""
        # Generate random pattern
        nqubits = 2
        depth = 1
        for _i in range(10):
            circuit = rand_circuit(nqubits, depth, fx_rng)
            pattern = circuit.transpile().pattern
            pattern.standardize()

            secrets = Secrets(theta=True, a=False, r=False)

            # Create a |+> state for each input node
            states = [BasicStates.PLUS for node in pattern.input_nodes]

            # Create the client with the input state
            client = Client(pattern=pattern, input_state=states, secrets=secrets, classical_output=False, rng=fx_rng)
            backend = StatevectorBackend()
            # Blinded simulation, between the client and the server
            ComputationRun(client).delegate(backend, rng=fx_rng)
            blinded_simulation = backend.state

            # Clear simulation = no secret, just simulate the circuit defined above
            clear_simulation = circuit.simulate_statevector().statevec

            np.testing.assert_almost_equal(
                np.abs(np.dot(blinded_simulation.psi.flatten().conjugate(), clear_simulation.psi.flatten())), 1
            )

    def test_a_secret_simulation(self, fx_rng: Generator) -> None:
        """Test for equal output state when Client blinds the computation with only a 'a' secret"""
        # Generate random pattern
        nqubits = 2
        depth = 1
        for _ in range(10):
            circuit = rand_circuit(nqubits, depth, fx_rng)
            pattern = circuit.transpile().pattern
            pattern.standardize()

            secrets = Secrets(a=True, r=False, theta=False)

            # Create a |+> state for each input node
            states = [BasicStates.PLUS for __ in pattern.input_nodes]

            # Create the client with the input state
            client = Client(pattern=pattern, input_state=states, secrets=secrets, classical_output=False, rng=fx_rng)
            backend = StatevectorBackend()
            # Blinded simulation, between the client and the server
            ComputationRun(client).delegate(backend, rng=fx_rng)
            blinded_simulation = backend.state

            # Clear simulation = no secret, just simulate the circuit defined above
            clear_simulation = circuit.simulate_statevector().statevec
            np.testing.assert_almost_equal(
                np.abs(np.dot(blinded_simulation.psi.flatten().conjugate(), clear_simulation.psi.flatten())), 1
            )

    def test_r_secret_results(self, fx_rng: Generator) -> None:
        """Tests that when the Client has a 'r' secret, the measurement outcomes returned by the Server are indeed XORed by 'r' before"""
        # Generate and standardize pattern
        nqubits = 2
        depth = 1
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        server_results = dict()

        class CacheMeasureMethod(ClientMeasureMethod):
            @override
            def store_measurement_outcome(self, node: int, result: Outcome) -> None:
                nonlocal server_results
                server_results[node] = result
                super().store_measurement_outcome(node, result)

        # Initialize the client
        secrets = Secrets(r=True, a=False)
        # Giving it empty will create a random secret
        client = Client(pattern=pattern, measure_method_cls=CacheMeasureMethod, secrets=secrets, rng=fx_rng)
        backend = StatevectorBackend()
        ComputationRun(client).delegate(backend, rng=fx_rng)

        for measured_node in client.measurement_db:
            # Compare results on the client side and on the server side : should differ by r[node]
            result = client.results[measured_node]
            client_flip_value = client.secret_datas.r[measured_node] ^ client.secret_datas.a.a_N.get(measured_node, 0)
            server_result = server_results[measured_node]
            assert result == (server_result + client_flip_value) % 2

    def test_qubits_preparation(self, fx_rng: Generator) -> None:
        nqubits = 2
        depth = 1
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        secrets = Secrets(a=True, r=True, theta=True)

        # Create a |+> state for each input node, and associate index
        states = [BasicStates.PLUS for node in pattern.input_nodes]

        # Create the client with the input state
        client = Client(pattern=pattern, input_state=states, secrets=secrets, rng=fx_rng)

        backend = StatevectorBackend()
        # Blinded simulation, between the client and the server
        client.prepare_states(backend, states_dict=client.computation_states)
        assert set(backend.node_index) == set(pattern.input_nodes)

    def test_UBQC(self, fx_rng: Generator) -> None:
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
            client = Client(pattern=pattern, input_state=states, secrets=secrets, classical_output=False, rng=fx_rng)

            backend = StatevectorBackend()
            # Blinded simulation, between the client and the server
            # ComputationRun(client).delegate(backend)
            computation = ComputationRun(client=client)
            computation.delegate(backend=backend, rng=fx_rng)
            blinded_simulation = backend.state

            # Clear simulation = no secret, just simulate the circuit defined above
            clear_simulation = circuit.simulate_statevector().statevec
            np.testing.assert_almost_equal(
                np.abs(np.dot(blinded_simulation.psi.flatten().conjugate(), clear_simulation.psi.flatten())), 1
            )

    def test_delegate_pattern(self, fx_rng: Generator) -> None:
        nqubits = 5
        depth = 10
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        client = Client(pattern=pattern, rng=fx_rng)

        comp_run = ComputationRun(client=client)
        backend = StatevectorBackend()
        outcomes = comp_run.delegate(backend=backend, rng=fx_rng)
        assert outcomes is not None
        # TODO: assert something ? generate BQP computation for that

    def test_graph_clifford_structure(self, fx_rng: Generator) -> None:
        nqubits = 5
        depth = 10
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        client = Client(pattern=pattern, rng=fx_rng)
        node_upper_bound = max(client.graph.nodes) + 1
        for node in client.graph.nodes:
            x_string = PauliString(node_upper_bound)
            x_string[node] = "X"
            conjugated_string = client.clifford_structure.inverse()(x_string)
            expected_conjugated_string = PauliString(node_upper_bound)
            expected_conjugated_string[node] = "X"
            for i in client.graph.neighbors(node):
                expected_conjugated_string[i] = "Z"
            assert conjugated_string == expected_conjugated_string

    def test_reorder_output_nodes(self, fx_rng: Generator) -> None:
        # Verify that the delegate simulation respects the
        # non-standard ordering of output nodes (i.e., [2, 1] instead
        # of the usual [1, 2]).
        og = OpenGraph(graph=nx.path_graph(3), input_nodes=[], output_nodes=[2, 1], measurements={0: Measurement.X})
        pattern = og.to_pattern()
        state_ref = pattern.simulate_pattern()
        backend = StatevectorBackend()
        secrets = Secrets()
        client = Client(pattern=pattern, secrets=secrets, classical_output=False, rng=fx_rng)
        ComputationRun(client).delegate(backend, rng=fx_rng)
        state_veriphix = backend.state
        assert state_veriphix.isclose(state_ref)

    def test_refresh_computation_states(self, fx_rng: Generator) -> None:
        # Verify that the "computation states" are regenerated when
        # `refresh_randomness` is called.
        # In particular, if the initial secret and the refreshed one
        # do not flip the input state in the same way, the computation
        # states must be updated after the secret refresh.
        og = OpenGraph(
            graph=nx.Graph([(0, 1)]), input_nodes=[0], output_nodes=[1], measurements={0: Measurement.XY(0.75)}
        )
        pattern = og.to_pattern()
        state_ref = pattern.simulate_pattern()
        backend = StatevectorBackend()
        secrets = Secrets(a=True)
        fixed_rng = np.random.default_rng(3)
        client = Client(pattern=pattern, secrets=secrets, classical_output=False, rng=fixed_rng)
        old_a = client.secret_datas.a.a.get(0, 0)
        ComputationRun(client).delegate(backend, rng=fixed_rng)
        new_a = client.secret_datas.a.a.get(0, 0)
        # fixed_rng is chosen to satisfy the following assertion.
        assert old_a != new_a
        state_veriphix = backend.state
        assert state_veriphix.isclose(state_ref)

    def test_reject_yz_measurement(self, fx_rng: Generator) -> None:
        og = OpenGraph(
            graph=nx.Graph([(0, 1)]), input_nodes=[], output_nodes=[1], measurements={0: Measurement.YZ(0.5)}
        )
        pattern = og.to_pattern()
        state_ref = pattern.simulate_pattern()
        backend = StatevectorBackend()
        secrets = Secrets()
        with pytest.raises(ValueError, match="UBQC only works for measurements in plane XY"):
            client = Client(pattern=pattern, secrets=secrets, classical_output=False, rng=fx_rng)
            ComputationRun(client).delegate(backend, rng=fx_rng)
            state_veriphix = backend.state
            assert state_veriphix.isclose(state_ref)


if __name__ == "__main__":
    unittest.main()
