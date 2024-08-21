import random

import graphix.gflow
import graphix.pauli
import graphix.visualization
import networkx as nx
import numpy as np
from graphix.random_objects import rand_circuit
from graphix.sim.statevec import StatevectorBackend
from graphix.states import BasicStates

from veriphix.client import Client, Secrets, Stabilizer


class TestVBQC:
    def test_trap_delegated(self, fx_rng: np.random.Generator):
        nqubits = 2
        depth = 2
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        states = [BasicStates.PLUS for _ in pattern.input_nodes]
        secrets = Secrets(r=True, a=True, theta=True)
        client = Client(pattern=pattern, input_state=states, secrets=secrets)
        test_runs, _ = client.create_test_runs()
        for run in test_runs:
            backend = StatevectorBackend()
            trap_outcomes = client.delegate_test_run(backend=backend, run=run)
            assert trap_outcomes == [0 for _ in run.traps_list]

    def test_stabilizer(self, fx_rng: np.random.Generator):
        nqubits = 2
        depth = 2
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        nodes, edges = pattern.get_graph()[0], pattern.get_graph()[1]
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        k = random.randint(0, len(graph.nodes) - 1)
        nodes_sample = random.sample(list(graph.nodes), k)
        # nodes_sample=[0]

        stabilizer = Stabilizer(graph=graph, nodes=nodes_sample)

        expected_stabilizer = [graphix.pauli.I for _ in graph.nodes]
        for node in nodes_sample:
            expected_stabilizer[node] @= graphix.pauli.X
            for n in graph.neighbors(node):
                expected_stabilizer[n] @= graphix.pauli.Z

        assert expected_stabilizer == stabilizer.chain
