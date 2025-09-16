import random

import numpy as np
from graphix.random_objects import rand_circuit
from graphix.sim.statevec import StatevectorBackend

from veriphix.client import Client, Secrets
from veriphix.verifying import TestRun


class TestVerifying:
    def test_delegate_test(self, fx_rng: np.random.Generator):
        nqubits = 3
        depth = 5
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        secrets = Secrets(r=True, a=True, theta=True)
        client = Client(pattern=pattern, secrets=secrets)

        for _ in range(10):
            # Test noiseless trap delegation
            trap_size = random.choice(range(len(client.nodes)))
            random_nodes = random.sample(client.nodes, k=trap_size)

            random_multi_qubit_trap = tuple(random_nodes)
            # Only one trap
            traps = (random_multi_qubit_trap,)

            test_run = TestRun(client=client, traps=traps)
            backend = StatevectorBackend()
            outcomes = test_run.delegate(backend=backend).trap_outcomes

            for trap in traps:
                assert outcomes[trap] == 0
