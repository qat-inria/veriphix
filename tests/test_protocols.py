import random

import numpy as np
from graphix.random_objects import rand_circuit
from graphix.sim.statevec import StatevectorBackend
from itertools import combinations

from veriphix.client import Client, Secrets
from veriphix.verifying import TestRun
from veriphix.protocols import FK12, RandomTraps, Dummyless, VerificationProtocol
from stim import PauliString

import pytest

class TestProtocols:

    @pytest.mark.parametrize("protocol_class", (FK12, RandomTraps))
    def test_noiseless_all_protocols(self, fx_rng: np.random.Generator, protocol_class:type[VerificationProtocol]):
        nqubits = 3
        depth = 5
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        secrets = Secrets(r=True, a=True, theta=True)
        client = Client(pattern=pattern, secrets=secrets, protocol_cls=protocol_class)
        canvas = client.sample_canvas()
        run_results = client.delegate_canvas(canvas=canvas, backend_cls=StatevectorBackend)
        decision, outcome, result_analysis = client.analyze_outcomes(canvas=canvas, outcomes=run_results)
        assert decision == True
        assert result_analysis.nr_failed_test_rounds == 0


    def test_FK(self, fx_rng: np.random.Generator):
        nqubits = 3
        depth = 5
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        secrets = Secrets(r=True, a=True, theta=True)
        client = Client(pattern=pattern, secrets=secrets, protocol_cls=FK12)


        # TODO: assert nice coloring
        assert client.test_runs != []

    def test_random_traps(self, fx_rng: np.random.Generator):
        """
        Nothing is done more than in 'test_noiseless_all_protocols'
        """
        nqubits = 3
        depth = 5
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        secrets = Secrets(r=True, a=True, theta=True)
        client = Client(pattern=pattern, secrets=secrets, protocol_cls=RandomTraps)
        canvas = client.sample_canvas()
        run_results = client.delegate_canvas(canvas=canvas, backend_cls=StatevectorBackend)
        decision, outcome, result_analysis = client.analyze_outcomes(canvas=canvas, outcomes=run_results)
        assert decision == True
        assert result_analysis.nr_failed_test_rounds == 0

    def test_dummyless(self, fx_rng: np.random.Generator):
        nqubits = 3
        depth = 5
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        secrets = Secrets(r=True, a=True, theta=True)
        client = Client(pattern=pattern, secrets=secrets, protocol_cls=Dummyless)
