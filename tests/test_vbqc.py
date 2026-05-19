from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from graphix.noise_models import DepolarisingNoiseModel, NoiseModel
from graphix.random_objects import rand_circuit
from graphix.sim.density_matrix import DensityMatrixBackend
from graphix.sim.statevec import StatevectorBackend
from graphix.states import BasicStates
from graphix_qasm_parser import OpenQASMParser

from veriphix.blinding import Secrets
from veriphix.client import Client
from veriphix.malicious_noise_model import MaliciousNoiseModel
from veriphix.protocols import FK12, RandomTraps
from veriphix.util_rounds import optimize_with_robustness_constraint
from veriphix.verifying import QuantumComputationResult, TrappifiedSchemeParameters

if TYPE_CHECKING:
    from graphix.measurements import Outcome
    from graphix.pattern import Pattern
    from numpy.random import Generator


def load_pattern_from_circuit(circuit_label: str) -> Pattern:
    parser = OpenQASMParser()
    circuit = parser.parse_file(Path("tests/test_circuits") / circuit_label)
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
            trap_outcomes = test_run.accept(client, backend, None, fx_rng).trap_outcomes
            assert sum(trap_outcomes.values()) == 0

    def test_sample_canvas(self, fx_rng: Generator) -> None:
        nqubits = 3
        depth = 5
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        client = Client(pattern=pattern, rng=fx_rng)
        canvas = client.sample_canvas(rng=fx_rng)
        assert canvas
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
        traps_decision, _computation_decision, _result_analysis = client.analyze_outcomes(canvas, outcomes)
        assert traps_decision

    @pytest.mark.parametrize("blind", (False, True))
    def test_BQP_circuit(self, fx_rng: Generator, blind: bool) -> None:
        with Path("tests/test_circuits/table.json").open() as f:
            table = json.load(f)
            circuits = [name for name, prob in table.items()]
        for circuit_label in circuits:
            pattern = load_pattern_from_circuit(circuit_label=circuit_label)

            secrets = Secrets(r=blind, a=blind, theta=blind)

            parameters = TrappifiedSchemeParameters(comp_rounds=20, test_rounds=20, threshold=5)
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
            backend = DensityMatrixBackend()
            trap_outcomes = test_run.accept(client, backend, noise_model, fx_rng).trap_outcomes
            assert sum(trap_outcomes.values()) == 0

    @pytest.mark.parametrize("noise_model_name", ["depolarising", "malicious"])
    def test_random_traps_performance(self, fx_rng: Generator, noise_model_name: str) -> None:
        """Check that RandomTraps detects deviations at the 1/2 rate expected by the literature [KKLM+22].

        RandomTraps samples a fresh random multi-qubit trap for each test round via
        sample_test_run, so no pre-computed test run list is needed. Under any deviation (here, depolarising/malicious noise), each
        trap fires independently with probability ~1/2, giving an expected detection
        rate of ≈1/2 over many rounds.

        The canvas is configured with test rounds only (0 computation rounds).
        The threshold is set to 0, arbitrarily.
        After delegation, the empirical detection rate must fall in [0.35, 0.65].
        """
        nqubits = 2
        depth = 2
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        secrets = Secrets(a=True, r=True, theta=True)
        d, s, w = 0, 30, 0
        parameters = TrappifiedSchemeParameters(comp_rounds=d, test_rounds=s, threshold=w)

        client = Client(
            pattern=pattern,
            secrets=secrets,
            protocol=RandomTraps(),
            parameters=parameters,
            rng=fx_rng,
        )

        if noise_model_name == "depolarising":
            noise_model = DepolarisingNoiseModel(
                measure_error_prob=1,
                entanglement_error_prob=1,
                x_error_prob=1,
                z_error_prob=1,
                measure_channel_prob=1,
            )
        elif noise_model_name == "malicious":
            subset_size = int(fx_rng.integers(1, len(client.nodes)))
            malicious_nodes = [int(node) for node in fx_rng.choice(client.nodes, size=subset_size, replace=False)]
            noise_model = MaliciousNoiseModel(nodes=malicious_nodes, prob=1, rng=fx_rng)
        else:
            raise ValueError(f"Unknown noise model: {noise_model_name}")

        canvas = client.sample_canvas(rng=fx_rng)
        outcomes = client.delegate_canvas(
            canvas=canvas, backend_cls=DensityMatrixBackend, noise_model=noise_model, rng=fx_rng
        )
        _, _, result_analysis = client.analyze_outcomes(canvas, outcomes)

        detection_rate = result_analysis.nr_failed_test_rounds / s
        assert 0.35 <= detection_rate <= 0.65, f"Expected ≈1/2 detection rate, got {detection_rate:.3f}"

    def test_designed_rounds_bqp_circuit(self, fx_rng: Generator) -> None:
        """Designed round counts tolerate sub-threshold malicious noise on the output node.

        For each BQP circuit:
        - optimize_with_robustness_constraint computes (d, s, w) from RandomTraps's
          detection_rate with epsilon=1e-10 and rho_min=0.1.
        - A MaliciousNoiseModel is applied exclusively on the output node(s) with
          prob = rho_min / 2 = 0.05, strictly below the robustness threshold.
        - Delegation runs under DensityMatrixBackend.

        Because noise < rho_min, the protocol's guarantees hold: traps should still
        pass and the majority-vote answer should match the expected BQP output.
        """
        import random

        rho_min = 0.1

        with Path("tests/test_circuits/table.json").open() as f:
            table = json.load(f)
        table_keys = list(table.keys())
        circuit_label = table_keys[fx_rng.integers(len(table_keys))]
        prob = table[circuit_label]
        bqp_error = prob if prob <= 1 / 2 else 1 - prob

        pattern = load_pattern_from_circuit(circuit_label)
        correct_answer = round(prob)
        secrets = Secrets(a=True, r=True, theta=True)

        protocol = RandomTraps()
        client = Client(pattern=pattern, secrets=secrets, protocol=protocol, rng=fx_rng)

        design = optimize_with_robustness_constraint(
            c=bqp_error,
            detection_rate=protocol.detection_rate,
            epsilon_target=1e-2,
            rho_min=rho_min,
            n_grid=200,
        )

        parameters = TrappifiedSchemeParameters(
            comp_rounds=design.d,
            test_rounds=design.s,
            threshold=design.w,
        )
        client.trappifiedScheme.params = parameters

        noise_model = MaliciousNoiseModel(
            nodes=[int(n) for n in client.output_nodes],
            prob=rho_min * 0.5,
            rng=fx_rng,
        )

        canvas = client.sample_canvas(rng=fx_rng)
        outcomes = client.delegate_canvas(
            canvas=canvas,
            backend_cls=DensityMatrixBackend,
            noise_model=noise_model,
            rng=fx_rng,
        )
        traps_decision, computation_decision, _ = client.analyze_outcomes(canvas, outcomes)

        assert traps_decision, f"{circuit_label}: honest server rejected"
        assert int(computation_decision) == correct_answer, (
            f"{circuit_label}: majority vote gave {int(computation_decision)}, expected {correct_answer}"
        )

    def test_designed_rounds_noiseless(self, fx_rng: Generator) -> None:
        """Round counts from util_rounds integrate correctly with the VBQC pipeline.

        Uses optimize_with_robustness_constraint to compute (d, s, w) from the
        protocol's detection_rate, then feeds those values into TrappifiedSchemeParameters.
        With a noiseless honest server, all traps must pass (traps_decision=True).

        Loose security parameters (epsilon=0.05, rho_min=0.01, coarse grid) are used
        to keep the optimised round counts small and the test fast.
        """
        nqubits = 2
        depth = 1
        epsilon_target = 0.1
        rho_min = 0.1
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        protocol = FK12()
        _ = protocol.create_test_runs(graph=pattern.extract_graph())

        design = optimize_with_robustness_constraint(
            c=0,
            detection_rate=protocol.detection_rate,
            epsilon_target=epsilon_target,
            rho_min=rho_min,
            n_grid=200,
        )

        assert design.total_bound <= epsilon_target
        assert design.w_over_s >= rho_min  # TODO: missing alpha

        parameters = TrappifiedSchemeParameters(
            comp_rounds=design.d,
            test_rounds=design.s,
            threshold=design.w,
        )
        secrets = Secrets(a=True, r=True, theta=True)
        pattern = circuit.transpile().pattern
        client = Client(pattern=pattern, secrets=secrets, protocol=protocol, parameters=parameters, rng=fx_rng)

        canvas = client.sample_canvas(rng=fx_rng)
        outcomes = client.delegate_canvas(canvas=canvas, backend_cls=StatevectorBackend, rng=fx_rng)
        traps_decision, _, _ = client.analyze_outcomes(canvas, outcomes)

        assert traps_decision

    @pytest.mark.parametrize("noise_model_name", ["depolarising", "malicious"])
    def test_noisy(self, fx_rng: Generator, noise_model_name: str) -> None:
        nqubits = 3
        depth = 3
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        states = [BasicStates.PLUS for _ in pattern.input_nodes]
        secrets = Secrets(a=True, r=True, theta=True)

        client = Client(pattern=pattern, input_state=states, secrets=secrets, rng=fx_rng)

        noise_model: NoiseModel
        if noise_model_name == "depolarising":
            noise_model = DepolarisingNoiseModel(
                measure_error_prob=1,
                entanglement_error_prob=1,
                x_error_prob=1,
                z_error_prob=1,
                measure_channel_prob=1,
            )
        elif noise_model_name == "malicious":
            subset_size = int(fx_rng.integers(1, len(client.nodes)))
            malicious_nodes = [int(node) for node in fx_rng.choice(client.nodes, size=subset_size, replace=False)]

            noise_model = MaliciousNoiseModel(
                nodes=malicious_nodes,
                prob=1,
                rng=fx_rng,
            )
        else:
            raise ValueError(f"Unknown noise model: {noise_model_name}")

        failures = 0

        for test_run in client.test_runs:
            backend = DensityMatrixBackend()
            trap_outcomes = test_run.accept(client, backend, noise_model, fx_rng).trap_outcomes

            if sum(trap_outcomes.values()) > 0:
                failures += 1

        assert failures > 0


def find_correct_value(circuit_name: str) -> Outcome:
    with Path("tests/test_circuits/table.json").open() as f:
        table = json.load(f)
        # return 1 if yes instance
        # return 0 else (no instance, as circuits are already filtered)
        # print(table[circuit_name])
        return round(table[circuit_name])


def sample_non_empty_subset(nodes, rng: np.random.Generator) -> frozenset:
    nodes = list(nodes)

    while True:
        keep = rng.random(len(nodes)) < 0.5
        if keep.any():
            return frozenset(node for node, k in zip(nodes, keep, strict=True) if k)
