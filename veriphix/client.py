from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, cast

import graphix.command
import graphix.ops
import graphix.pattern
import graphix.pauli
import graphix.sim.base_backend
import graphix.sim.statevec
import graphix.simulator
import stim
from graphix import command
from graphix.clifford import Clifford
from graphix.command import BaseCommand, BaseM, BaseN, Command, CommandKind
from graphix.measurements import BlochMeasurement, Measurement, toggle_outcome
from graphix.pattern import Pattern
from graphix.rng import ensure_rng
from graphix.sim.statevec import Statevec
from graphix.simulator import MeasureMethod, PrepareMethod
from graphix.states import BasicStates, State
from typing_extensions import override

from veriphix.blinding import SecretDatas, Secrets
from veriphix.malicious_noise_model import MaliciousNoiseModel
from veriphix.protocols import FK12, VerificationProtocol
from veriphix.verifying import (
    ComputationRun,
    ResultAnalysis,
    Run,
    RunResult,
    TrappifiedScheme,
    TrappifiedSchemeParameters,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from typing import TypeVar

    import networkx as nx
    from graphix.measurements import Outcome
    from graphix.noise_models import NoiseModel
    from graphix.sim.base_backend import Backend
    from numpy.random import Generator

    _StateT = TypeVar("_StateT")


@dataclass
class ByProduct:
    z_domain: set[int]
    x_domain: set[int]


def get_byproduct_db(pattern: Pattern) -> dict[int, ByProduct]:
    byproduct_db = dict()
    for node in pattern.output_nodes:
        byproduct_db[node] = ByProduct(z_domain=set(), x_domain=set())

    for cmd in pattern.correction_commands():
        if cmd.node in pattern.output_nodes:
            if cmd.kind == CommandKind.Z:
                byproduct_db[cmd.node].z_domain = cmd.domain
            if cmd.kind == CommandKind.X:
                byproduct_db[cmd.node].x_domain = cmd.domain
    return byproduct_db


def remove_flow(pattern: Pattern) -> Pattern:
    clean_pattern = Pattern(pattern.input_nodes)
    for cmd in pattern:
        if cmd.kind in (CommandKind.X, CommandKind.Z):
            # If byproduct, remove it so it's not done by the server
            continue
        if cmd.kind == CommandKind.M:
            # If measure, remove measure parameters
            new_cmd: BaseCommand = graphix.command.BaseM(node=cmd.node)
        else:
            new_cmd = cmd
        # The type annotations require the pattern to contain only
        # elements of type `Command`, which excludes `BaseM`. We hope
        # pattern types will become more precise in the near future.
        # See https://github.com/TeamGraphix/graphix/issues/266
        clean_pattern.add(cast("Command", new_cmd))
    return clean_pattern


def get_graph_clifford_structure(graph: nx.Graph[int]) -> stim.Tableau:
    circuit = stim.Circuit()
    for edge in graph.edges:
        i, j = edge
        circuit.append_from_stim_program_text(f"CZ {i} {j}")
    return circuit.to_tableau()


def qCircuit_predicate(output_string: str) -> bool:
    return int(output_string[0]) == 1


class Client:
    input_state: list[State]
    results: dict[int, Outcome]

    def __init__(
        self,
        pattern: Pattern,
        input_state: Iterable[State] | None = None,
        classical_output: bool = True,
        output_predicate: Callable[[str], bool] = qCircuit_predicate,
        measure_method_cls: Callable[[Client], MeasureMethod] | None = None,
        test_measure_method_cls: Callable[[Client], MeasureMethod] | None = None,
        secrets: Secrets | None = None,
        parameters: TrappifiedSchemeParameters | None = None,
        protocol: VerificationProtocol | None = None,
        autogen: bool = True,
        rng: Generator | None = None,
        *,
        stacklevel: int = 1,
    ) -> None:
        self.initial_pattern: Pattern = pattern
        self.classical_output = classical_output
        self.output_predicate = output_predicate
        self.input_nodes = pattern.input_nodes.copy()
        self.output_nodes = pattern.output_nodes.copy()
        self.input_state = [BasicStates.PLUS for _ in self.input_nodes] if input_state is None else list(input_state)
        self.protocol = protocol or FK12()
        self.parameters = parameters
        if autogen:
            self.preprocess_pattern(classical_output=classical_output)
            self.create_blind_patterns(
                measure_method_cls=measure_method_cls,
                test_measure_method_cls=test_measure_method_cls,
                secrets=secrets,
                rng=rng,
                stacklevel=stacklevel + 1,
            )
            self.create_trappified_scheme(rng=rng, stacklevel=stacklevel + 1)

    def preprocess_pattern(self, classical_output: bool = True) -> None:
        if classical_output:
            self._add_measurement_commands(self.initial_pattern)

        self.graph = self.initial_pattern.extract_graph()
        self.clifford_structure = get_graph_clifford_structure(self.graph)

        self.results = self.initial_pattern.results.copy()

    def create_blind_patterns(
        self,
        measure_method_cls: Callable[[Client], MeasureMethod] | None = None,
        test_measure_method_cls: Callable[[Client], MeasureMethod] | None = None,
        secrets: Secrets | None = None,
        rng: Generator | None = None,
        *,
        stacklevel: int = 1,
    ) -> None:
        rng = ensure_rng(rng, stacklevel=stacklevel + 1)
        self.measure_method = (measure_method_cls or ClientMeasureMethod)(self)
        self.test_measure_method = (test_measure_method_cls or TestMeasureMethod)(self)

        self.measurement_db = self._get_measurement_db()
        self.byproduct_db = get_byproduct_db(self._copy_pattern())

        self.secrets = secrets or Secrets()
        self.secret_datas = SecretDatas.from_secrets(
            self.secrets, self.graph, set(self.input_nodes), set(self.output_nodes), rng=rng
        )

        self.clean_pattern = remove_flow(self.initial_pattern)
        if not self.classical_output:
            self.test_pattern = remove_flow(self.initial_pattern)
            self._add_measurement_commands(self.test_pattern)
        else:
            self.test_pattern = self.clean_pattern
        self.computation_states = self.get_computation_states()

        self.preparation_bank: dict[int, Statevec] = {}
        self.prepare_method = ClientPrepareMethod(self.preparation_bank)

    def create_trappified_scheme(self, rng: Generator | None = None, *, stacklevel: int = 1) -> None:
        self.computationRun = ComputationRun(self)
        self.test_runs = self.protocol.create_test_runs(client=self, rng=rng, stacklevel=stacklevel + 1)
        self.trappifiedScheme = TrappifiedScheme(
            params=self.parameters or TrappifiedSchemeParameters(20, 20, 5), test_runs=self.test_runs
        )

    @property
    def nodes(self) -> list[int]:
        return list(self.graph.nodes)

    def _add_measurement_commands(self, pattern: Pattern) -> None:
        for onode in self.output_nodes:
            pattern.add(graphix.command.M(node=onode))

    def _copy_pattern(self) -> Pattern:
        pattern_copy = Pattern(self.initial_pattern.input_nodes)
        for cmd in self.initial_pattern:
            pattern_copy.add(cmd)
        pattern_copy.standardize()
        return pattern_copy

    def _get_measurement_db(self) -> dict[int, command.M]:
        copied_pattern = self._copy_pattern()
        return copied_pattern.extract_measurement_commands()

    def refresh_randomness(self, rng: Generator | None = None, *, stacklevel: int = 1) -> None:
        "method to refresh random randomness using parameters from Clinent instatiation."

        # refresh only if secrets bool is True; False is no randomness at all.
        if self.secrets is not None:
            self.secret_datas = SecretDatas.from_secrets(
                self.secrets,
                self.graph,
                set(self.input_nodes),
                set(self.output_nodes),
                rng=rng,
                stacklevel=stacklevel + 1,
            )

    def get_computation_states(self) -> dict[int, State]:
        states = dict()
        for node in self.graph.nodes:
            if node in self.input_nodes:
                state = self.input_state[node]

            elif node in self.output_nodes:
                r_value = self.secret_datas.r.get(node, 0)
                a_N_value = self.secret_datas.a.a_N.get(node, 0)
                state = BasicStates.PLUS if r_value ^ a_N_value == 0 else BasicStates.MINUS

            else:
                state = BasicStates.PLUS
            states[node] = state
        return states

    def prepare_states_virtual(self, states_dict: Mapping[int, State]) -> None:
        """
        The Client creates the qubits and blind them in its preparation_bank
        """
        for node in states_dict:
            blinded_qubit_state = self.secret_datas.blind_qubit(node=node, state=states_dict[node])
            self.preparation_bank[node] = Statevec(blinded_qubit_state)

    def prepare_states(self, backend: Backend[_StateT], states_dict: Mapping[int, State]) -> None:
        # Initializes the bank (all the nodes)
        self.prepare_states_virtual(states_dict=states_dict)
        # Server asks the backend to create them
        ## Except for the input! The Client creates them itself
        for node in self.input_nodes:
            self.prepare_method.prepare_node(backend, node)

    def sample_canvas(self, rng: Generator | None = None, *, stacklevel: int = 1) -> dict[int, Run]:
        rng = ensure_rng(rng, stacklevel=stacklevel + 1)
        N = self.trappifiedScheme.params.comp_rounds + self.trappifiedScheme.params.test_rounds
        computation_rounds = set(rng.integers(N, size=self.trappifiedScheme.params.comp_rounds))

        return {
            r: self.computationRun if r in computation_rounds else self.test_runs[rng.integers(len(self.test_runs))]
            for r in range(N)
        }

    def delegate_canvas(
        self,
        canvas: dict[int, Run],
        backend_cls: type[Backend[_StateT]],
        noise_model: NoiseModel | None = None,
        rng: Generator | None = None,
    ) -> dict[int, RunResult[_StateT]]:
        outcomes = dict()
        for r in canvas:
            backend = backend_cls()
            if isinstance(noise_model, MaliciousNoiseModel):
                noise_model.refresh_randomness(rng=rng)
            outcomes[r] = canvas[r].delegate(backend=backend, noise_model=noise_model, rng=rng)
        return outcomes

    def analyze_outcomes(
        self, canvas: dict[int, Run], outcomes: dict[int, RunResult[_StateT]]
    ) -> tuple[bool, bool, ResultAnalysis[_StateT]]:
        result_analysis: ResultAnalysis[_StateT] = ResultAnalysis()
        for r in canvas:
            outcomes[r].analyze(result_analysis=result_analysis, client=self)

        # True if Accept, False if Reject
        traps_decision = result_analysis.nr_failed_test_rounds <= self.trappifiedScheme.params.threshold
        # The Client decides that the instance passes the predicate if more than half of the test rounds did pass
        computation_decision = result_analysis.computation_count >= ceil(self.trappifiedScheme.params.comp_rounds / 2)

        return traps_decision, computation_decision, result_analysis

    def decode_output_state(self, backend: Backend[_StateT]) -> None:
        for node in self.output_nodes:
            z_decoding, x_decoding = self.decode_output(node)
            if z_decoding:
                backend.correct_byproduct(command.Z(node))
            if x_decoding:
                backend.correct_byproduct(command.X(node))

    def decode_output(self, node: int) -> tuple[int, int]:
        z_decoding = sum(self.results[z_dep] for z_dep in self.byproduct_db[node].z_domain) % 2
        z_decoding ^= self.secret_datas.r.get(node, 0)
        x_decoding = sum(self.results[x_dep] for x_dep in self.byproduct_db[node].x_domain) % 2
        x_decoding ^= self.secret_datas.a.a.get(node, 0)
        return z_decoding, x_decoding


class ClientPrepareMethod(PrepareMethod):
    def __init__(self, preparation_bank: dict[int, Statevec]) -> None:
        self.__preparation_bank = preparation_bank

    def prepare_node(self, backend: Backend[_StateT], node: int) -> None:
        """Prepare a node."""
        backend.add_nodes(nodes=[node], data=self.__preparation_bank[node])

    @override
    def prepare(self, backend: Backend[_StateT], cmd: BaseN, rng: Generator | None = None) -> None:
        """Prepare a node."""
        self.prepare_node(backend, cmd.node)


class BlindMeasureMethod(MeasureMethod):
    def __init__(self, client: Client):
        self._client = client

    @override
    def measurement_outcome(self, node: int) -> Outcome:
        raise ValueError("Server cannot have access to measurement results")

    @override
    def store_measurement_outcome(self, node: int, result: Outcome) -> None:
        if self._client.secret_datas.r:
            if self._client.secret_datas.r[node]:
                result = toggle_outcome(result)
        self._client.results[node] = result


class ClientMeasureMethod(BlindMeasureMethod):
    @override
    def describe_measurement(self, cmd: BaseM) -> BlochMeasurement:
        parameters = self._client.measurement_db[cmd.node]

        # Extract secrets from Client
        a_value = self._client.secret_datas.a.a.get(cmd.node, 0)

        # Extract signals and compute the angle for the computation
        s_signal = sum(self._client.results[j] for j in parameters.s_domain) % 2
        t_signal = sum(self._client.results[j] for j in parameters.t_domain) % 2
        measurement = parameters.measurement
        if s_signal:
            measurement = measurement.clifford(Clifford.X)
        if t_signal:
            measurement = measurement.clifford(Clifford.Z)
        bloch = measurement.to_bloch()
        # Blind the angle using the Client's secrets
        angle = (-1) ** a_value * bloch.angle + self._client.secret_datas.blind_angle(
            cmd.node, cmd.node in self._client.output_nodes, test=False
        )
        return BlochMeasurement(angle, bloch.plane)


class TestMeasureMethod(BlindMeasureMethod):
    @override
    def describe_measurement(self, cmd: BaseM) -> Measurement:
        # Blind the angle using the Client's secrets
        angle = self._client.secret_datas.blind_angle(cmd.node, cmd.node in self._client.output_nodes, test=True)

        return Measurement.XY(angle)
