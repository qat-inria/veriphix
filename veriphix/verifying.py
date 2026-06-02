# override introduced in Python 3.12
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

import stim
from graphix.fundamentals import IXYZ_VALUES
from graphix.pauli import Pauli
from stim import PauliString
from typing_extensions import override

Trap = AbstractSet[int]
Traps = AbstractSet[Trap]


if TYPE_CHECKING:
    from typing import Literal

    import networkx as nx
    from graphix.measurements import Outcome
    from graphix.noise_models import NoiseModel
    from graphix.sim.base_backend import Backend
    from graphix.states import PlanarState
    from numpy.random import Generator

    from veriphix.client import Client

_StateT = TypeVar("_StateT")


class Run(ABC):
    @abstractmethod
    def accept(
        self,
        executor: Client,
        backend: Backend[_StateT],
        noise_model: NoiseModel | None,
        rng: Generator | None,
    ) -> RunResult[_StateT]:
        pass


class ComputationRun(Run):
    @override
    def accept(
        self,
        executor: Client,
        backend: Backend[_StateT],
        noise_model: NoiseModel | None,
        rng: Generator | None,
    ) -> ComputationResult[_StateT]:
        return executor.execute_computation_run(backend, noise_model, rng)


def merge_pauli_strings(stabilizer_1: PauliString, stabilizer_2: PauliString) -> PauliString:
    result = stabilizer_2.sign * stabilizer_1
    # We can iterate through the support of stab2 as they are supposed to have equal support
    for node in stabilizer_2.pauli_indices("XYZ"):
        result[node] = stabilizer_2[node]
    return result


def merge(strings: list[PauliString]) -> PauliString:
    n = len(strings)
    l = len(strings[0])
    common_string = strings[0]
    for i in range(1, n):
        common_string.sign *= strings[i].sign
        for j in range(l):
            if (common_string[j] != 0 and strings[i][j] != 0) and (common_string[j] != strings[i][j]):
                raise Exception("Traps are not compatible.")
            if common_string[j] == 0 and strings[i][j] != 0:
                common_string[j] = strings[i][j]
    return common_string


def generate_eigenstate(stabilizer: PauliString) -> list[PlanarState]:
    states = []
    for pauli in stabilizer:  # type: ignore[attr-defined]
        # default coin = 0
        operator = Pauli(IXYZ_VALUES[pauli])
        states.append(operator.eigenstate())
    return states


def get_graph_clifford_structure(graph: nx.Graph[int]) -> stim.Tableau:
    nodes = list(graph.nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    circuit = stim.Circuit()
    for i, j in graph.edges:
        circuit.append_from_stim_program_text(f"CZ {node_to_idx[i]} {node_to_idx[j]}")
    return circuit.to_tableau()


def build_stabilizer(
    graph: nx.Graph,
    nqubits: int,
    traps: Traps,
    meas_basis: str = "X",
) -> PauliString:
    nodes = list(graph.nodes)
    measurement_strings = [PauliString([meas_basis if node in trap else "I" for node in nodes]) for trap in traps]
    clifford_structure = get_graph_clifford_structure(graph=graph)
    conjugated_measurements = [clifford_structure.inverse()(meas) for meas in measurement_strings]
    return merge(conjugated_measurements)


class TestRun(Run):
    meas_basis: Literal["_", "I", "X", "Y", "Z"]

    __test__ = False  # this is not a pytest test-suite

    def __init__(
        self,
        graph: nx.Graph,
        nqubits: int,
        traps: Traps,
        meas_basis: Literal["_", "I", "X", "Y", "Z"] = "X",
    ) -> None:
        self.traps = frozenset(traps)
        self.meas_basis = meas_basis
        self.nqubits = nqubits
        self.stabilizer = build_stabilizer(graph, nqubits, traps, meas_basis)
        self.input_state = dict(zip(graph.nodes, generate_eigenstate(self.stabilizer)))

    @override
    def accept(
        self,
        executor: Client,
        backend: Backend[_StateT],
        noise_model: NoiseModel | None,
        rng: Generator | None,
    ) -> TestResult[_StateT]:
        return executor.execute_test_run(self, backend, noise_model, rng)

    def __str__(self) -> str:
        return f"""
        Traps: {self.traps}
        Stabilizer: {self.stabilizer}"""


class RunResult(ABC, Generic[_StateT]):
    @abstractmethod
    def analyze(self, result_analysis: ResultAnalysis[_StateT], client: Client) -> None: ...


class ComputationResult(RunResult[_StateT], Generic[_StateT]):
    pass


class ClassicalComputationResult(ComputationResult[_StateT], Generic[_StateT]):
    def __init__(self, outcomes: dict[int, Outcome]):
        self.outcomes = outcomes
        self.outcome_string = None

    def analyze(self, result_analysis: ResultAnalysis[_StateT], client: Client) -> None:
        output_string = "".join(str(int(v)) for v in self.outcomes.values())
        result_analysis.computation_count += client.output_predicate(output_string)
        return

    def __str__(self) -> str:
        return f"""
        Outcomes of Computation round: {self.outcome_string}
        """


class QuantumComputationResult(Generic[_StateT], ComputationResult[_StateT]):
    def __init__(self, output_state: _StateT):
        self.output_state = output_state

    def analyze(self, result_analysis: ResultAnalysis[_StateT], client: Client) -> None:
        result_analysis.quantum_output_states.append(self.output_state)


class TestResult(Generic[_StateT], RunResult[_StateT]):
    def __init__(self, trap_outcomes: dict[frozenset[int], int]):
        self.trap_outcomes = trap_outcomes
        self.failed_round = False

    def analyze(self, result_analysis: ResultAnalysis[_StateT], client: Client) -> None:
        self.failed_round = sum(self.trap_outcomes.values()) > 0
        result_analysis.nr_failed_test_rounds += self.failed_round

    def __str__(self) -> str:
        return f"""
        Outcomes of Test round.
        Trap outcomes: {self.trap_outcomes}
        Overall outcome for the round: {int(self.failed_round)} ({"failed" if self.failed_round else "passed"})
        """


@dataclass
class TrappifiedSchemeParameters:
    comp_rounds: int  # nr of comp rounds
    test_rounds: int  # nr of test rounds
    threshold: int  # threshold (nr of allowed test rounds failure)


@dataclass
class TrappifiedScheme:
    params: TrappifiedSchemeParameters
    test_runs: list[TestRun]


@dataclass
class ResultAnalysis(Generic[_StateT]):
    nr_failed_test_rounds: int = 0
    # number of computation rounds for which the Client's predicate evaluated to True
    computation_count: int = 0
    quantum_output_states: list[_StateT] = field(default_factory=list)


@dataclass
class TrappifiedRun(Generic[_StateT]):
    input_state: list[_StateT]
    tested_qubits: list[int]
    stabilizer: Pauli
