# override introduced in Python 3.12
from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
from graphix.pattern import PatternSimulator
from graphix.pauli import IXYZ, Pauli
from graphix.sim.statevec import State
from stim import PauliString
from typing_extensions import override

Trap = AbstractSet[int]
Traps = AbstractSet[Trap]


if TYPE_CHECKING:
    from graphix.sim.base_backend import Backend
    from graphix.states import State

    from veriphix.client import Client, ResultAnalysis


class Run(ABC):
    def __init__(self, client: Client) -> None:
        self.client = client

    @abstractmethod
    def delegate(self, backend: Backend, **kwargs) -> RunResult:
        # Delegates using UBQC
        pass


class ComputationRun(Run):
    def __init__(self, client: Client) -> None:
        super().__init__(client=client)

    @override
    def delegate(self, backend: Backend, **kwargs) -> ComputationResult:
        # Initializes the bank & asks backend to create the input
        self.client.prepare_states(backend, states_dict=self.client.computation_states)

        sim = PatternSimulator(
            backend=backend,
            pattern=self.client.clean_pattern,
            prepare_method=self.client.prepare_method,
            measure_method=self.client.measure_method,
            **kwargs,
        )
        sim.run(input_state=None)

        # If quantum output, decode the state, nothing needs to be returned (backend.state can be accessed by the Client)
        if not self.client.classical_output:
            self.client.decode_output_state(backend)
            return QuantumComputationResult(backend.state)
        # If classical output, return the output
        else:
            results = {onode: self.client.results[onode] for onode in self.client.output_nodes}
            return ClassicalComputationResult(outcomes=results)


def merge_pauli_strings(stabilizer_1: PauliString, stabilizer_2: PauliString) -> PauliString:
    result = stabilizer_2.sign * stabilizer_1
    # We can iterate through the support of stab2 as they are supposed to have equal support
    for node in stabilizer_2.pauli_indices("XYZ"):
        result[node] = stabilizer_2[node]
    return result


def merge(strings: list[PauliString]):
    n = len(strings)
    l = len(strings[0])
    common_string = strings[0]
    # if common_string.sign == -1:
    #     print("Have to change the sign")
    # common_string.sign = 1
    for i in range(1, n):
        common_string.sign *= strings[i].sign
        for j in range(l):
            if (common_string[j] != 0 and strings[i][j] != 0) and (common_string[j] != strings[i][j]):
                raise Exception("Traps are not compatible.")
            if common_string[j] == 0 and strings[i][j] != 0:
                common_string[j] = strings[i][j]
    return common_string


def generate_eigenstate(stabilizer: PauliString) -> list[State]:
    states = []
    for pauli in stabilizer:
        # default coin = 0
        operator = Pauli(IXYZ(pauli))
        states.append(operator.eigenstate())
    return states


class TestRun(Run):
    def __init__(self, client: Client, traps: Traps, meas_basis: str = "X") -> None:
        super().__init__(client=client)
        self.traps = frozenset(traps)
        self.meas_basis = meas_basis
        self.clifford_structure = client.clifford_structure
        self.nqubits = len(self.clifford_structure)
        self.stabilizer = self.build_common_stabilizer()
        self.input_state = self.build_common_eigenstate()

    def build_common_stabilizer(self):
        # Build the PauliStrings representing the individual measurement of each trap qubit
        measurement_strings = [
            PauliString([self.meas_basis if i in trap else "I" for i in range(self.nqubits)]) for trap in self.traps
        ]
        # Conjugate each measurement
        conjugated_measurements = [self.clifford_structure.inverse()(meas) for meas in measurement_strings]
        common_stabilizer = merge(conjugated_measurements)
        return common_stabilizer

    def build_common_eigenstate(self):
        input_state = generate_eigenstate(self.stabilizer)
        return input_state

    @override
    def delegate(self, backend: Backend, **kwargs) -> dict[int, int]:
        states_dict = {node: self.input_state[node] for node in self.client.nodes_list}
        self.client.prepare_states(backend=backend, states_dict=states_dict)
        sim = PatternSimulator(
            backend=backend,
            pattern=self.client.test_pattern,
            prepare_method=self.client.prepare_method,
            measure_method=self.client.test_measure_method,
            **kwargs,
        )
        sim.run(input_state=None)

        trap_outcomes = dict()
        for trap in self.traps:
            outcomes = [self.client.results[component] for component in trap]
            trap_outcome = sum(outcomes) % 2 ^ (self.stabilizer.sign == -1)
            trap_outcomes[trap] = trap_outcome
            # trap_outcomes.append(trap_outcome)
        return TestResult(trap_outcomes)


class RunResult(ABC):
    @abstractmethod
    def analyze(self, result_analysis: ResultAnalysis, client: Client) -> None:
        pass


class ComputationResult(RunResult, ABC):
    @abstractmethod
    def analyze(self, result_analysis: ResultAnalysis, client: Client) -> None:
        pass


class ClassicalComputationResult(ComputationResult):
    def __init__(self, outcomes: dict[int, int]):
        self.outcomes = outcomes
        self.outcome_string = None

    def analyze(self, result_analysis: ResultAnalysis, client: Client) -> None:
        if client.desired_outputs is None:
            outcome_string = "".join(str(o) for o in self.outcomes.values())
        else:
            outputs = list(self.outcomes.values())
            restricted_outputs = [int(outputs[i]) for i in client.desired_outputs]
            outcome_string = "".join(str(o) for o in restricted_outputs)

        self.outcome_string = outcome_string
        result_analysis.computation_outcomes_count[self.outcome_string] = (
            result_analysis.computation_outcomes_count.get(self.outcome_string, 0) + 1
        )
    def __str__(self) -> str:
        return f"""
        Outcomes of Computation round: {self.outcome_string}
        """


class QuantumComputationResult(ComputationResult):
    def __init__(self, output_state: State):
        self.output_state = output_state

    def analyze(self, result_analysis: ResultAnalysis, client: Client) -> None:
        result_analysis.quantum_output_states.append(self.output_state)


class TestResult(RunResult):
    def __init__(self, trap_outcomes: dict[frozenset[int], int]):
        self.trap_outcomes = trap_outcomes
        self.failed_round = False

    def analyze(self, result_analysis: ResultAnalysis, client: Client) -> None:
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
    test_runs: list


@dataclass
class ResultAnalysis:
    nr_failed_test_rounds: int
    computation_outcomes_count: dict[str, int]
    quantum_output_states: dict[int, State]


@dataclass
class TrappifiedRun:
    input_state: list
    tested_qubits: list[int]
    stabilizer: Pauli


def create_test_runs(client: Client, manual_colouring: Sequence[set[int]] | None = None) -> list[TestRun]:
    """Creates test runs according to a graph colouring according to [FK12].
    A test run, or a Trappified Canvas, is associated to each color in the colouring.
    For a given test run, the trap nodes are defined as being the nodes belonging to the color the run corresponds to.
    This procedure only allows for the definition of single-qubit traps.
    If `manual_colouring` is not specified, then a colouring is found by using `networkx`'s greedy algorithm.

    Parameters
    ----------
    manual_colouring : Sequence[set[int]] | None, optional
        manual colouring to use if `networkx` is bypassed, by default None.

    Returns
    -------
    list[TestRun]
        list of TestRun objects defining all the possible test runs to perform.

    Raises
    ------
    ValueError
        if the provided colouring is not an actual colouring (does not cover all nodes).
    ValueError
        if the provided colouring is not a proper colouring (a node belongs to more than one color).

    Warnings
    ------
    when providing a `manual_colouring` make sure that the numbering of the nodes in the original pattern/graph and the one in the colouring are consistent.

    Notes
    ------
    [FK12]: Fitzsimons, J. F., & Kashefi, E. (2017). Unconditionally verifiable blind quantum computation. Physical Review A, 96(1), 012303.
    https://arxiv.org/abs/1203.5217
    """

    # Create the graph coloring
    # Networkx output format: dict[int, int] eg {0: 0, 1: 1, 2: 0, 3: 1}
    if manual_colouring is None:
        coloring = nx.coloring.greedy_color(client.graph, strategy="largest_first")
        colors = set(coloring.values())
        nodes_by_color: dict[int, list[int]] = {c: [] for c in colors}
        for node in sorted(client.graph.nodes):
            color = coloring[node]
            nodes_by_color[color].append(node)
    else:
        # checks that manual_colouring is a proper colouring
        ## first check uniion is the whole graph
        color_union = set().union(*manual_colouring)
        if not color_union == set(client.graph.nodes):
            raise ValueError("The provided colouring does not include all the nodes of the graph.")
        # check that colors are two by two disjoint
        # if sets are disjoint, empty set from intersection is interpreted as False.
        # so one non-empty set -> one True value -> use any()
        if any([i & j for i, j in itertools.combinations(manual_colouring, 2)]):
            raise ValueError(
                "The provided colouring is not a proper colouring i.e the same node belongs to at least two colours."
            )

        nodes_by_color = {i: list(c) for i, c in enumerate(manual_colouring)}
        colors = nodes_by_color.keys()

    # Create the test runs : one per color
    test_runs: list[TestRun] = []
    for color in colors:
        # 1 color = 1 test run = 1 collection of single-qubit traps
        traps_list = [frozenset([colored_node]) for colored_node in nodes_by_color[color]]
        traps = frozenset(traps_list)
        test_run = TestRun(client=client, traps=traps)
        test_runs.append(test_run)

    # print(test_runs)
    return test_runs
