from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from graphix.pattern import PatternSimulator
from graphix.pauli import IXYZ, Pauli
from stim import PauliString

# override introduced in Python 3.12
from typing_extensions import override

if TYPE_CHECKING:
    from graphix.sim.base_backend import Backend
    from graphix.states import State

    from veriphix.client import Client, ResultAnalysis, Traps


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


# Trap=tuple[int] # Better because immutable, so can be used as key in dictionary

## TODO: pourquoi pas frozenset ?
## collections abstraites: classe set, par défaut immutable
## TODO: Traps pourrait être AbstractSet
## L'avoir comme abstractSet en argument de la fonction, le caster en frozenset à chaque fois


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

    def analyze(self, result_analysis: ResultAnalysis, client: Client) -> None:
        if client.desired_outputs is None:
            outcome_string = "".join(str(o) for o in self.outcomes.values())
        else:
            outputs = list(self.outcomes.values())
            restricted_outputs = [int(outputs[i]) for i in client.desired_outputs]
            outcome_string = "".join(str(o) for o in restricted_outputs)

        result_analysis.computation_outcomes_count[outcome_string] = (
            result_analysis.computation_outcomes_count.get(outcome_string, 0) + 1
        )


class QuantumComputationResult(ComputationResult):
    def __init__(self, output_state: State):
        self.output_state = output_state

    def analyze(self, result_analysis: ResultAnalysis, client: Client) -> None:
        result_analysis.quantum_output_states.append(self.output_state)


class TestResult(RunResult):
    def __init__(self, trap_outcomes: dict[frozenset[int], int]):
        self.trap_outcomes = trap_outcomes

    def analyze(self, result_analysis: ResultAnalysis, client: Client) -> None:
        result_analysis.nr_failed_test_rounds += sum(self.trap_outcomes.values()) > 0
