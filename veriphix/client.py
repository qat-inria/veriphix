from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING

import graphix.command
import graphix.ops
import graphix.pattern
import graphix.pauli
import graphix.sim.base_backend
import graphix.sim.statevec
import graphix.simulator
import networkx as nx
import numpy as np
from graphix.clifford import Clifford
from graphix.command import BaseM, BaseN, CommandKind, MeasureUpdate
from graphix.fundamentals import Plane
from graphix.measurements import Measurement
from graphix.ops import Ops
from graphix.pattern import Pattern
from graphix.pauli import Pauli
from graphix.sim.statevec import Statevec, StatevectorBackend
from graphix.simulator import MeasureMethod, PrepareMethod
from graphix.states import BasicStates
from stim import Circuit

if TYPE_CHECKING:
    from collections.abc import Sequence

    import stim
    from graphix.sim.base_backend import Backend


from collections.abc import Set as AbstractSet

Trap = AbstractSet[int]
Traps = AbstractSet[Trap]


@dataclass
class TrappifiedScheme:
    params: TrappifiedSchemeParameters
    test_runs: list


@dataclass
class TrappifiedSchemeParameters:
    comp_rounds: int  # nr of comp rounds
    test_rounds: int  # nr of test rounds
    threshold: int  # threshold (nr of allowed test rounds failure)


# TODO update docstring
"""
Usage:

client = Client(pattern:Pattern, blind=False) ## For pure MBQC
sv_backend = StatevecBackend(client.pattern, meas_op = client.meas_op)

simulator = PatternSimulator(client.pattern, backend=sv_backend)
simulator.run()

"""


## TODO : implémenter ça, et l'initialiser
@dataclass
class ResultAnalysis:
    nr_failed_test_rounds: int
    computation_outcomes_count: dict[str, int]


@dataclass
class TrappifiedRun:
    input_state: list
    tested_qubits: list[int]
    stabilizer: Pauli


@dataclass
class Secret_a:
    a: dict[int, int]
    a_N: dict[int, int]


@dataclass
class Secrets:
    r: bool = True
    a: bool = True
    theta: bool = True


@dataclass
class SecretDatas:
    r: dict[int, int]
    a: Secret_a
    theta: dict[int, int]

    # NOTE: not a class method?
    @staticmethod
    def from_secrets(secrets: Secrets, graph: nx.Graph, input_nodes, output_nodes):
        r = {}
        if secrets.r:
            # Need to generate the random bit for each measured qubit, 0 for the rest (output qubits)
            for node in graph.nodes:
                r[node] = np.random.randint(0, 2)  # if node not in output_nodes else 0

        theta = {}
        if secrets.theta:
            # Create theta secret for all non-output nodes (measured qubits)
            for node in graph.nodes:
                theta[node] = np.random.randint(0, 8) if node not in output_nodes else 0  # Expressed in pi/4 units
                ## TODO:
        a = {}
        a_N = {}
        if secrets.a:
            # Create `a` secret for all
            # order is Z(theta) X |+>
            for node in graph.nodes:
                a[node] = np.random.randint(0, 2) if node in input_nodes else 0

            # After all the `a` secrets have been generated, the `a_N` value can be
            # computed from the graph topology
            for i in graph.nodes:
                a_N_value = sum([a[j] for j in graph.neighbors(i)]) % 2
                # for j in graph.neighbors(i):
                #     a_N_value ^= a[j]
                a_N[i] = a_N_value

        return SecretDatas(r, Secret_a(a, a_N), theta)


@dataclass
class ByProduct:
    z_domain: list[int]
    x_domain: list[int]


def get_byproduct_db(pattern: Pattern) -> dict[int, ByProduct]:
    byproduct_db = dict()
    for node in pattern.output_nodes:
        byproduct_db[node] = ByProduct(z_domain=[], x_domain=[])

    for cmd in pattern.correction_commands():
        if cmd.node in pattern.output_nodes:
            if cmd.kind == CommandKind.Z:
                byproduct_db[cmd.node].z_domain = cmd.domain
            if cmd.kind == CommandKind.X:
                byproduct_db[cmd.node].x_domain = cmd.domain
    return byproduct_db


def remove_flow(pattern):
    clean_pattern = Pattern(pattern.input_nodes)
    for cmd in pattern:
        if cmd.kind in (CommandKind.X, CommandKind.Z):
            # If byproduct, remove it so it's not done by the server
            continue
        if cmd.kind == CommandKind.M:
            # If measure, remove measure parameters
            new_cmd = graphix.command.BaseM(node=cmd.node)
        else:
            new_cmd = cmd
        clean_pattern.add(new_cmd)
    return clean_pattern


def get_graph_clifford_structure(graph: nx.Graph):
    circuit = Circuit()
    for edge in graph.edges:
        i, j = edge
        circuit.append_from_stim_program_text(f"CZ {i} {j}")
    return circuit.to_tableau()


class Client:
    def __init__(
        self,
        pattern,
        input_state=None,
        classical_output: bool = True,
        desired_outputs: list[int] | None = None,
        measure_method_cls=None,
        test_measure_method_cls=None,
        secrets: Secrets | None = None,
        parameters: TrappifiedSchemeParameters | None = None,
    ) -> None:
        self.initial_pattern: Pattern = pattern
        self.classical_output = classical_output
        self.desired_outputs = desired_outputs
        self.input_nodes = pattern.input_nodes.copy()
        self.output_nodes = pattern.output_nodes.copy()

        if classical_output:
            self._add_measurement_commands(self.initial_pattern)

        self.graph = self._build_graph()
        self.clifford_structure = get_graph_clifford_structure(self.graph)
        self.nodes_list = list(self.graph.nodes)

        self.results = pattern.results.copy()
        self.measure_method = (measure_method_cls or ClientMeasureMethod)(self)
        self.test_measure_method = (test_measure_method_cls or TestMeasureMethod)(self)

        self.measurement_db = self._get_measurement_db()
        self.byproduct_db = get_byproduct_db(self._copy_pattern())

        self.secrets = secrets or Secrets()
        self.secrets_bool = secrets is not None
        self.secret_datas = SecretDatas.from_secrets(self.secrets, self.graph, self.input_nodes, self.output_nodes)

        self.clean_pattern = remove_flow(self.initial_pattern)
        if not self.classical_output:
            self.test_pattern = self._add_measurement_commands(remove_flow(self.initial_pattern))
        else:
            self.test_pattern = self.clean_pattern
        self.input_state = input_state or [BasicStates.PLUS for _ in self.input_nodes]
        self.computation_states = self.get_computation_states()

        self.preparation_bank = {}
        self.prepare_method = ClientPrepareMethod(self.preparation_bank)

        from veriphix.run import ComputationRun

        self.computationRun = ComputationRun(self)
        self.test_runs = self.create_test_runs()
        self.trappifiedScheme = TrappifiedScheme(
            params=parameters or TrappifiedSchemeParameters(20, 20, 5), test_runs=self.test_runs
        )

    def _add_measurement_commands(self, pattern):
        for onode in self.output_nodes:
            pattern.add(graphix.command.M(node=onode))
        return pattern

    def _build_graph(self):
        raw_graph = self.initial_pattern.get_graph()
        graph = nx.Graph()
        graph.add_nodes_from(raw_graph[0])
        graph.add_edges_from(raw_graph[1])
        return graph

    def _copy_pattern(self) -> Pattern:
        pattern_copy = Pattern(self.initial_pattern.input_nodes)
        for cmd in self.initial_pattern:
            pattern_copy.add(cmd)
        pattern_copy.standardize()
        return pattern_copy

    def _get_measurement_db(self):
        copied_pattern = self._copy_pattern()
        return {m.node: m for m in copied_pattern.get_measurement_commands()}

    def refresh_randomness(self) -> None:
        "method to refresh random randomness using parameters from Clinent instatiation."

        # refresh only if secrets bool is True; False is no randomness at all.
        if self.secrets is not None:
            self.secret_datas = SecretDatas.from_secrets(self.secrets, self.graph, self.input_nodes, self.output_nodes)

    def get_computation_states(self):
        states = dict()
        for node in self.nodes_list:
            if node in self.input_nodes:
                state = self.input_state[node]

            elif node in self.output_nodes:
                r_value = self.secret_datas.r.get(node, 0)
                a_N_value = self.secret_datas.a.a_N.get(node, 0)
                # TODO: here is where the additional decoding happens
                state = BasicStates.PLUS if r_value ^ a_N_value == 0 else BasicStates.MINUS

            else:
                state = BasicStates.PLUS
            states[node] = state
        return states

    def prepare_states_virtual(self, states_dict: dict[(int, BasicStates)], backend: Backend) -> None:
        """
        The Client creates the qubits and blind them in its preparation_bank
        """
        for node in states_dict:
            blinded_qubit_state = self.blind_qubit(node=node, state=states_dict[node])
            self.preparation_bank[node] = Statevec(blinded_qubit_state)

    def blind_qubit(self, node: int, state) -> None:
        def z_rotation(theta) -> np.array:
            return np.array([[1, 0], [0, np.exp(1j * theta * np.pi / 4)]])

        def x_blind(a) -> Pauli:
            return Pauli.X if (a == 1) else Pauli.I

        theta = self.secret_datas.theta.get(node, 0)
        a = self.secret_datas.a.a.get(node, 0)
        single_qubit_backend = StatevectorBackend()
        single_qubit_backend.add_nodes([0], [state])
        if a:
            single_qubit_backend.apply_single(node=0, op=x_blind(a).matrix)
        if theta:
            single_qubit_backend.apply_single(node=0, op=z_rotation(theta))
        return single_qubit_backend.state

    def prepare_states(self, backend: Backend, states_dict: dict[(int, BasicStates)]) -> None:
        # Initializes the bank (all the nodes)
        self.prepare_states_virtual(backend=backend, states_dict=states_dict)
        # Server asks the backend to create them
        ## Except for the input! The Client creates them itself
        for node in self.input_nodes:
            self.prepare_method.prepare_node(backend, node)

    def create_test_runs(self, manual_colouring: Sequence[set[int]] | None = None):
        from veriphix.run import TestRun

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
            coloring = nx.coloring.greedy_color(self.graph, strategy="largest_first")
            colors = set(coloring.values())
            nodes_by_color: dict[int, list[int]] = {c: [] for c in colors}
            for node in sorted(self.graph.nodes):
                color = coloring[node]
                nodes_by_color[color].append(node)
        else:
            # checks that manual_colouring is a proper colouring
            ## first check uniion is the whole graph
            color_union = set().union(*manual_colouring)
            if not color_union == set(self.graph.nodes):
                raise ValueError("The provided colouring does not include all the nodes of the graph.")
            # check that colors are two by two disjoint
            # if sets are disjoint, empty set from intersection is interpreted as False.
            # so one non-empty set -> one True value -> use any()
            if any([i & j for i, j in itertools.combinations(manual_colouring, 2)]):
                raise ValueError(
                    "The provided colouring is not a proper colouring i.e the same node belongs to at least two colours."
                )

            colors = set(range(len(manual_colouring)))
            nodes_by_color = {i: list(c) for i, c in enumerate(manual_colouring)}

        # Create the test runs : one per color
        test_runs: list[TestRun] = []
        for color in colors:
            # 1 color = 1 test run = 1 collection of single-qubit traps
            traps_list = [frozenset([colored_node]) for colored_node in nodes_by_color[color]]
            traps: Traps = frozenset(traps_list)
            test_run = TestRun(client=self, traps=traps)
            test_runs.append(test_run)

        # print(test_runs)
        return test_runs

    def sample_canvas(self):
        N = self.trappifiedScheme.params.comp_rounds + self.trappifiedScheme.params.test_rounds
        computation_rounds = set(random.sample(range(N), self.trappifiedScheme.params.comp_rounds))

        rounds_run = dict()
        for r in range(N):
            if r in computation_rounds:
                rounds_run[r] = self.computationRun
            else:
                rounds_run[r] = random.choice(self.test_runs)
        return rounds_run

    def delegate_canvas(self, canvas: dict, backend: Backend, **kwargs):
        outcomes = dict()
        for r in canvas:
            round_outcome = canvas[r].delegate(backend=backend, **kwargs)
            if round_outcome == {}:
                outcomes[r] = backend.state
            else:
                outcomes[r] = round_outcome
            # Ugly reset of backend. Needed in case of quantum output
            # TODO: how to do that cleaner ?
            backend = backend.__class__()
        return outcomes

    def analyze_outcomes(self, canvas: dict, outcomes: dict):
        result_analysis = ResultAnalysis(nr_failed_test_rounds=0, computation_outcomes_count=dict())
        for r in canvas:
            canvas[r].analyze(result_analysis=result_analysis, round_outcomes=outcomes[r])

        # True if Accept, False if Reject
        decision = result_analysis.nr_failed_test_rounds <= self.trappifiedScheme.params.threshold

        # Compute majority vote
        biased_outcome = [
            k
            for k, v in result_analysis.computation_outcomes_count.items()
            if v >= ceil(self.trappifiedScheme.params.comp_rounds / 2)
        ]
        final_outcome = biased_outcome[0] if biased_outcome else None

        return decision, final_outcome

    def decode_output_state(self, backend: Backend):
        for node in self.output_nodes:
            z_decoding, x_decoding = self.decode_output(node)
            if z_decoding:
                backend.apply_single(node=node, op=Ops.Z)
            if x_decoding:
                backend.apply_single(node=node, op=Ops.X)

    def get_secrets_size(self):
        secrets_size = {}
        for secret in self.secret_datas:
            secrets_size[secret] = len(self.secret_datas[secret])
        return secrets_size

    def decode_output(self, node):
        z_decoding = sum(self.results[z_dep] for z_dep in self.byproduct_db[node].z_domain) % 2
        z_decoding ^= self.secret_datas.r.get(node, 0)
        x_decoding = sum(self.results[x_dep] for x_dep in self.byproduct_db[node].x_domain) % 2
        x_decoding ^= self.secret_datas.a.a.get(node, 0)
        return z_decoding, x_decoding


class CircuitUtils:
    def tableau_to_pattern(tableau: stim.Tableau) -> graphix.Pattern:
        n = len(tableau)
        circuit = tableau.to_circuit()
        graphix_circuit = graphix.Circuit(n)
        for i in circuit:
            if i.name == "H":
                for t in i.target_groups():
                    qubit = t[0].value
                    graphix_circuit.h(qubit)
            if i.name == "S":
                for t in i.target_groups():
                    qubit = t[0].value
                    graphix_circuit.s(qubit)
            if i.name == "CX":
                for t in i.target_groups():
                    ctrl, targ = t[0].value, t[1].value
                    graphix_circuit.cnot(control=ctrl, target=targ)
        return graphix_circuit.transpile().pattern


class ClientPrepareMethod(PrepareMethod):
    def __init__(self, preparation_bank):
        self.__preparation_bank = preparation_bank

    def prepare_node(self, backend: Backend, node: int) -> None:
        """Prepare a node."""
        backend.add_nodes(nodes=[node], data=self.__preparation_bank[node])

    def prepare(self, backend: Backend, cmd: BaseN) -> None:
        """Prepare a node."""
        self.prepare_node(backend, cmd.node)


class ClientMeasureMethod(MeasureMethod):
    def __init__(self, client: Client):
        self.__client = client

    def get_measurement_description(self, cmd: BaseM) -> Measurement:
        parameters = self.__client.measurement_db[cmd.node]

        # Extract secrets from Client
        r_value = self.__client.secret_datas.r.get(cmd.node, 0) if cmd.node not in self.__client.output_nodes else 0
        theta_value = self.__client.secret_datas.theta.get(cmd.node, 0)
        a_value = self.__client.secret_datas.a.a.get(cmd.node, 0)
        a_N_value = self.__client.secret_datas.a.a_N.get(cmd.node, 0)

        # Extract signals and compute the angle for the computation
        s_signal = sum(self.__client.results[j] for j in parameters.s_domain)
        t_signal = sum(self.__client.results[j] for j in parameters.t_domain)
        measure_update = MeasureUpdate.compute(parameters.plane, s_signal % 2 == 1, t_signal % 2 == 1, Clifford.I)
        angle = parameters.angle * np.pi
        angle = angle * measure_update.coeff + measure_update.add_term

        # Blind the angle using the Client's secrets
        angle = (-1) ** a_value * angle + theta_value * np.pi / 4 + np.pi * (r_value + a_N_value)
        return Measurement(plane=measure_update.new_plane, angle=angle)

    def get_measure_result(self, node: int) -> bool:
        raise ValueError("Server cannot have access to measurement results")

    def set_measure_result(self, node: int, result: bool) -> None:
        if self.__client.secret_datas.r:
            result ^= self.__client.secret_datas.r[node]
        self.__client.results[node] = result


class TestMeasureMethod(MeasureMethod):
    def __init__(self, client: Client):
        self.__client = client

    def get_measurement_description(self, cmd: BaseM) -> Measurement:
        # Extract secrets from Client
        r_value = self.__client.secret_datas.r.get(cmd.node, 0)
        theta_value = self.__client.secret_datas.theta.get(cmd.node, 0)
        a_N_value = self.__client.secret_datas.a.a_N.get(cmd.node, 0)

        # Blind the angle using the Client's secrets
        angle = theta_value * np.pi / 4 + np.pi * (r_value + a_N_value)
        return Measurement(plane=Plane.XY, angle=angle)

    def get_measure_result(self, node: int) -> bool:
        raise ValueError("Server cannot have access to measurement results")

    def set_measure_result(self, node: int, result: bool) -> None:
        if self.__client.secret_datas.r:
            result ^= self.__client.secret_datas.r[node]
        self.__client.results[node] = result


## TODO: factoriser la measuremethod?
