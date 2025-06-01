from __future__ import annotations

import itertools
from dataclasses import dataclass
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
from graphix.measurements import Measurement
from graphix.ops import Ops
from graphix.pattern import Pattern
from graphix.pauli import Pauli
from graphix.sim.statevec import Statevec, StatevectorBackend
from graphix.simulator import MeasureMethod, PatternSimulator, PrepareMethod
from graphix.states import BasicStates
from stim import Tableau, Circuit


if TYPE_CHECKING:
    from collections.abc import Sequence

    import stim
    from graphix.sim.base_backend import Backend

Trap=tuple[int]

# TODO update docstring
"""
Usage:

client = Client(pattern:Pattern, blind=False) ## For pure MBQC
sv_backend = StatevecBackend(client.pattern, meas_op = client.meas_op)

simulator = PatternSimulator(client.pattern, backend=sv_backend)
simulator.run()

"""


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

def get_graph_clifford_structure(graph:nx.Graph):
    circuit = Circuit()
    for edge in graph.edges:
        i, j = edge
        circuit.append_from_stim_program_text(f"CZ {i} {j}")
    return circuit.to_tableau()

class Client:
    def __init__(self, pattern, input_state=None, measure_method_cls=None, test_measure_method_cls = None, secrets: None | Secrets = None) -> None:
        self.initial_pattern: Pattern = pattern

        self.input_nodes = self.initial_pattern.input_nodes.copy()
        self.output_nodes = self.initial_pattern.output_nodes.copy()
        graph = self.initial_pattern.get_graph()
        self.graph = nx.Graph()
        self.graph.add_nodes_from(graph[0])
        self.graph.add_edges_from(graph[1])
        self.clifford_structure = get_graph_clifford_structure(self.graph)

        self.nodes_list = list(self.graph.nodes)

        # Copy the pauli-preprocessed nodes' measurement outcomes
        self.results = pattern.results.copy()
        if measure_method_cls is None:
            measure_method_cls = ClientMeasureMethod
        if test_measure_method_cls is None:
            test_measure_method_cls = TestMeasureMethod

        self.measure_method = measure_method_cls(self)
        self.test_measure_method = test_measure_method_cls(self)

        # careful: 'get_measurement_commands' standardizes the pattern
        # but we want the measurement commands as if the pattern was standardized (that means, include the byproducts absorbed in the measurement as much as possible)
        # So we copy the pattern so we can standardize it without touching the non-standardized version
        pattern_copy = Pattern(pattern.input_nodes)
        for cmd in pattern:
            pattern_copy.add(cmd)

        self.measurement_db = {measure.node: measure for measure in pattern_copy.get_measurement_commands()}

        self.byproduct_db = get_byproduct_db(pattern_copy)

        # self.secrets_bool : bool -> self.secrets is not None
        # self.secrets_type : Secrets -> self.secrets
        # self.secrets : SecretDatas-> self.secret_datas
        self.secrets = secrets
        if secrets is None:
            self.secrets_bool = False
            secrets = Secrets()

        self.secret_datas = SecretDatas.from_secrets(secrets, self.graph, self.input_nodes, self.output_nodes)


        # Remove informations from the pattern, leaves only the graph structure, and order of measurements/creation of qubits
        self.clean_pattern = remove_flow(self.initial_pattern)

        self.input_state = input_state if input_state is not None else [BasicStates.PLUS for _ in self.input_nodes]
        # Creates the states to be used during the protocol
        self.computation_states = self.get_computation_states()

        self.preparation_bank = {}
        self.prepare_method = ClientPrepareMethod(self.preparation_bank)

    def refresh_randomness(self) -> None:
        "method to refresh random randomness using parameters from Clinent instatiation."

        # refresh only if secrets bool is True; False is no randomness at all.
        if self.secrets is not None:
            self.secret_datas = SecretDatas.from_secrets(self.secrets, self.graph, self.input_nodes, self.output_nodes)
    

    def get_computation_states(self) :
        states = dict()
        for node in self.nodes_list:
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
    
    def prepare_states_virtual(self, states_dict:dict[(int, BasicStates)], backend: Backend) -> None:
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


    def prepare_states(self, backend: Backend, states_dict:dict[(int, BasicStates)]) -> None:
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
        test_runs:list[TestRun] = []
        for color in colors:
            # 1 color = 1 test run = 1 collection of single-qubit traps
            traps_list = [(colored_node,) for colored_node in nodes_by_color[color]]
            traps = tuple(traps_list)
            test_run = TestRun(client=self, traps=traps)
            test_runs.append(test_run)

        # print(test_runs)
        return test_runs

        

    def delegate_pattern(self, backend: Backend, **kwargs) -> None:
        # Initializes the bank & asks backend to create the input
        self.prepare_states(backend, states_dict=self.computation_states)

        sim = PatternSimulator(
            backend=backend,
            pattern=self.clean_pattern,
            prepare_method=self.prepare_method,
            measure_method=self.measure_method,
            **kwargs,
        )
        sim.run(input_state=None)
        self.decode_output_state(backend)

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
        r_value = self.__client.secret_datas.r.get(cmd.node, 0)
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
        parameters = self.__client.measurement_db[cmd.node]

        # Extract secrets from Client
        r_value = self.__client.secret_datas.r.get(cmd.node, 0)
        theta_value = self.__client.secret_datas.theta.get(cmd.node, 0)
        a_value = self.__client.secret_datas.a.a.get(cmd.node, 0)
        a_N_value = self.__client.secret_datas.a.a_N.get(cmd.node, 0)


        # Blind the angle using the Client's secrets
        angle = theta_value * np.pi / 4 + np.pi * (r_value + a_N_value)
        return Measurement(plane=parameters.plane, angle=angle)

    def get_measure_result(self, node: int) -> bool:
        raise ValueError("Server cannot have access to measurement results")

    def set_measure_result(self, node: int, result: bool) -> None:
        if self.__client.secret_datas.r:
            result ^= self.__client.secret_datas.r[node]
        self.__client.results[node] = result
