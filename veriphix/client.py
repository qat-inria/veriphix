from __future__ import annotations

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
from graphix.sim.statevec import Statevec
from graphix.simulator import MeasureMethod, PrepareMethod
from graphix.states import BasicStates
from stim import Circuit

from veriphix.blinding import SecretDatas, Secrets
from veriphix.verifying import (
    ComputationRun,
    ResultAnalysis,
    Run,
    RunResult,
    TrappifiedScheme,
    TrappifiedSchemeParameters,
)
from veriphix.protocols import FK12, VerificationProtocol

from veriphix.malicious_noise_model import MaliciousNoiseModel

if TYPE_CHECKING:
    from graphix.sim.base_backend import Backend


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
    # Généraliser: le client prend un prédicat de 'output' à booleen, par exemple qubit 0 doit renvoyer 0 
    
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
        protocol_cls: type[VerificationProtocol]=FK12,
        **kwargs,
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

        self.computationRun = ComputationRun(self)
        protocol = protocol_cls(client=self)
        
        self.test_runs = protocol.create_test_runs(**kwargs)

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
                state = BasicStates.PLUS if r_value ^ a_N_value == 0 else BasicStates.MINUS

            else:
                state = BasicStates.PLUS
            states[node] = state
        return states

    def prepare_states_virtual(self, states_dict: dict[(int, BasicStates)]) -> None:
        """
        The Client creates the qubits and blind them in its preparation_bank
        """
        for node in states_dict:
            blinded_qubit_state = self.secret_datas.blind_qubit(node=node, state=states_dict[node])
            self.preparation_bank[node] = Statevec(blinded_qubit_state)

    def prepare_states(self, backend: Backend, states_dict: dict[(int, BasicStates)]) -> None:
        # Initializes the bank (all the nodes)
        self.prepare_states_virtual(states_dict=states_dict)
        # Server asks the backend to create them
        ## Except for the input! The Client creates them itself
        for node in self.input_nodes:
            self.prepare_method.prepare_node(backend, node)

    def sample_canvas(self):
        N = self.trappifiedScheme.params.comp_rounds + self.trappifiedScheme.params.test_rounds
        computation_rounds = set(random.sample(range(N), self.trappifiedScheme.params.comp_rounds))

        return {r: self.computationRun if r in computation_rounds else random.choice(self.test_runs) for r in range(N)}

    def delegate_canvas(self, canvas: dict[int, Run], backend_cls: type[Backend], **kwargs) -> dict[int, RunResult]:
        outcomes = dict()
        noise_model = kwargs.get("noise_model")
        for r in canvas:
            backend = backend_cls()
            if isinstance(noise_model, MaliciousNoiseModel):
                noise_model.refresh_randomness()
            outcomes[r] = canvas[r].delegate(backend=backend, **kwargs)
        return outcomes


    # TODO: fix bug in case of 'desired_outputs'
    """
    TODO: généralisation.
    Sommer sur les rounds de calcul dont l'output vérifie le prédicat
    regarder si cette somme est > d/2
    """
    def analyze_outcomes(self, canvas, outcomes: dict[int, RunResult]) -> tuple[bool, str, ResultAnalysis]:
        result_analysis = ResultAnalysis(
            nr_failed_test_rounds=0, computation_outcomes_count=dict(), quantum_output_states={}
        )
        for r in canvas:
            outcomes[r].analyze(result_analysis=result_analysis, client=self)

        # True if Accept, False if Reject
        decision = result_analysis.nr_failed_test_rounds <= self.trappifiedScheme.params.threshold

        # Compute majority vote (only if classical output)
        biased_outcome = (
            [
                k
                for k, v in result_analysis.computation_outcomes_count.items()
                if v >= ceil(self.trappifiedScheme.params.comp_rounds / 2)
            ]
            if self.classical_output
            else None
        )
        final_outcome = biased_outcome[0] if biased_outcome else None

        return decision, final_outcome, result_analysis

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
        a_value = self.__client.secret_datas.a.a.get(cmd.node, 0)

        # Extract signals and compute the angle for the computation
        s_signal = sum(self.__client.results[j] for j in parameters.s_domain)
        t_signal = sum(self.__client.results[j] for j in parameters.t_domain)
        measure_update = MeasureUpdate.compute(parameters.plane, s_signal % 2 == 1, t_signal % 2 == 1, Clifford.I)
        angle = parameters.angle * np.pi
        angle = angle * measure_update.coeff + measure_update.add_term

        # Blind the angle using the Client's secrets
        angle = (-1) ** a_value * angle + self.__client.secret_datas.blind_angle(
            cmd.node, cmd.node in self.__client.output_nodes, test=False
        )
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
        # Blind the angle using the Client's secrets
        angle = self.__client.secret_datas.blind_angle(cmd.node, cmd.node in self.__client.output_nodes, test=True)

        return Measurement(plane=Plane.XY, angle=angle)

    def get_measure_result(self, node: int) -> bool:
        raise ValueError("Server cannot have access to measurement results")

    def set_measure_result(self, node: int, result: bool) -> None:
        if self.__client.secret_datas.r:
            result ^= self.__client.secret_datas.r[node]
        self.__client.results[node] = result
