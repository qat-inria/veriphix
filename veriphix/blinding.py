from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from graphix.pauli import Pauli
from graphix.sim.statevec import StatevectorBackend

if TYPE_CHECKING:
    import networkx as nx


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

    def blind_angle(secret_datas, node: int, output_node: bool, test: bool) -> float:
        r_value = 0 if (output_node and not test) else secret_datas.r.get(node, 0)
        theta_value = secret_datas.theta.get(node, 0)
        a_N_value = secret_datas.a.a_N.get(node, 0)
        return theta_value * np.pi / 4 + np.pi * (r_value ^ a_N_value)

    def blind_qubit(secret_datas, node: int, state) -> None:
        def z_rotation(theta) -> np.array:
            return np.array([[1, 0], [0, np.exp(1j * theta * np.pi / 4)]])

        def x_blind(a) -> Pauli:
            return Pauli.X if (a == 1) else Pauli.I

        theta = secret_datas.theta.get(node, 0)
        a = secret_datas.a.a.get(node, 0)
        single_qubit_backend = StatevectorBackend()
        single_qubit_backend.add_nodes([0], [state])
        if a:
            single_qubit_backend.apply_single(node=0, op=x_blind(a).matrix)
        if theta:
            single_qubit_backend.apply_single(node=0, op=z_rotation(theta))
        return single_qubit_backend.state
