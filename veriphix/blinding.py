from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from graphix.fundamentals import ANGLE_PI
from graphix.measurements import outcome
from graphix.pauli import Pauli
from graphix.rng import ensure_rng
from graphix.sim.statevec import StatevectorBackend

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    import networkx as nx
    import numpy.typing as npt
    from graphix.fundamentals import Angle
    from graphix.measurements import Outcome
    from graphix.sim.statevec import Statevec
    from graphix.states import State
    from numpy.random import Generator


@dataclass
class Secret_a:
    a: dict[int, Outcome]
    a_N: dict[int, Outcome]


@dataclass
class Secrets:
    r: bool = True
    a: bool = True
    theta: bool = True


@dataclass
class SecretDatas:
    r: dict[int, Outcome]
    a: Secret_a
    theta: dict[int, int]

    @staticmethod
    def from_secrets(
        secrets: Secrets,
        graph: nx.Graph[int],
        input_nodes: AbstractSet[int],
        output_nodes: AbstractSet[int],
        rng: Generator | None = None,
        *,
        stacklevel: int = 1,
    ) -> SecretDatas:
        rng = ensure_rng(rng, stacklevel=stacklevel + 1)
        r = {}
        if secrets.r:
            # Need to generate the random bit for each measured qubit, 0 for the rest (output qubits)
            for node in graph.nodes:
                r[node] = outcome(rng.integers(2) == 1) if node not in output_nodes else 0

        theta = {}
        if secrets.theta:
            # Create theta secret for all non-output nodes (measured qubits)
            for node in graph.nodes:
                theta[node] = int(rng.integers(0, 8)) if node not in output_nodes else 0  # Expressed in pi/4 units
                ## TODO:
        a: dict[int, Outcome] = {}
        a_N: dict[int, Outcome] = {}
        if secrets.a:
            # Create `a` secret for all
            # order is Z(theta) X |+>
            for node in graph.nodes:
                a[node] = outcome(rng.integers(0, 2) == 1) if node in input_nodes else 0

            # After all the `a` secrets have been generated, the `a_N` value can be
            # computed from the graph topology
            for i in graph.nodes:
                a_N_value = outcome(sum([a[j] for j in graph.neighbors(i)]) % 2 == 1)
                # for j in graph.neighbors(i):
                #     a_N_value ^= a[j]
                a_N[i] = a_N_value

        return SecretDatas(r, Secret_a(a, a_N), theta)

    def blind_angle(self, node: int, output_node: bool, test: bool) -> Angle:
        r_value = 0 if (output_node and not test) else self.r.get(node, 0)
        theta_value = self.theta.get(node, 0)
        a_N_value = self.a.a_N.get(node, 0)
        return theta_value * ANGLE_PI / 4 + ANGLE_PI * (r_value ^ a_N_value)

    def blind_qubit(self, node: int, state: State) -> Statevec:
        def z_rotation(theta: float) -> npt.NDArray[np.complex128]:
            return np.array([[1, 0], [0, np.exp(1j * theta * np.pi / 4)]], dtype=np.complex128)

        def x_blind(a: Outcome) -> Pauli:
            return Pauli.X if a == 1 else Pauli.I

        theta = self.theta.get(node, 0)
        a = self.a.a.get(node, 0)
        single_qubit_backend = StatevectorBackend()
        single_qubit_backend.add_nodes([0], [state])
        if a:
            single_qubit_backend.apply_single(node=0, op=x_blind(a).matrix)
        if theta:
            single_qubit_backend.apply_single(node=0, op=z_rotation(theta))
        return single_qubit_backend.state
