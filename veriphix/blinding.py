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
    """The ``X``-Pauli encryption secrets.

    ``a`` is the per-input-node ``X`` encryption (only input nodes carry it,
    since ``X|+> = |+>`` makes it useless on the default-prepared nodes). ``a_N``
    is the induced ``X`` encryption seen at each node, computed from the parity of
    the ``a`` secrets of its neighbours once the entangling ``CZ``\\ s are applied.
    """

    a: dict[int, Outcome]
    a_N: dict[int, Outcome]


@dataclass
class Secrets:
    """Toggles for which blinding secrets are generated.

    - ``r``: the one-time-pad applied to every node, so that every result
      (including a quantum output) is masked.
    - ``a``: the ``X``-Pauli encryption, only meaningful on input nodes.
    - ``theta``: the ``Z``-rotation encryption applied to every measured node.
    """

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
        unmeasured_nodes: AbstractSet[int],
        rng: Generator | None = None,
        *,
        stacklevel: int = 1,
    ) -> SecretDatas:
        """Sample the blinding secrets for the whole graph.

        ``unmeasured_nodes`` should be empty for a classical output and the set of
        quantum output nodes otherwise: it is exactly the set of *unmeasured*
        nodes, which therefore receive no ``theta`` encryption.
        """
        rng = ensure_rng(rng, stacklevel=stacklevel + 1)
        r = {}
        if secrets.r:
            # `r` is a one-time-pad on every node: every result, including a
            # quantum output, must be masked, so we draw a random bit per qubit.
            for node in graph.nodes:
                r[node] = outcome(rng.integers(2) == 1)

        theta = {}
        if secrets.theta:
            # `theta` (a Z-rotation, in pi/4 units) is only applied to measured
            # nodes, i.e. all nodes except the quantum output ones. Unmeasured
            # output nodes get theta = 0.
            for node in graph.nodes:
                theta[node] = int(rng.integers(0, 8)) if node not in unmeasured_nodes else 0  # Expressed in pi/4 units
        a: dict[int, Outcome] = {}
        a_N: dict[int, Outcome] = {}
        if secrets.a:
            # `a` is the X-Pauli encryption. Only input nodes get it: the other
            # nodes are prepared in |+> and X|+> = |+>, so it would be a no-op.
            # The blinding order on a node is Z(theta) X |+>.
            for node in graph.nodes:
                a[node] = outcome(rng.integers(0, 2) == 1) if node in input_nodes else 0

            # Once all `a` secrets are known, the induced encryption `a_N` at each
            # node is the parity of its neighbours' `a` (the X Paulis propagated
            # through the entangling CZs).
            for i in graph.nodes:
                a_N_value = outcome(sum([a[j] for j in graph.neighbors(i)]) % 2 == 1)
                a_N[i] = a_N_value

        return SecretDatas(r, Secret_a(a, a_N), theta)

    def blind_angle(self, node: int) -> Angle:
        theta_value = self.theta.get(node, 0)
        return theta_value * ANGLE_PI / 4

    def blind_qubit(self, node: int, state: State) -> Statevec:
        """Apply the secret-dependent blinding to a single prepared qubit.

        The unblinded ``state`` is the secret-independent computation state (the
        input state for input nodes, ``|+>`` otherwise). The blinding applies, in
        order, ``Z(theta) Z(r) X(a)`` on top of it: the ``X`` encryption ``a``
        (input nodes only), the ``Z`` one-time-pad ``r``, then the ``theta``
        Z-rotation.
        """

        def z_rotation(theta: float) -> npt.NDArray[np.complex128]:
            return np.array([[1, 0], [0, np.exp(1j * theta * np.pi / 4)]], dtype=np.complex128)

        def x_blind(a: Outcome) -> Pauli:
            return Pauli.X if a == 1 else Pauli.I

        def z_blind(r: Outcome) -> Pauli:
            return Pauli.Z if r == 1 else Pauli.I

        theta = self.theta.get(node, 0)
        x_blind_value = self.a.a.get(node, 0)
        r = self.r.get(node, 0)
        single_qubit_backend = StatevectorBackend()
        single_qubit_backend.add_nodes([0], [state])
        if x_blind_value:
            single_qubit_backend.apply_single(node=0, op=x_blind(x_blind_value).matrix)
        if r:
            single_qubit_backend.apply_single(node=0, op=z_blind(r).matrix)
        if theta:
            single_qubit_backend.apply_single(node=0, op=z_rotation(theta))
        return single_qubit_backend.state
