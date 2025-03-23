from __future__ import annotations

from typing import TYPE_CHECKING

import stim
from graphix.fundamentals import IXYZ
from graphix.pauli import Pauli
from graphix.rng import ensure_rng

if TYPE_CHECKING:
    import networkx as nx
    from graphix.states import State
    from numpy.random import Generator

    Trap = set[int]


class TrapStabilizers:
    def __init__(self, graph: nx.Graph, traps_list: set[Trap]) -> None:
        self.graph = graph
        self.traps_list = traps_list
        self.__trap_qubits = frozenset(node for trap in self.traps_list for node in trap)
        self.trap_stabilizers = [self.compute_trap_stabilizer(trap) for trap in self.traps_list]
        self.stabilizer = merge_pauli_list(self.trap_stabilizers)

    @property
    def trap_qubits(self) -> frozenset[int]:
        return self.__trap_qubits

    @property
    def dummy_qubits(self):
        return [neighbor for trap in self.traps_list for node in trap for neighbor in list(self.graph.neighbors(node))]

    def get_canonical_stabilizer(self, node: int) -> stim.PauliString:
        chain = stim.PauliString(len(self.graph.nodes))

        chain[node] = "X"
        for n in self.graph.neighbors(node):
            chain[n] = "Z"
        return chain

    def compute_trap_stabilizer(self, trap: set[int]) -> stim.PauliString:
        trap_stabilizer = stim.PauliString(len(self.graph.nodes))
        for node in trap:
            trap_stabilizer *= self.get_canonical_stabilizer(node)
        return trap_stabilizer

    def pick_random_coins_dummies(self, rng: Generator | None = None) -> dict[int, bool]:
        rng = ensure_rng(rng)
        coins = {node: bool(rng.integers(2)) if node not in self.trap_qubits else False for node in self.graph.nodes}
        for node in self.trap_qubits:
            coins[node] = bool(sum(coins[n] for n in self.graph.neighbors(node)) % 2)
        return coins

    def __str__(self) -> str:
        return f"List of traps: {self.traps_list}\nStabilizer: {self.stabilizer}"


def check_common_eigenstate(stabilizer_1: stim.PauliString, stabilizer_2: stim.PauliString) -> bool:
    """
    Returns `True` if the two stabilizers have a common eigenstate
    """
    for axis in ["X", "Y", "Z"]:
        if not set(stabilizer_1.pauli_indices(axis)).issubset(set(stabilizer_2.pauli_indices("I" + axis))):
            return False
    return True


def merge_pauli_strings(stabilizer_1: stim.PauliString, stabilizer_2: stim.PauliString) -> stim.PauliString:
    result = stabilizer_2.sign * stabilizer_1
    # We can iterate through the support of stab2 as they are supposed to have equal support
    for node in stabilizer_2.pauli_indices("XYZ"):
        result[node] = stabilizer_2[node]
    return result


def merge_pauli_list(pauli_list) -> stim.PauliString:
    """
    If the protocol is coherent, all the stabilizers should be able to merge in one Pauli string.
    """
    # Initial flag to track if any merging happens in the iteration
    merged = True

    while merged:
        merged = False
        i = 0

        while i < len(pauli_list) - 1:
            j = i + 1

            while j < len(pauli_list):
                if check_common_eigenstate(pauli_list[i], pauli_list[j]):
                    # Merge the two PauliStrings
                    merged_pauli = merge_pauli_strings(pauli_list[i], pauli_list[j])

                    # Replace the first element with the merged result
                    pauli_list[i] = merged_pauli

                    # Remove the second element
                    del pauli_list[j]

                    # Set merged flag to True to indicate a merge happened
                    merged = True

                    # Break the inner loop since we need to recheck the new list
                    break
                else:
                    j += 1

            if not merged:
                i += 1

    return pauli_list[0]


class TrappifiedCanvas:
    """
    Trappified canvas according to
    Kapourniotis, T., Kashefi, E., Leichtle, D., Music, L., & Ollivier, H. (2024). Unifying quantum verification and error-detection: theory and tools for optimisations. Quantum Science and Technology, 9(3), 035036.
    https://arxiv.org/abs/2206.00631

    A trap is a set of nodes (i_1, ..., i_t) that are used in a verification protocol, to verify that the sum of their outcomes is 0 modulo 2.

    A trappified canvas is specified by a set of traps and the description of a state on which an honest execution of the protocol should yield the expected outcomes for all traps.

    This class uses the stabilizer formalism, to generate from a list of traps, the appropriate input state.

    To one trappified canvas corresponds a list of traps, and a resulting stabilizer expressed in a Pauli string. The input that satisfies all traps is a +1 eigenstate of that operator.
    """

    def __init__(self, stabilizers: TrapStabilizers, rng: Generator | None = None) -> None:
        self.__rng = ensure_rng(rng)
        self.stabilizers = stabilizers
        self.refresh_randomness()

    @property
    def graph(self):
        return self.stabilizers.graph

    @property
    def trap_qubits(self):
        return self.stabilizers.trap_qubits

    @property
    def traps_list(self):
        return self.stabilizers.traps_list

    @property
    def stabilizer(self):
        return self.stabilizers.stabilizer

    def refresh_randomness(self) -> None:
        self.coins = self.stabilizers.pick_random_coins_dummies(rng=self.__rng)
        self.states = self.generate_eigenstate()

    def generate_eigenstate(self) -> list[State]:
        states = []
        for node in sorted(self.graph.nodes):
            operator = Pauli(IXYZ(self.stabilizer[node]))
            states.append(operator.eigenstate(binary=self.coins[node]))
        return states

    def __str__(self) -> str:
        return f"Canvas({self.stabilizers})"
