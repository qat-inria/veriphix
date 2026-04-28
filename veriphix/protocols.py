from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from array import array
from collections.abc import Callable
from functools import reduce
from operator import mul
from typing import TYPE_CHECKING, TypeAlias

import networkx as nx
import stim
from graphix.rng import ensure_rng
from typing_extensions import override

from veriphix.verifying import TestRun

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypeVar

    from graphix import Pattern
    from numpy.random import Generator

    from veriphix.client import Client

    _StateT = TypeVar("_StateT")


class GraphStabilizer:
    def __init__(self, node_indices: set[int], string: stim.PauliString) -> None:
        self.node_indices = node_indices
        self.string = string

    def __mul__(self, other: GraphStabilizer) -> GraphStabilizer:
        return GraphStabilizer(
            node_indices=self.node_indices ^ other.node_indices,
            string=self.string * other.string,
        )


class VerificationProtocol(ABC):
    @abstractmethod
    def create_test_runs(self, client: Client, rng: Generator | None = None, *, stacklevel: int = 1) -> list[TestRun]:
        pass


class FK12(VerificationProtocol):
    def __init__(self, manual_colouring: Sequence[set[int]] | None = None) -> None:
        super().__init__()
        self.manual_colouring = manual_colouring

    @override
    def create_test_runs(self, client: Client, rng: Generator | None = None, *, stacklevel: int = 1) -> list[TestRun]:
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
        if self.manual_colouring is None:
            coloring = nx.coloring.greedy_color(client.graph, strategy="largest_first")
            colors = set(coloring.values())
            nodes_by_color: dict[int, list[int]] = {c: [] for c in colors}
            for node in sorted(client.nodes):
                color = coloring[node]
                nodes_by_color[color].append(node)
        else:
            # checks that manual_colouring is a proper colouring
            ## first check uniion is the whole graph
            color_union = set().union(*self.manual_colouring)
            if not color_union == set(client.graph.nodes):
                raise ValueError("The provided colouring does not include all the nodes of the graph.")
            # check that colors are two by two disjoint
            # if sets are disjoint, empty set from intersection is interpreted as False.
            # so one non-empty set -> one True value -> use any()
            if any([i & j for i, j in itertools.combinations(self.manual_colouring, 2)]):
                raise ValueError(
                    "The provided colouring is not a proper colouring i.e the same node belongs to at least two colours."
                )

            nodes_by_color = {i: list(c) for i, c in enumerate(self.manual_colouring)}
            colors = set(nodes_by_color.keys())

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


def get_bipartite_coloring(pattern: Pattern) -> tuple[set[int], set[int]]:
    """Return a bipartite coloring for the given pattern."""
    positions = get_node_positions(pattern)
    red = set()
    blue = set()
    for node, position in positions.items():
        if (position[0] + position[1]) % 2:
            red.add(node)
        else:
            blue.add(node)
    return (red, blue)


def get_node_positions(pattern: Pattern, scale: float = 1, reverse_qubit_order: bool = False) -> dict[int, array[int]]:
    """Return node positions in a grid layout."""
    width = len(pattern.input_nodes)
    return {
        node: array(
            "i",
            [
                int((node // width) * scale),
                int((width - node % width if reverse_qubit_order else node % width) * scale),
            ],
        )
        for node in range(pattern.n_node)
    }


class RandomTraps(VerificationProtocol):
    """
    A bad and naive way of generating traps, but exposing the modularity of the interface.
    """

    def __init__(self) -> None:
        super().__init__()

    @override
    def create_test_runs(self, client: Client, rng: Generator | None = None, *, stacklevel: int = 1) -> list[TestRun]:
        rng = ensure_rng(rng, stacklevel=stacklevel + 1)
        test_runs = []
        # Create 1 random trap per node
        n = len(client.graph.nodes)
        for _ in range(n):
            # Choose a random subset of nodes to create a trap (random size, random nodes)
            trap_size = rng.integers(n)
            random_nodes = [client.nodes[i] for i in rng.choice(len(client.nodes), size=trap_size, replace=False)]
            # Create a single-trap test round from it. The trap is multi-qubit.
            random_multi_qubit_trap = frozenset(random_nodes)
            # Only one trap
            traps = frozenset({random_multi_qubit_trap})
            test_run = TestRun(client=client, traps=traps)
            test_runs.append(test_run)

        return test_runs


def _odd_pair_generators_bfs(
    graph: nx.Graph,
    stabdict: dict[int, GraphStabilizer],
    rfull: GraphStabilizer,
) -> list[GraphStabilizer]:
    """Find R\\(u,w) generators for all pairs of odd-degree nodes using a BFS spanning tree.

    Builds a spanning tree of odd-degree nodes by doing BFS over the whole graph.
    Even-degree nodes are traversed transparently; each time the BFS reaches a new
    odd-degree node w from a current odd-degree source src, the path src→…→w (whose
    interior nodes are all even-degree by BFS construction) defines one generator
    R\\(src,w) = Rfull x S_src x … x S_w.

    Produces exactly |odd|-1 generators per connected component, i.e. one per edge of
    the spanning tree of odd-degree nodes. Time complexity: O(|V| + |E|).

    Parameters
    ----------
    graph : nx.Graph
        The resource graph.
    stabdict : dict[int, GraphStabilizer]
        Per-node canonical stabilizers S_v.
    rfull : GraphStabilizer
        Product of all S_v (Rfull).

    Returns
    -------
    list[GraphStabilizer]
        The R\\(u,w) generators, one per spanning-tree edge of odd-degree nodes.
    """
    odd_nodes_set = {v for v in graph.nodes if graph.degree(v) % 2 == 1}
    visited: set[int] = set()
    pred: dict[int, int] = {}
    odd_source: dict[int, int] = {}
    generators: list[GraphStabilizer] = []

    for start in odd_nodes_set:
        if start in visited:
            continue
        visited.add(start)
        odd_source[start] = start
        queue: list[int] = [start]
        while queue:
            u = queue.pop(0)
            for w in graph.neighbors(u):
                if w in visited:
                    continue
                pred[w] = u
                visited.add(w)
                queue.append(w)
                if w in odd_nodes_set:
                    odd_source[w] = w
                    src = odd_source[u]
                    path: list[int] = []
                    node: int = w
                    while node != src:
                        path.append(node)
                        node = pred[node]
                    path.append(src)
                    path.reverse()
                    generators.append(reduce(mul, (stabdict[v] for v in path), rfull))
                else:
                    odd_source[w] = odd_source[u]

    return generators


def _odd_pair_generators_exhaustive(
    graph: nx.Graph,
    stabdict: dict[int, GraphStabilizer],
    rfull: GraphStabilizer,
) -> list[GraphStabilizer]:
    """Find R\\(u,w) generators for all pairs of odd-degree nodes by exhaustive search.

    Tries every pair (u, w) of odd-degree nodes. For each pair, computes the shortest
    path between them and accepts it only if all interior nodes are even-degree. The
    resulting generator R\\(u,w) = Rfull x S_u x … x S_w is further validated by
    checking it contains no Z Paulis.

    This approach is correct but has time complexity O(|odd|² x (|V| + |E|)) and
    produces more generators than necessary (not a minimal spanning set).
    Prefer :func:`_odd_pair_generators_bfs` for large graphs.

    Parameters
    ----------
    graph : nx.Graph
        The resource graph.
    stabdict : dict[int, GraphStabilizer]
        Per-node canonical stabilizers S_v.
    rfull : GraphStabilizer
        Product of all S_v (Rfull).

    Returns
    -------
    list[GraphStabilizer]
        All valid R\\(u,w) generators found.
    """
    odd_nodes = [v for v in graph.nodes if graph.degree(v) % 2 == 1]
    generators: list[GraphStabilizer] = []
    for u, w in itertools.combinations(odd_nodes, 2):
        try:
            path = nx.shortest_path(graph, u, w)
        except nx.NetworkXNoPath:
            continue
        if not all(graph.degree(v) % 2 == 0 for v in path[1:-1]):
            continue
        candidate = reduce(mul, (stabdict[v] for v in path), rfull)
        if candidate.string.pauli_indices("Z") == []:
            generators.append(candidate)
    return generators


OddPairGeneratorFn: TypeAlias = Callable[
    [nx.Graph, dict[int, GraphStabilizer], GraphStabilizer],
    list[GraphStabilizer],
]


class Dummyless(VerificationProtocol):
    def __init__(self, odd_pair_generator: OddPairGeneratorFn = _odd_pair_generators_bfs) -> None:
        super().__init__()
        self.odd_pair_generator = odd_pair_generator

    @override
    def create_test_runs(self, client, rng=None, *, stacklevel=1):
        rng = ensure_rng(rng, stacklevel=stacklevel + 1)

        # Step 0: build per-node GraphStabilizers (dict keyed by node id)
        stabdict: dict[int, GraphStabilizer] = {
            node: GraphStabilizer(
                node_indices={node},
                string=TestRun(client=client, traps=frozenset({frozenset({node})})).stabilizer,
            )
            for node in client.graph.nodes
        }

        # Step 1: Rfull = product of all S_v
        n_qubits = len(client.clifford_structure)
        identity = GraphStabilizer(node_indices=set(), string=stim.PauliString(n_qubits))
        rfull: GraphStabilizer = reduce(mul, stabdict.values(), identity)

        # (removing S_v leaves I at v and flips Z count on its neighbours — stays in I/X/Y)
        # Step 2a: for each even-degree node v, R\v = Rfull * S_v
        generators: list[GraphStabilizer] = [
            rfull * stabdict[v] for v in client.graph.nodes if client.graph.degree(v) % 2 == 0
        ]

        # Step 2b: one R\(u,w) generator per odd-degree node pair
        generators.extend(self.odd_pair_generator(client.graph, stabdict, rfull))

        # Step 3: build TestRuns — each generator's node_indices form one multi-qubit trap
        return [TestRun(client=client, traps=frozenset({frozenset(gs.node_indices)})) for gs in generators]
