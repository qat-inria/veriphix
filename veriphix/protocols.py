from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from veriphix.verifying import TestRun
import itertools
from collections.abc import Sequence
import networkx as nx
import random
import stim
from itertools import combinations
import numpy as np


if TYPE_CHECKING:
    from veriphix.client import Client

class VerificationProtocol(ABC):
    def __init__(self, client) -> None:
        self.client = client
        pass

    @abstractmethod
    def create_test_runs(self, client, **kwargs) -> list[TestRun]:
        pass

class FK12(VerificationProtocol):
    def __init__(self, client) -> None:
        super().__init__(client)
    
    def create_test_runs(self, manual_colouring: Sequence[set[int]] | None = None) -> list[TestRun]:
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
            coloring = nx.coloring.greedy_color(self.client.graph, strategy="largest_first")
            colors = set(coloring.values())
            nodes_by_color: dict[int, list[int]] = {c: [] for c in colors}
            for node in sorted(self.client.graph.nodes):
                color = coloring[node]
                nodes_by_color[color].append(node)
        else:
            # checks that manual_colouring is a proper colouring
            ## first check uniion is the whole graph
            color_union = set().union(*manual_colouring)
            if not color_union == set(self.client.graph.nodes):
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
            test_run = TestRun(client=self.client, traps=traps)
            test_runs.append(test_run)

        # print(test_runs)
        return test_runs

    

class RandomTraps(VerificationProtocol):
    """
    A bad and naive way of generating traps, but exposing the modularity of the interface.
    """
    def __init__(self, client) -> None:
        super().__init__(client)
    
    def create_test_runs(self, **kwargs) -> list[TestRun]:
        test_runs = []
        # Create 1 random trap per node
        n = len(self.client.graph.nodes)
        for _ in range(n):
            # Choose a random subset of nodes to create a trap (random size, random nodes)
            trap_size = random.choice(range(n))
            random_nodes = random.sample(self.client.nodes_list, k=trap_size)
            # Create a single-trap test round from it. The trap is multi-qubit.
            random_multi_qubit_trap = tuple(random_nodes)
            # Only one trap
            traps = (random_multi_qubit_trap,)
            test_run = TestRun(client=self.client, traps=traps)
            test_runs.append(test_run)

        return test_runs

class Dummyless(VerificationProtocol):
    def __init__(self, client) -> None:
        super().__init__(client)
    
    def create_test_runs(self, **kwargs) -> list[TestRun]:        
        G:nx.Graph = self.client.graph
        nodes = list(G.nodes)
        n = len(nodes)

        S_paulis = generate_graph_stabilizers(G)

        # Construct symplectic matrix: 2n × n
        S_bin = np.array([pauli_to_symplectic(p) for p in S_paulis]).T

        R_gens = generate_stabilizers_I_X_Y_only(G)

        test_runs = []
        for _, R in enumerate(R_gens):
            R_bin = pauli_to_symplectic(R)
            coeffs = gf2_solve(S_bin, R_bin)
            support = [nodes[j] for j in range(n) if coeffs[j] == 1]
            test_run = TestRun(client=self.client, traps=(tuple(support),))
            test_runs.append(test_run)

        return test_runs
    
# === GF(2) Solver ===
def gf2_solve(A, b):
    A = A.copy() % 2
    b = b.copy() % 2
    n_rows, n_cols = A.shape
    x = np.zeros(n_cols, dtype=int)
    pivot_rows = [-1] * n_rows

    row = 0
    for col in range(n_cols):
        for r in range(row, n_rows):
            if A[r, col] == 1:
                A[[row, r]] = A[[r, row]]
                b[[row, r]] = b[[r, row]]
                break
        else:
            continue
        for r in range(row + 1, n_rows):
            if A[r, col] == 1:
                A[r] = (A[r] + A[row]) % 2
                b[r] = (b[r] + b[row]) % 2
        pivot_rows[row] = col
        row += 1

    for r in reversed(range(row)):
        col = pivot_rows[r]
        x[col] = b[r]
        for k in range(r):
            if A[k, col] == 1:
                b[k] = (b[k] + x[col]) % 2
    return x


# === Pauli utility ===
def to_binary_XY_support(pstring: stim.PauliString) -> tuple[int]:
    return tuple(1 if p in ['X', 'Y'] else 0 for p in str(pstring))
def pauli_to_symplectic(p: stim.PauliString) -> np.ndarray:
    # Concatenate Z then X bits (Stim convention)
    xs, zs = p.to_numpy()
    return np.concatenate([zs.astype(int), xs.astype(int)])


# === Construct graph stabilizers (S_v) ===
def generate_graph_stabilizers(G: nx.Graph) -> list[stim.PauliString]:
    n = G.number_of_nodes()
    idx_map = {v: i for i, v in enumerate(G.nodes)}
    S = []
    for v in G.nodes:
        pauli = ['I'] * n
        pauli[idx_map[v]] = 'X'
        for u in G.neighbors(v):
            pauli[idx_map[u]] = 'Z'
        S.append(stim.PauliString(''.join(pauli)))
    return S

# === Generate Z-free stabilizers (candidates) ===
def generate_stabilizers_I_X_Y_only(G: nx.Graph) -> list[stim.PauliString]:
    n = G.number_of_nodes()
    idx_map = {v: i for i, v in enumerate(G.nodes)}
    nodes = list(G.nodes)

    S = generate_graph_stabilizers(G)

    R_full = stim.PauliString("I" * n)
    for stab in S:
        R_full *= stab

    candidates = []

    # Even-degree node flips
    even_nodes = [v for v in nodes if G.degree[v] % 2 == 0]
    for v in even_nodes:
        Rv = R_full * S[idx_map[v]]
        if Rv.pauli_indices("Z") == []:
            candidates.append(Rv)

    # Odd-degree pairs connected by even-degree interior paths
    odd_nodes = [v for v in nodes if G.degree[v] % 2 == 1]
    for u, w in combinations(odd_nodes, 2):
        try:
            path = nx.shortest_path(G, u, w)
        except:
            continue
        if all(G.degree[v] % 2 == 0 for v in path[1:-1]):
            Ruw = R_full.copy()
            for v in path:
                Ruw *= S[idx_map[v]]
            if Ruw.pauli_indices("Z") == []:
                candidates.append(Ruw)

    # Greedily pick n - 1 linearly independent (based on XY support)
    basis = []
    basis_bin = []
    for cand in candidates:
        vec = to_binary_XY_support(cand)
        if vec not in basis_bin:
            basis.append(cand)
            basis_bin.append(vec)
        if len(basis) == n - 1:
            break
    return basis