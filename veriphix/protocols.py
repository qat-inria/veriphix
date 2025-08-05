from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from veriphix.verifying import TestRun
import itertools
from collections.abc import Sequence
import networkx as nx
import random

if TYPE_CHECKING:
    from veriphix.client import Client

class VerificationProtocol(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def create_test_runs(self, client, **kwargs) -> list[TestRun]:
        pass

class FK12(VerificationProtocol):
    def __init__(self) -> None:
        super().__init__()
    
    def create_test_runs(self, client, manual_colouring: Sequence[set[int]] | None = None) -> list[TestRun]:
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
            coloring = nx.coloring.greedy_color(client.graph, strategy="largest_first")
            colors = set(coloring.values())
            nodes_by_color: dict[int, list[int]] = {c: [] for c in colors}
            for node in sorted(client.graph.nodes):
                color = coloring[node]
                nodes_by_color[color].append(node)
        else:
            # checks that manual_colouring is a proper colouring
            ## first check uniion is the whole graph
            color_union = set().union(*manual_colouring)
            if not color_union == set(client.graph.nodes):
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
            test_run = TestRun(client=client, traps=traps)
            test_runs.append(test_run)

        # print(test_runs)
        return test_runs

    

class RandomTraps(VerificationProtocol):
    """
    A bad and naive way of generating traps, but exposing the modularity of the interface.
    """
    def __init__(self) -> None:
        super().__init__()
    
    def create_test_runs(self, client, **kwargs) -> list[TestRun]:
        test_runs = []
        # Create 1 random trap per node
        n = len(client.graph.nodes)
        for _ in range(n):
            # Choose a random subset of nodes to create a trap (random size, random nodes)
            trap_size = random.choice(range(n))
            random_nodes = random.sample(client.nodes_list, k=trap_size)
            # Create a single-trap test round from it. The trap is multi-qubit.
            random_multi_qubit_trap = tuple(random_nodes)
            # Only one trap
            traps = (random_multi_qubit_trap,)
            test_run = TestRun(client=client, traps=traps)
            test_runs.append(test_run)

        return test_runs

class Dummyless(VerificationProtocol):
    def __init__(self) -> None:
        super().__init__()
    
    def create_test_runs(self, client, **kwargs) -> list[TestRun]:
        return super().create_test_runs(client, **kwargs)