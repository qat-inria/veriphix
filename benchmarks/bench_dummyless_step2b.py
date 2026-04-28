"""
Benchmark: exhaustive combinations vs BFS spanning tree for step 2b of Dummyless.
Run with:  python benchmarks/bench_dummyless_step2b.py
"""

from __future__ import annotations

import itertools
import time
from functools import reduce
from operator import mul

import networkx as nx
import numpy as np
import stim
from graphix.random_objects import rand_circuit

from veriphix.client import Client
from veriphix.protocols import Dummyless, GraphStabilizer
from veriphix.verifying import TestRun


def _build_stabdict(client, identity: GraphStabilizer) -> tuple[dict[int, GraphStabilizer], GraphStabilizer]:
    stabdict: dict[int, GraphStabilizer] = {
        node: GraphStabilizer(
            node_indices={node},
            string=TestRun(client=client, traps=frozenset({frozenset({node})})).stabilizer,
        )
        for node in client.graph.nodes
    }
    rfull = reduce(mul, stabdict.values(), identity)
    return stabdict, rfull


def step2b_exhaustive(client, stabdict, rfull, identity) -> list[GraphStabilizer]:
    odd_nodes = [v for v in client.graph.nodes if client.graph.degree(v) % 2 == 1]
    generators = []
    for u, w in itertools.combinations(odd_nodes, 2):
        try:
            path = nx.shortest_path(client.graph, u, w)
        except nx.NetworkXNoPath:
            continue
        if not all(client.graph.degree(v) % 2 == 0 for v in path[1:-1]):
            continue
        candidate = reduce(mul, (stabdict[v] for v in path), rfull)
        if candidate.string.pauli_indices("Z") == []:
            generators.append(candidate)
    return generators


def step2b_bfs(client, stabdict, rfull, identity) -> list[GraphStabilizer]:
    odd_nodes_set = {v for v in client.graph.nodes if client.graph.degree(v) % 2 == 1}
    visited: set[int] = set()
    pred: dict[int, int] = {}
    odd_source: dict[int, int] = {}
    generators = []

    for start in odd_nodes_set:
        if start in visited:
            continue
        visited.add(start)
        odd_source[start] = start
        queue: list[int] = [start]
        while queue:
            u = queue.pop(0)
            for w in client.graph.neighbors(u):
                if w in visited:
                    continue
                pred[w] = u
                visited.add(w)
                queue.append(w)
                if w in odd_nodes_set:
                    odd_source[w] = w
                    src = odd_source[u]
                    path: list[int] = []
                    node = w
                    while node != src:
                        path.append(node)
                        node = pred[node]
                    path.append(src)
                    path.reverse()
                    generators.append(reduce(mul, (stabdict[v] for v in path), rfull))
                else:
                    odd_source[w] = odd_source[u]
    return generators


def benchmark(nqubits: int, depth: int, repeats: int = 5) -> None:
    rng = np.random.default_rng(42)
    circuit = rand_circuit(nqubits, depth, rng)
    pattern = circuit.transpile().pattern
    protocol = Dummyless()
    client = Client(pattern=pattern, protocol=protocol, autogen=False, rng=rng)
    client.preprocess_pattern()
    client.create_blind_patterns(rng=rng)

    n_qubits = len(client.clifford_structure)
    identity = GraphStabilizer(node_indices=set(), string=stim.PauliString(n_qubits))
    stabdict, rfull = _build_stabdict(client, identity)

    odd_count = sum(1 for v in client.graph.nodes if client.graph.degree(v) % 2 == 1)
    print(f"\nnqubits={nqubits}, depth={depth} → {client.graph.number_of_nodes()} graph nodes, {odd_count} odd-degree")

    t0 = time.perf_counter()
    for _ in range(repeats):
        g_exhaustive = step2b_exhaustive(client, stabdict, rfull, identity)
    t_exhaustive = (time.perf_counter() - t0) / repeats

    t0 = time.perf_counter()
    for _ in range(repeats):
        g_bfs = step2b_bfs(client, stabdict, rfull, identity)
    t_bfs = (time.perf_counter() - t0) / repeats

    print(f"  exhaustive : {t_exhaustive*1000:.2f} ms  ({len(g_exhaustive)} generators)")
    print(f"  BFS        : {t_bfs*1000:.2f} ms  ({len(g_bfs)} generators)")
    print(f"  speedup    : {t_exhaustive / t_bfs:.1f}x")


if __name__ == "__main__":
    for nqubits, depth in [(2, 2), (3, 4), (4, 6), (5, 8), (6, 10), (10, 15)]:
        benchmark(nqubits, depth, repeats=10)
