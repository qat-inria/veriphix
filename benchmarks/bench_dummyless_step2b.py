"""
Benchmark: exhaustive combinations vs BFS spanning tree for step 2b of Dummyless.
Run with:  python benchmarks/bench_dummyless_step2b.py
"""

from __future__ import annotations

import time
from functools import reduce
from operator import mul

import numpy as np
import stim
from graphix.random_objects import rand_circuit

from veriphix.client import Client
from veriphix.protocols import Dummyless, GraphStabilizer, odd_pair_generators_bfs, odd_pair_generators_exhaustive
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
        g_exhaustive = odd_pair_generators_exhaustive(client.graph, stabdict, rfull)
    t_exhaustive = (time.perf_counter() - t0) / repeats

    t0 = time.perf_counter()
    for _ in range(repeats):
        g_bfs = odd_pair_generators_bfs(client.graph, stabdict, rfull)
    t_bfs = (time.perf_counter() - t0) / repeats

    print(f"  exhaustive : {t_exhaustive * 1000:.2f} ms  ({len(g_exhaustive)} generators)")
    print(f"  BFS        : {t_bfs * 1000:.2f} ms  ({len(g_bfs)} generators)")
    print(f"  speedup    : {t_exhaustive / t_bfs:.1f}x")


if __name__ == "__main__":
    for nqubits, depth in [(2, 2), (3, 4), (4, 6), (5, 8), (6, 10), (10, 15)]:
        benchmark(nqubits, depth, repeats=10)
