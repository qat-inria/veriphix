import random

import numpy as np
from graphix.random_objects import rand_circuit
from graphix.sim.statevec import StatevectorBackend
from itertools import combinations

from veriphix.client import Client, Secrets
from veriphix.verifying import TestRun
from stim import PauliString


class TestVerifying:
    def test_delegate_test(self, fx_rng: np.random.Generator):
        nqubits = 3
        depth = 5
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        secrets = Secrets(r=True, a=True, theta=True)
        client = Client(pattern=pattern, secrets=secrets)

        for _ in range(10):
            # Test noiseless trap delegation
            trap_size = random.choice(range(len(client.nodes_list)))
            random_nodes = random.sample(client.nodes_list, k=trap_size)

            random_multi_qubit_trap = tuple(random_nodes)
            # Only one trap
            traps = (random_multi_qubit_trap,)

            test_run = TestRun(client=client, traps=traps)
            backend = StatevectorBackend()
            outcomes = test_run.delegate(backend=backend).trap_outcomes

            for trap in traps:
                assert outcomes[trap] == 0
    def test_dummyless_proof(self, fx_rng:np.random.Generator):

        import networkx as nx
        from itertools import combinations

        def pauli_mult(p1, p2):
            table = {
                ('I', 'I'): 'I', ('I', 'X'): 'X', ('I', 'Y'): 'Y', ('I', 'Z'): 'Z',
                ('X', 'I'): 'X', ('X', 'X'): 'I', ('X', 'Y'): 'Z', ('X', 'Z'): 'Y',
                ('Y', 'I'): 'Y', ('Y', 'X'): 'Z', ('Y', 'Y'): 'I', ('Y', 'Z'): 'X',
                ('Z', 'I'): 'Z', ('Z', 'X'): 'Y', ('Z', 'Y'): 'X', ('Z', 'Z'): 'I'
            }
            return table[(p1, p2)]

        def multiply_stabilizers(s1, s2):
            return [pauli_mult(a, b) for a, b in zip(s1, s2)]

        def is_all_I_X_Y(pauli_string):
            return all(p in ['I', 'X', 'Y'] for p in pauli_string)

        def generate_stabilizers_I_X_Y_only(G):
            n = G.number_of_nodes()
            nodes = list(G.nodes())
            idx_map = {node: i for i, node in enumerate(nodes)}

            # Step 0: Define S_v for each node
            S = {}
            for v in nodes:
                pauli = ['I'] * n
                pauli[idx_map[v]] = 'X'
                for u in G.neighbors(v):
                    pauli[idx_map[u]] = pauli_mult(pauli[idx_map[u]], 'Z')
                S[v] = pauli

            # Step 1: Compute R_full
            R_full = ['I'] * n
            for stab in S.values():
                R_full = multiply_stabilizers(R_full, stab)

            # Step 2: Build candidate generators
            candidates = []

            # From even-degree nodes: R\v = R_full * S_v
            for v in nodes:
                if G.degree[v] % 2 == 0:
                    cand = multiply_stabilizers(R_full, S[v])
                    if is_all_I_X_Y(cand):
                        candidates.append(cand)

            # From odd-degree pairs and even paths: R\(u,w)
            odd_nodes = [v for v in nodes if G.degree[v] % 2 == 1]
            used_pairs = set()
            for u, w in combinations(odd_nodes, 2):
                try:
                    path = nx.shortest_path(G, u, w)
                except nx.NetworkXNoPath:
                    continue
                if all(G.degree[v] % 2 == 0 for v in path[1:-1]):
                    cand = R_full.copy()
                    for v in path:
                        cand = multiply_stabilizers(cand, S[v])
                    if is_all_I_X_Y(cand):
                        candidates.append(cand)

            # Step 3: Use greedy method to get |V|-1 linearly independent stabilizers
            # over GF(2)^n by representing each Pauli string in binary over {I,X,Y}
            # We ignore phase and use a naive basis check
            def to_binary(pstring):
                # Convert ['I','X','Y','I'] → vector over GF(2) (X,Y count as 1)
                return [1 if p in ['X', 'Y'] else 0 for p in pstring]

            basis = []
            basis_bin = []

            for cand in candidates:
                vec = to_binary(cand)
                if not vec in basis_bin:
                    basis.append(cand)
                    basis_bin.append(vec)
                if len(basis) == n - 1:
                    break

            assert len(basis) == n - 1, f"Generated only {len(basis)} out of {n-1} stabilizers."
            return basis

        ## CHECK IF STAB ##

        def gf2_row_reduce(A, b):
            """
            Solves A x = b mod 2 by Gaussian elimination.
            Returns (solution_exists, x_solution).
            """
            A = A.copy() % 2
            b = b.copy() % 2
            n_rows, n_cols = A.shape
            row = 0
            for col in range(n_cols):
                pivot_rows = np.where(A[row:, col] == 1)[0]
                if len(pivot_rows) == 0:
                    continue
                pivot = pivot_rows[0] + row
                # Swap rows
                if pivot != row:
                    A[[row, pivot]] = A[[pivot, row]]
                    b[[row, pivot]] = b[[pivot, row]]
                # Eliminate below
                for r in range(row + 1, n_rows):
                    if A[r, col] == 1:
                        A[r] = (A[r] + A[row]) % 2
                        b[r] = (b[r] + b[row]) % 2
                row += 1
                if row == n_rows:
                    break
            # Back substitution
            x = np.zeros(n_cols, dtype=int)
            for r in reversed(range(n_rows)):
                pivot_cols = np.where(A[r] == 1)[0]
                if len(pivot_cols) == 0:
                    if b[r] == 1:
                        return False, None  # No solution
                    else:
                        continue
                pivot_col = pivot_cols[0]
                x[pivot_col] = b[r]
                # Substitute above
                for rr in range(r):
                    if A[rr, pivot_col] == 1:
                        b[rr] = (b[rr] + x[pivot_col]) % 2
            return True, x

        def check_stabilizer_membership_GF2(G, candidate_stabilizers):
            n = G.number_of_nodes()
            nodes = list(G.nodes())
            idx_map = {v: i for i, v in enumerate(nodes)}

            # Construct S_v generators symplectic vectors
            def pauli_to_symplectic(pauli_str):
                z = []
                x = []
                for p in pauli_str:
                    if p == 'I':
                        z.append(0); x.append(0)
                    elif p == 'X':
                        z.append(0); x.append(1)
                    elif p == 'Y':
                        z.append(1); x.append(1)
                    elif p == 'Z':
                        z.append(1); x.append(0)
                return np.array(z + x, dtype=int)

            S_bin = []
            for v in nodes:
                pauli = ['I'] * n
                pauli[idx_map[v]] = 'X'
                for u in G.neighbors(v):
                    pauli[idx_map[u]] = pauli_mult(pauli[idx_map[u]], 'Z')
                S_bin.append(pauli_to_symplectic(pauli))
            S_bin = np.array(S_bin).T  # shape (2n, n)

            all_ok = True
            for i, stab in enumerate(candidate_stabilizers):
                rhs = pauli_to_symplectic(stab)
                solvable, sol = gf2_row_reduce(S_bin, rhs)
                print(f"Generator {i+1}: {'✓' if solvable else '✗'}")
                if not solvable:
                    all_ok = False
            return all_ok


        # Example usage:
        nqubits = 2
        depth = 3
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        secrets = Secrets(r=True, a=True, theta=True)
        client = Client(pattern=pattern, secrets=secrets)
        print(len(client.nodes_list))
        stabilizers = generate_stabilizers_I_X_Y_only(client.graph)
        check_stabilizer_membership_GF2(client.graph,stabilizers)
        for i, stab in enumerate(stabilizers):
            print(f"Generator {i+1}: {''.join(stab)}")

        


    def test_find_dummyless(self, fx_rng: np.random.Generator):
        nqubits = 2
        depth = 1
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        secrets = Secrets(r=True, a=True, theta=True)
        client = Client(pattern=pattern, secrets=secrets)
        target_stabilizers = []
        for node in client.graph.nodes:
            stab = PauliString(len(client.graph.nodes))
            stab[node] = 1
            for neigh in client.graph.neighbors(node):
                stab[neigh]=3
            target_stabilizers.append(stab)
        print(target_stabilizers)
        
        import stim
        import numpy as np
        import itertools

        def pauli_to_bin(p: stim.PauliString):
            x, z = p.to_numpy(bit_packed=False)
            return np.concatenate([x, z]).astype(np.uint8)

        def gf2_rank(matrix):
            A = matrix.copy() % 2
            rows, cols = A.shape
            rank = 0
            for col in range(cols):
                pivot_row = None
                for row in range(rank, rows):
                    if A[row, col]:
                        pivot_row = row
                        break
                if pivot_row is not None:
                    if pivot_row != rank:
                        A[[rank, pivot_row]] = A[[pivot_row, rank]]
                    for row in range(rows):
                        if row != rank and A[row, col]:
                            A[row] ^= A[rank]
                    rank += 1
            return rank

        def find_independent_subset(target_generators, candidate_generators):
            n = len(target_generators)
            target_matrix = np.stack([pauli_to_bin(p) for p in target_generators])
            target_rank = gf2_rank(target_matrix)

            selected = []
            selected_matrix = []

            for p in candidate_generators:
                vec = pauli_to_bin(p)
                trial_matrix = (
                    np.stack(selected_matrix + [vec]) if selected_matrix else vec[np.newaxis, :]
                )
                if gf2_rank(trial_matrix) > len(selected_matrix):
                    selected.append(p)
                    selected_matrix.append(vec)
                if len(selected) == target_rank:
                    return selected, len(selected)  # early exit

            # Return what was found, even if incomplete
            return selected, len(selected)


        candidates = []
        reduction_found = False
        k=0
        while (not reduction_found) and (k <= len(client.graph.nodes)):
            k += 1
            # Test noiseless trap delegation
            set_of_traps = list(combinations(client.nodes_list, k))
            for possible_trap in set_of_traps:
                # print(possible_trap)
                # random_nodes = random.sample(client.nodes_list, k=trap_size)
                new_trap = tuple(possible_trap)
                traps = (new_trap,)
                test_run = TestRun(client=client, traps=traps)
                if len(test_run.stabilizer.pauli_indices("Z")) == 0 and len(test_run.stabilizer.pauli_indices("X"))+len(test_run.stabilizer.pauli_indices("Y")) != 0:
                    if test_run.stabilizer not in candidates:
                        candidates.append(test_run.stabilizer)
                        print("New dummyless candidate. ", len(candidates))

                    selected, rank = find_independent_subset(target_stabilizers, candidates)
                    if rank == len(target_stabilizers) -1:
                        print(f"Found (nearly) full-rank subset with rank = {rank}")
                        reduction_found = True
                        break
                    else:
                        print(f"Could not find full-rank subset. Max rank found = {rank} out of {len(target_stabilizers)}")
                        print("Need to add another dummyless candidate.")
                        print(len(candidates))
        # print(candidates)

