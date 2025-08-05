import numpy as np
import networkx as nx
import stim
from itertools import combinations


def generate_stabilizers_I_X_Y_only_stim(G):
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    idx_map = {node: i for i, node in enumerate(nodes)}

    # Step 0: Define S_v for each node as stim.PauliString
    S = {}
    for v in nodes:
        # start with identity
        ps = stim.PauliString({}, num_qubits=n)
        # X on v
        ps = ps & stim.PauliString({idx_map[v]: 'X'}, num_qubits=n)
        # Z on neighbors
        for u in G.neighbors(v):
            ps = ps & stim.PauliString({idx_map[u]: 'Z'}, num_qubits=n)
        S[v] = ps

    # Step 1: Compute R_full = product of all S_v
    R_full = stim.PauliString({}, num_qubits=n)
    for ps in S.values():
        R_full = R_full * ps

    # Step 2: Build candidate generators
    candidates = []

    def all_I_X_Y(ps: stim.PauliString) -> bool:
        # check no Z anywhere
        return 'Z' not in str(ps)

    # From even-degree nodes
    for v in nodes:
        if G.degree[v] % 2 == 0:
            cand = R_full * S[v]
            if all_I_X_Y(cand):
                candidates.append(cand)

    # From odd-degree pairs with even interior path
    odd_nodes = [v for v in nodes if G.degree[v] % 2 == 1]
    for u, w in combinations(odd_nodes, 2):
        try:
            path = nx.shortest_path(G, u, w)
        except nx.NetworkXNoPath:
            continue
        if all(G.degree[x] % 2 == 0 for x in path[1:-1]):
            cand = R_full
            for v in path:
                cand = cand * S[v]
            if all_I_X_Y(cand):
                candidates.append(cand)

    # Step 3: pick n-1 independent over GF(2)
    def to_binary(ps: stim.PauliString):
        vec = np.zeros(n, dtype=int)
        for q, p in ps.items():
            if p in ['X', 'Y']:
                vec[q] = 1
        return vec.tolist()

    basis = []  
    basis_bin = []
    for cand in candidates:
        vec = to_binary(cand)
        if vec not in basis_bin:
            basis.append(cand)
            basis_bin.append(vec)
        if len(basis) == n - 1:
            break

    assert len(basis) == n - 1, f"Generated only {len(basis)} out of {n-1} stabilizers."
    return basis


def gf2_row_reduce(A, b):
    A = A.copy() % 2
    b = b.copy() % 2
    n_rows, n_cols = A.shape
    row = 0
    for col in range(n_cols):
        piv = np.where(A[row:, col] == 1)[0]
        if len(piv) == 0:
            continue
        pivot = piv[0] + row
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
            b[[row, pivot]] = b[[pivot, row]]
        for r in range(row + 1, n_rows):
            if A[r, col] == 1:
                A[r] ^= A[row]
                b[r] ^= b[row]
        row += 1
        if row == n_rows:
            break
    x = np.zeros(n_cols, dtype=int)
    for r in range(row - 1, -1, -1):
        ones = np.where(A[r] == 1)[0]
        if len(ones) == 0:
            if b[r] == 1:
                return False, None
            continue
        lead = ones[0]
        x[lead] = b[r]
        for rr in range(r):
            if A[rr, lead] == 1:
                b[rr] ^= x[lead]
    return True, x


def check_stabilizer_membership_GF2_stim(G, candidate_stabilizers):
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    idx_map = {v: i for i, v in enumerate(nodes)}

    def pauli_to_symplectic(ps: stim.PauliString):
        z = np.zeros(n, dtype=int)
        x = np.zeros(n, dtype=int)
        for q, p in ps.items():
            if p in ['Z', 'Y']:
                z[q] = 1
            if p in ['X', 'Y']:
                x[q] = 1
        return np.concatenate([z, x])

    # build S_v bin
    S_bin = []
    for v in nodes:
        ps = stim.PauliString({idx_map[v]: 'X'}, num_qubits=n)
        for u in G.neighbors(v):
            ps = ps * stim.PauliString({idx_map[u]: 'Z'}, num_qubits=n)
        S_bin.append(pauli_to_symplectic(ps))
    S_bin = np.array(S_bin).T

    all_ok = True
    for i, stab in enumerate(candidate_stabilizers):
        rhs = pauli_to_symplectic(stab)
        ok, sol = gf2_row_reduce(S_bin, rhs)
        print(f"Generator {i+1}: {'✓' if ok else '✗'}")
        all_ok &= ok
    return all_ok


# Example usage:
if __name__ == '__main__':
    # build a simple test graph
    G = nx.path_graph(4)
    stabs = generate_stabilizers_I_X_Y_only_stim(G)
    print("Selected stabilizers:")
    for s in stabs:
        print(str(s))
    check_stabilizer_membership_GF2_stim(G, stabs)
