from abc import ABC, abstractmethod
from veriphix.client import Client
from graphix.sim.base_backend import Backend
from graphix.states import State
from graphix.pauli import Pauli, IXYZ
from stim import Tableau, PauliString

class Run(ABC):
    def __init__(self, client:Client) -> None:
        self.client = client

    @abstractmethod
    def delegate_UBQC(self, backend:Backend):
        pass

class ComputationRun(Run):
    def __init__(self) -> None:
        super().__init__()

    def delegate_UBQC(self, backend:Backend):
        results = self.client.delegate_pattern(backend=backend)
        return results
    

def merge(strings:list[PauliString]):
    n = len(strings)
    l = len(strings[0])
    common_string = strings[0]
    for i in range(n):
        for j in range(l):
            if common_string[j] == 0 and strings[i][j] != 0:
                common_string[j] = strings[i][j]
    return common_string

def generate_eigenstate(stabilizer:PauliString) -> list[State]:
    states = []
    for pauli in stabilizer:
        # default coin = 0
        operator = Pauli(IXYZ(pauli))
        states.append(operator.eigenstate())
    return states
        


class TestRun(Run):
    def __init__(self, traps:set[set[int]], clifford_structure:Tableau, meas_basis:str="X") -> None:
        super().__init__()
        self.traps = traps
        self.meas_basis = meas_basis
        self.clifford_structure = clifford_structure
        self.nqubits = len(self.clifford_structure)

    def build_common_stabilizer(self):
        # Build the PauliStrings representing the individual measurement of each trap qubit
        measurement_strings = [
            PauliString([
                    self.meas_basis if i in trap else "I"
                ]
                for i in len(self.nqubits)
            )
            for trap in self.traps
        ]
        # Conjugate each measurement
        conjugated_measurements = [self.clifford_structure(meas) for meas in measurement_strings]
        common_stabilizer = merge(conjugated_measurements)
        return common_stabilizer

    def build_common_eigenstate(self):
        stabilizer = self.build_common_stabilizer()
        self.input_state = generate_eigenstate(stabilizer)



    def delegate_UBQC(self, backend:Backend):
        self.client.delegate_pattern(backend=backend)
    