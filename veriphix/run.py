from abc import ABC, abstractmethod
from veriphix.client import Client
from graphix.sim.base_backend import Backend
from graphix.states import State
from graphix.pauli import Pauli, IXYZ
from stim import Tableau, PauliString
from graphix.pattern import PatternSimulator
from dataclasses import dataclass

class Run(ABC):
    def __init__(self, client:Client) -> None:
        self.client = client

    @abstractmethod
    def delegate(self, backend:Backend, **kwargs):
        # Delegates using UBQC
        pass



class ComputationRun(Run):
    def __init__(self, client:Client) -> None:
        super().__init__(client=client)

    ## TODO: replace this by "delegate_pattern" and delete that
    def delegate(self, backend:Backend, **kwargs) -> dict[int, int]:

        # Initializes the bank & asks backend to create the input
        self.client.prepare_states(backend, states_dict=self.client.computation_states)

        sim = PatternSimulator(
            backend=backend,
            pattern=self.client.clean_pattern,
            prepare_method=self.client.prepare_method,
            measure_method=self.client.measure_method,
            **kwargs,
        )
        sim.run(input_state=None)

        # If quantum output, decode the state, nothing needs to be returned (backend.state can be accessed by the Client)
        if self.client.output_nodes == self.client.initial_pattern.output_nodes:
            self.client.decode_output_state(backend)
            return {}
        # If classical output, return the output
        else:
            return {onode: self.client.results[onode] for onode in self.client.output_nodes}
    

def merge_pauli_strings(stabilizer_1: PauliString, stabilizer_2: PauliString) -> PauliString:
    result = stabilizer_2.sign * stabilizer_1
    # We can iterate through the support of stab2 as they are supposed to have equal support
    for node in stabilizer_2.pauli_indices("XYZ"):
        result[node] = stabilizer_2[node]
    return result

def merge(strings:list[PauliString]):
    n = len(strings)
    l = len(strings[0])
    common_string = strings[0]
    # if common_string.sign == -1:
    #     print("Have to change the sign")
    # common_string.sign = 1
    for i in range(1, n):
        common_string.sign *= strings[i].sign
        for j in range(l):
            if (common_string[j] != 0 and strings[i][j] != 0) and (common_string[j] != strings[i][j]):
                raise Exception("Traps are not compatible.")
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
        

# Trap=set[int]
Trap=tuple[int]

class TestRun(Run):
    def __init__(self, client:Client, traps:tuple[Trap], meas_basis:str="X") -> None:
        super().__init__(client=client)
        self.traps = traps
        self.meas_basis = meas_basis
        self.clifford_structure = client.clifford_structure
        self.nqubits = len(self.clifford_structure)
        self.stabilizer = self.build_common_stabilizer()
        self.input_state = self.build_common_eigenstate()

    def build_common_stabilizer(self):
        # Build the PauliStrings representing the individual measurement of each trap qubit
        measurement_strings = [
            PauliString([
                    self.meas_basis if i in trap else "I"
                    for i in range(self.nqubits)
                ]
            )
            for trap in self.traps
        ]
        # Conjugate each measurement
        conjugated_measurements = [self.clifford_structure.inverse()(meas) for meas in measurement_strings]
        common_stabilizer = merge(conjugated_measurements)
        return common_stabilizer

    def build_common_eigenstate(self):
        input_state = generate_eigenstate(self.stabilizer)
        return input_state



    def delegate(self, backend:Backend, **kwargs):
        states_dict = {node:self.input_state[node] for node in self.client.nodes_list}
        self.client.prepare_states(backend=backend, states_dict=states_dict)
        sim = PatternSimulator(
            backend=backend,
            pattern=self.client.clean_pattern,
            prepare_method=self.client.prepare_method,
            measure_method=self.client.test_measure_method,
            **kwargs,
        )
        sim.run(input_state=None)


        trap_outcomes = dict()
        for trap in self.traps:
            
            outcomes = [self.client.results[component] for component in trap]
            trap_outcome = sum(outcomes) % 2 ^ (self.stabilizer.sign==-1) 
            trap_outcomes[
                trap
                ] = trap_outcome
            # trap_outcomes.append(trap_outcome)
        return trap_outcomes


    