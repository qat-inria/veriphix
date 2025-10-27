import perceval as pcvl
import perceval.components as comp
from perceval.components import catalog
import numpy as np

from graphix.states import BasicStates
from graphix.measurements import Measurement
from graphix.sim.base_backend import Backend
from graphix.command import CommandKind
from graphix.clifford import Clifford
from graphix.fundamentals import Plane
from graphix.sim.density_matrix import DensityMatrix

class PercevalBackend(Backend):

    def __init__(
        self,
        source: pcvl.Source,
        perceval_state: pcvl.StateVector = pcvl.BasicState()
    ):

        self._source = source
        self._perceval_state = perceval_state

        self._sim = pcvl.simulators.Simulator(pcvl.BackendFactory.get_backend("SLOS"))
        self._sim.set_min_detected_photons_filter(0)
        self._sim.keep_heralds(False)

        super().__init__(DensityMatrix(nqubit=0), pr_calc=True, rng=None)

    
    @property
    def source(self):
        return self._source
    
    @property
    def state(self):
        return self._perceval_state
    
    @property
    def nqubit(self) -> int:
        return int(self.state.m/2)
    
    def copy(self):
        return PercevalBackend(self._source, self._perceval_state)

    def add_nodes(self, nodes, data=BasicStates.PLUS):

        init_circ = pcvl.Circuit(2)

        alpha = data.psi[0]
        beta = data.psi[1]

        if np.abs(beta) != 0:
            if np.abs(alpha) == 0:
                init_circ.add(0, comp.PERM([1, 0]))
            else:
                gamma = np.abs(beta)
                delta = -np.conjugate(alpha)*gamma/np.conjugate(beta)
                matrix = pcvl.Matrix(np.asarray([[alpha, gamma], [beta, delta]]))
                init_circ.add(0, comp.Unitary(U = matrix))

        self._sim.set_circuit(init_circ)

        zero_mixed_state = self._source.generate_distribution(pcvl.BasicState([1, 0]))
        sampled_zero_mixed_state = zero_mixed_state.sample(1)[0]
        init_qubit = self._sim.evolve(sampled_zero_mixed_state)
        self._perceval_state *= init_qubit

        self.node_index.extend(nodes)

    def entangle_nodes(self, edge: tuple[int, int]) -> None:

        index_0 = self.node_index.index(edge[0])
        index_1 = self.node_index.index(edge[1])
        ctrl = min(index_0, index_1)
        target = max(index_0, index_1)
        cz_input_modes = [2*ctrl, 2*ctrl + 1, 2*target, 2*target + 1]

        ent_proc = pcvl.Processor("SLOS", 2*self.nqubit)
        ent_proc.add(cz_input_modes, catalog["heralded cz"].build_processor())
        ent_circ = ent_proc.linear_circuit()

        self._sim.set_circuit(ent_circ)

        heralds = dict.fromkeys(list(range(2*self.nqubit, ent_circ.m)), 1)
        self._sim.set_heralds(heralds)
        herald_state = self._source.generate_distribution(pcvl.BasicState([1, 1]))
        sampled_herald_state = herald_state.sample(1)[0]

        self._perceval_state = self._sim.evolve(self._perceval_state*sampled_herald_state)

        self._sim.clear_heralds()

    def measure(self, node: int, measurement: Measurement) -> bool:
        """Perform measurement of a node and trace out the qubit.

        Parameters
        ----------
        node: int
        measurement: Measurement
        """
        index = self.node_index.index(node)

        meas_circ = pcvl.Circuit(2*self.nqubit)
        match measurement.plane:

            case Plane.XY:
                # rotation around Z axis by -angle
                meas_circ.add(2*index + 1, comp.PS(-measurement.angle))
                # transformation from X basis to Z basis
                meas_circ.add(2*index, comp.BS.H())

            case Plane.YZ:
                # rotation around X axis by -angle
                meas_circ.add(2*index, comp.BS.H())
                meas_circ.add(2*index + 1, comp.PS(-measurement.angle))
                meas_circ.add(2*index, comp.BS.H())
                # transformation from Y basis to Z basis
                meas_circ.add(2*index + 1, comp.PS(-np.pi/2))
                meas_circ.add(2*index, comp.BS.H())

            case Plane.XZ:
                # rotation around Y axis by -angle
                meas_circ.add(2*index + 1, comp.PS(-np.pi/2))
                meas_circ.add(2*index, comp.BS.H())
                meas_circ.add(2*index + 1, comp.PS(-measurement.angle))
                meas_circ.add(2*index, comp.BS.H())
                meas_circ.add(2*index + 1, comp.PS(np.pi/2))
                # transformation from X basis to Z basis
                meas_circ.add(2*index, comp.BS.H())

        
        self._sim.set_circuit(meas_circ)
        self._perceval_state = self._sim.evolve(self._perceval_state)

        all_possible_meas_outcomes = self._perceval_state.measure([2*index, 2*index + 1])
        outcome_dist = {}
        for outcome, res in all_possible_meas_outcomes.items():
            outcome_dist[outcome] = res[0]
        outcomes = pcvl.BSDistribution(outcome_dist)
        ps = pcvl.PostSelect("([0] > 0 & [1] == 0) | ([0] == 0 & [1] > 0)")
        ps_outcomes = pcvl.utils.postselect.post_select_distribution(outcomes, ps)[0]
        meas_result = ps_outcomes.sample(1)[0]
        result = meas_result[0] == 0

        self._perceval_state = all_possible_meas_outcomes[meas_result][1]

        self.node_index.remove(node)

        return result

    def correct_byproduct(self, cmd, measure_method) -> None:
        """Byproduct correction correct for the X or Z byproduct operators, by applying the X or Z gate."""

        if np.mod(sum([measure_method.get_measure_result(j) for j in cmd.domain]), 2) == 1:

            index = self.node_index.index(cmd.node)
            correct_circ = pcvl.Circuit(2*self.nqubit)

            if cmd.kind == CommandKind.X:
                correct_circ.add(2*index, comp.PERM([1, 0]))
            elif cmd.kind == CommandKind.Z:
                correct_circ.add(2*index + 1, comp.PS(np.pi))

            self._sim.set_circuit(correct_circ)
            self._perceval_state = self._sim.evolve(self._perceval_state)

    def apply_clifford(self, node: int, clifford: Clifford) -> None:
        """Apply single-qubit Clifford gate, specified by vop index specified in graphix.clifford.CLIFFORD."""

        index = self.node_index.index(node)

        clifford_circ = pcvl.Circuit(2*self.nqubit).add(2*index, comp.Unitary(U = pcvl.Matrix(clifford.matrix)))

        self._sim.set_circuit(clifford_circ)
        self._perceval_state = self._sim.evolve(self._perceval_state)

    def sort_qubits(self, output_nodes) -> None:
        """Sort the qubit order in internal statevector."""
        if self.nqubit > 0:
            perm_circ = pcvl.Circuit(2*self.nqubit)

            for i, ind in enumerate(output_nodes):
                if self.node_index.index(ind) != i:
                    move_from = self.node_index.index(ind)
                    self.node_index.swap(i, move_from)

                    low = min(i, move_from)
                    high = max(i, move_from)
                    perm_circ.add(2*low, comp.PERM([2*(high - low), 2*(high - low) + 1] + list(range(2, 2*(high - low))) + [0, 1]))

            self._sim.set_circuit(perm_circ)
            self._perceval_state = self._sim.evolve(self._perceval_state)