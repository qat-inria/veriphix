import perceval as pcvl
import perceval.components as comp
from perceval.components import catalog
import numpy as np
import networkx as nx

from graphix.states import BasicStates
from graphix.simulator import MeasureMethod, PrepareMethod
from graphix.sim.base_backend import Backend, NodeIndex
from graphix.command import CommandKind
from graphix.pattern import Pattern

class PercevalBackend:

    def __init__(self, source: pcvl.Source, graph: nx.Graph, input_nodes):
        self._source = source
        self._graph = graph
        self._input_nodes = input_nodes
        self.node_index = NodeIndex()
        self._proc = pcvl.Processor("SLOS", 2*len(self._graph.nodes))
        self._count_nodes = 0

    def add_nodes(self, nodes, data=BasicStates.PLUS):

        if np.abs(data.psi[0]) != 1:
            if np.abs(data.psi[1]) == 1:
                self._proc.add(2*self._count_nodes, comp.PERM([1, 0]))
            else:
                self._proc.add(2*self._count_nodes, comp.BS.H())
                phase = data.psi[1]*np.sqrt(2)
                phase_min_pi_4 = np.exp(- np.pi * 1j / 4)
                k = 0
                while phase.real != 1:
                    phase *= phase_min_pi_4
                    k += 1
                self._proc.add(2*self._count_nodes + 1, comp.PS(np.pi*k/4))

        self._count_nodes += 1
        self.node_index.extend(nodes)

    def run(self, pattern: Pattern, prepare_method: PrepareMethod, measure_method: MeasureMethod):
        nodes = self._graph.nodes
        edges = self._graph.edges

        for node in nodes:
            if node not in self._input_nodes:
                prepare_method.prepare_node(self, node)

        for edge in edges:
            ctrl = min(edge)
            target = max(edge)
            cz_inp = [2*ctrl, 2*ctrl + 1, 2*target, 2*target + 1]
            self._proc.add(cz_inp,catalog["heralded cz"].build_processor())

        circ = self._proc.linear_circuit()

        sim = pcvl.simulators.Simulator(pcvl.BackendFactory.get_backend("SLOS"))
        sim.set_min_detected_photons_filter(0)
        sim.set_circuit(circ)

        heralds = dict.fromkeys(list(range(2*len(nodes), circ.m)), 1)
        sim.set_heralds(heralds)
        sim.keep_heralds(False)

        st_0 = pcvl.BasicState([1, 0])
        st_herald = pcvl.BasicState([1, 1])
        basic_input = (st_0 ** len(nodes))*(st_herald ** len(edges))
        input_svd = self._source.generate_distribution(basic_input)

        graph_state = pcvl.DensityMatrix.from_svd(sim.evolve_svd(input_svd)["results"])

        ps = pcvl.PostSelect("([0] > 0 & [1] == 0) | ([0] == 0 & [1] > 0)")
        position_in_state = list(range(len(nodes)))

        sim.clear_heralds()

        for cmd in pattern:

            if cmd.kind != CommandKind.M:
                continue

            description = measure_method.get_measurement_description(cmd)
            meas_circ = pcvl.Circuit(graph_state.m, "measurement") 
            meas_circ.add(2*position_in_state[cmd.node] + 1, comp.PS(-description.angle))
            meas_circ.add(2*position_in_state[cmd.node], comp.BS.H())
            sim.set_circuit(meas_circ)
            graph_state = sim.evolve_density_matrix(graph_state)

            all_possible_meas_outcomes = graph_state.measure([2*position_in_state[cmd.node], 2*position_in_state[cmd.node] + 1])

            outcome_dist = {}
            for outcome, res in all_possible_meas_outcomes.items():
                outcome_dist[outcome] = res[0]
            outcomes = pcvl.BSDistribution(outcome_dist)
            ps_outcomes = pcvl.utils.postselect.post_select_distribution(outcomes, ps)[0]
            meas_result = ps_outcomes.sample(1)[0]
            bool_result = meas_result[0] == 0
            measure_method.set_measure_result(cmd.node, bool_result)

            graph_state = all_possible_meas_outcomes[meas_result][1]
            for i in range(cmd.node, len(position_in_state)):
                position_in_state[i] -= 1

        self.node_index = NodeIndex()
        self._proc = pcvl.Processor("SLOS", 2*len(self._graph.nodes))
        self._count_nodes = 0