from __future__ import annotations

import enum
import math
import typing
from abc import ABC, abstractmethod
from array import array
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from graphix import Circuit, Pattern, command
from graphix.instruction import InstructionKind
from graphix.rng import ensure_rng

if TYPE_CHECKING:
    from graphix import instruction
    from np.random import Generator


class Brick(ABC):
    @abstractmethod
    def measures(self) -> list[list[float]]: ...

    @abstractmethod
    def to_circuit(self, target: Circuit, nqubit_top: int) -> None: ...


@dataclass
class CNOT(Brick):
    target_above: bool

    def measures(self) -> list[list[float]]:
        if self.target_above:
            return [[0, math.pi / 2, 0, -math.pi / 2], [0, 0, math.pi / 2, 0]]
        return [[0, 0, math.pi / 2, 0], [0, math.pi / 2, 0, -math.pi / 2]]

    def to_circuit(self, circuit: Circuit, nqubit_top: int) -> None:
        if self.target_above:
            control = nqubit_top + 1
            target = nqubit_top
        else:
            control = nqubit_top
            target = nqubit_top + 1
        circuit.cnot(control, target)


class XZ(Enum):
    X = enum.auto()
    Z = enum.auto()


def value_or_zero(v: float | None) -> float:
    if v is None:
        return 0
    return v


@dataclass
class SingleQubit:
    rz0: float | None = None
    rx: float | None = None
    rz1: float | None = None

    def measures(self) -> list[float]:
        return [
            -value_or_zero(self.rz0),
            -value_or_zero(self.rx),
            -value_or_zero(self.rz1),
            0,
        ]

    def to_circuit(self, circuit: Circuit, nqubit: int) -> None:
        if self.rz0 is not None:
            circuit.rz(nqubit, self.rz0)
        if self.rx is not None:
            circuit.rx(nqubit, self.rx)
        if self.rz1 is not None:
            circuit.rz(nqubit, self.rz1)

    def is_identity(self) -> bool:
        return self.rz0 is None and self.rx is None and self.rz1 is None

    def add(self, axis: XZ, angle: float) -> bool:
        match axis:
            case XZ.X:
                if self.rx is None and self.rz1 is None:
                    self.rx = angle
                    return True
                return False
            case XZ.Z:
                if self.rz0 is None and self.rx is None:
                    self.rz0 = angle
                    return True
                if self.rz1 is None:
                    self.rz1 = angle
                    return True
                return False
            case _:
                typing.assert_never(axis)


@dataclass
class SingleQubitPair(Brick):
    top: SingleQubit
    bottom: SingleQubit

    def get(self, position: bool) -> SingleQubit:
        if position:
            return self.bottom
        return self.top

    def measures(self) -> list[list[float]]:
        return [self.top.measures(), self.bottom.measures()]

    def to_circuit(self, circuit: Circuit, nqubit_top: int) -> None:
        self.top.to_circuit(circuit, nqubit_top)
        self.bottom.to_circuit(circuit, nqubit_top + 1)


def identity() -> SingleQubitPair:
    return SingleQubitPair(SingleQubit(), SingleQubit())


@dataclass
class Layer:
    odd: bool
    bricks: list[Brick]

    def get(self, qubit: int) -> tuple[Brick, bool]:
        index = (qubit - int(self.odd)) // 2
        return (self.bricks[index], bool(qubit % 2) != self.odd)


def __get_layer(width: int, layers: list[Layer], depth: int) -> Layer:
    for i in range(len(layers), depth + 1):
        odd = bool(i % 2)
        layer_size = (width - 1) // 2 if odd else max(width // 2, 1)
        layers.append(
            Layer(
                odd,
                [identity() for _ in range(layer_size)],
            )
        )
    return layers[depth]


def __insert_rotation(
    width: int,
    layers: list[Layer],
    depth: list[int],
    instr: instruction.RX | instruction.RZ,
) -> None:
    axis = XZ.X if instr.kind == InstructionKind.RX else XZ.Z
    target_depth = depth[instr.target]
    if target_depth > 0:
        previous_layer = layers[target_depth - 1]
        brick, position = previous_layer.get(instr.target)
        if isinstance(brick, SingleQubitPair):
            gate = brick.get(position)
            if gate.add(axis, instr.angle):
                return
        else:
            assert isinstance(brick, CNOT)
    if (instr.target == 0 and target_depth % 2) or (
        width >= 2 and instr.target == width - 1 and target_depth % 2 != width % 2
    ):
        target_depth += 1
    layer = __get_layer(width, layers, target_depth)
    brick, position = layer.get(instr.target)
    assert isinstance(brick, SingleQubitPair)
    gate = brick.get(position)
    assert gate.is_identity()
    added = gate.add(axis, instr.angle)
    assert added
    depth[instr.target] = target_depth + 1


def transpile_to_layers(circuit: Circuit) -> list[Layer]:
    layers: list[Layer] = []
    depth = [0 for _ in range(circuit.width)]
    for instr in circuit.instruction:
        # Use of `if` instead of `match` here for mypy
        if instr.kind == InstructionKind.CNOT:
            if abs(instr.control - instr.target) != 1:
                raise ValueError(
                    "Unsupported CNOT: control and target qubits should be consecutive"
                )
            target = min(instr.control, instr.target)
            min_depth = max(depth[target], depth[target + 1])
            target_depth = min_depth if target % 2 == min_depth % 2 else min_depth + 1
            target_layer = __get_layer(circuit.width, layers, target_depth)
            index = target // 2
            target_layer.bricks[index] = CNOT(target == instr.target)
            depth[target] = target_depth + 1
            depth[target + 1] = target_depth + 1
        # Use of `==` here for mypy
        elif instr.kind == InstructionKind.RX or instr.kind == InstructionKind.RZ:  # noqa: PLR1714
            __insert_rotation(circuit.width, layers, depth, instr)
        else:
            raise ValueError(
                "Unsupported gate: circuits should contain only CNOT, RX and RZ"
            )
    return layers


@dataclass
class NodeGenerator:
    from_index: int

    def fresh_command(self) -> tuple[int, command.Command]:
        index = self.from_index
        self.from_index += 1
        return index, command.N(node=index)

    def fresh(self, pattern: Pattern) -> int:
        index, command = self.fresh_command()
        pattern.add(command)
        return index


def j_commands(
    node_generator: NodeGenerator, node: int, angle: float
) -> tuple[int, list[command.Command]]:
    next_node, command_n = node_generator.fresh_command()
    commands = [
        command_n,
        command.E(nodes=(node, next_node)),
        command.M(node=node, angle=angle / math.pi),
        command.X(node=next_node, domain={node}),
    ]
    return next_node, commands


class ConstructionOrder(Enum):
    Canonical = enum.auto()
    Deviant = enum.auto()
    DeviantRight = enum.auto()


def nqubits_from_layers(layers: list[Layer]) -> int:
    if len(layers) == 0:
        raise ValueError("Layer list should not be empty")
    if len(layers) == 1:
        return 2 * len(layers[0].bricks)
    even_brick_count = len(layers[0].bricks)
    odd_brick_count = len(layers[1].bricks)
    return even_brick_count * 2 + int(even_brick_count == odd_brick_count)


def layers_to_measurement_table(layers: list[Layer]) -> list[list[float]]:
    nqubits = nqubits_from_layers(layers)
    table = []
    for layer_index, layer in enumerate(layers):
        all_brick_measures = [brick.measures() for brick in layer.bricks]
        for column_index in range(4):
            column: list[float] = []
            if layer.odd:
                column.append(0)
            column.extend(
                measures[i][column_index]
                for measures in all_brick_measures
                for i in (0, 1)
            )
            if layer_index % 2 != nqubits % 2:
                column.append(0)
            table.append(column)
    return table


def measurement_table_to_pattern(
    width: int, table: list[list[float]], order: ConstructionOrder
) -> Pattern:
    input_nodes = list(range(width))
    pattern = Pattern(input_nodes)
    nodes = input_nodes
    node_generator = NodeGenerator(width)
    for time, column in enumerate(table):
        postponed = None  # for deviant order
        for qubit, angle in enumerate(column):
            next_node, commands = j_commands(node_generator, nodes[qubit], angle)
            if (time % 4 in {2, 0} and time > 0) and order != ConstructionOrder.Deviant:
                brick_layer = (time - 1) // 4
                match order:
                    case ConstructionOrder.Canonical:
                        if qubit % 2 == brick_layer % 2 and qubit != width - 1:
                            pattern.add(
                                command.E(nodes=(nodes[qubit], nodes[qubit + 1]))
                            )
                        pattern.extend(commands)
                    case ConstructionOrder.DeviantRight:
                        if qubit % 2 == brick_layer % 2 and qubit != width - 1:
                            pattern.extend(commands[:2])
                            postponed = (nodes[qubit], commands[2:])
                        elif postponed is None:
                            pattern.extend(commands)
                        else:
                            pattern.extend(commands[:2])
                            previous_qubit, previous_commands = postponed
                            postponed = None
                            pattern.add(command.E(nodes=(previous_qubit, nodes[qubit])))
                            pattern.extend(previous_commands)
                            pattern.extend(commands[2:])
            elif time % 4 in {1, 3} and order == ConstructionOrder.Deviant:
                brick_layer = time // 4
                if qubit % 2 == brick_layer % 2 and qubit != width - 1:
                    pattern.add(commands[0])
                    postponed = (nodes[qubit], commands[1:])
                elif postponed is None:
                    pattern.extend(commands)
                else:
                    pattern.add(commands[0])
                    previous_qubit, previous_commands = postponed
                    postponed = None
                    pattern.add(command.E(nodes=(nodes[qubit - 1], next_node)))
                    pattern.extend(previous_commands)
                    pattern.extend(commands[1:])
                    pattern.extend(
                        [
                            command.Z(node=nodes[qubit - 1], domain={nodes[qubit]}),
                            command.Z(node=next_node, domain={previous_qubit}),
                        ]
                    )
            else:
                pattern.extend(commands)
            nodes[qubit] = next_node
    if order != ConstructionOrder.Deviant:
        last_brick_layer = (len(table) - 1) // 4
        for qubit in range(last_brick_layer % 2, width - 1, 2):
            pattern.add(command.E(nodes=(nodes[qubit], nodes[qubit + 1])))
    return pattern


def layers_to_pattern(
    width: int,
    layers: list[Layer],
    order: ConstructionOrder = ConstructionOrder.Canonical,
) -> Pattern:
    table = layers_to_measurement_table(layers)
    return measurement_table_to_pattern(width, table, order)


def transpile(
    circuit: Circuit, order: ConstructionOrder = ConstructionOrder.Canonical
) -> Pattern:
    layers = transpile_to_layers(circuit)
    return layers_to_pattern(circuit.width, layers, order)


def get_node_positions(
    pattern: Pattern, scale: float = 1, reverse_qubit_order: bool = False
) -> dict[int, array[int]]:
    """Return node positions in a grid layout."""
    width = len(pattern.input_nodes)
    return {
        node: array(
            "i",
            [
                int((node // width) * scale),
                int(
                    (width - node % width if reverse_qubit_order else node % width)
                    * scale
                ),
            ],
        )
        for node in range(pattern.n_node)
    }


def get_bipartite_coloring(pattern: Pattern) -> tuple[set[int], set[int]]:
    """Return a bipartite coloring for the given pattern."""
    positions = get_node_positions(pattern)
    red = set()
    blue = set()
    for node, position in positions.items():
        if (position[0] + position[1]) % 2:
            red.add(node)
        else:
            blue.add(node)
    return (red, blue)


PAULI_ANGLES: list[float] = [0, math.pi, math.pi / 2, -math.pi / 2]


def random_pauli_measurement_angle(rng: Generator) -> float:
    index: int = rng.integers(len(PAULI_ANGLES))
    return PAULI_ANGLES[index]


def generate_random_pauli_measurement_table(
    nqubits: int, nlayers: int, rng: Generator
) -> list[list[float]]:
    return [
        [random_pauli_measurement_angle(rng) for _ in range(nqubits)]
        for _ in range(nlayers * 4)
    ]


def generate_random_pauli_pattern(
    nqubits: int,
    nlayers: int,
    rng: Generator | None = None,
    order: ConstructionOrder = ConstructionOrder.Canonical,
) -> Pattern:
    rng = ensure_rng(rng)
    table = generate_random_pauli_measurement_table(nqubits, nlayers, rng)
    return measurement_table_to_pattern(nqubits, table, order)


def layers_to_circuit(layers: list[Layer]) -> Circuit:
    width = nqubits_from_layers(layers)
    circuit = Circuit(width)
    for layer in layers:
        nqubit = int(layer.odd)
        for brick in layer.bricks:
            brick.to_circuit(circuit, nqubit)
            nqubit += 2
    return circuit
