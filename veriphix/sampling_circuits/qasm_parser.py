from __future__ import annotations

import ast
import math
import operator
import re
from typing import TYPE_CHECKING, Callable

from graphix import Circuit

if TYPE_CHECKING:
    from io import TextIOBase

command_re = re.compile(r"([a-z]+)(?:\(([^)]*)\))?")
reg_re = re.compile(r"[a-z0-9]+\[([^]]*)\]")


def parse_reg(s: str) -> int:
    reg = reg_re.fullmatch(s)
    if reg is None:
        raise ValueError(f"Invalid register: {s}")
    return int(reg.group(1))


# Supported operators mapping for angles
unary_operators: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.USub: operator.neg,
}

binary_operators: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

safe_names: dict[str, float] = {
    "pi": math.pi,
}


def parse_angle(s: str) -> float:
    node = ast.parse(s, mode="eval").body
    return _eval(node)


def _eval(node: ast.expr) -> float:
    if isinstance(node, ast.Constant):  # e.g., 3 or 4.5
        if isinstance(node.value, int):
            return float(node.value)
        if isinstance(node.value, float):
            return node.value
        raise TypeError(f"Unsupported numeric value: {node.n}")
    if isinstance(node, ast.BinOp):  # e.g., 7*pi or 3+4
        left = _eval(node.left)
        right = _eval(node.right)
        return binary_operators[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):  # e.g., -5
        return unary_operators[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name):  # e.g., pi
        if node.id in safe_names:
            return safe_names[node.id]
        raise ValueError(f"Unknown identifier {node.id}")
    raise TypeError(f"Unsupported type {type(node)}")


def read_qasm(f: TextIOBase) -> Circuit:
    lines = "".join(line.strip() for line in f).split(";")
    if lines[0] != "OPENQASM 2.0":
        raise ValueError(f"Unexpected header: {lines[0]}")
    circuit = None
    for line in lines[1:]:
        if line == "":
            continue
        try:
            full_command, arguments_str = line.split(" ", 1)
        except ValueError:
            raise ValueError(f"Invalid syntax: {line}") from None
        arguments = arguments_str.split(",")
        command = command_re.fullmatch(full_command)
        if command is None:
            raise ValueError(f"Invalid syntax for command: {full_command}")
        command_name = command.group(1)
        match command_name:
            case "include":
                pass
            case "qreg":
                if len(arguments) != 1:
                    raise ValueError("qreg expects one argument")
                if circuit is not None:
                    raise ValueError("qreg cannot appear twice")
                circuit = Circuit(parse_reg(arguments[0]))
            case "creg":
                pass
            case "cx" | "rx" | "rz":
                if circuit is None:
                    raise ValueError("qreg is missing")
                if command_name == "cx":
                    if len(arguments) != 2:
                        raise ValueError("cx expects two arguments")
                    control = parse_reg(arguments[0])
                    target = parse_reg(arguments[1])
                    circuit.cnot(control, target)
                else:
                    angle = parse_angle(command.group(2))
                    if len(arguments) != 1:
                        raise ValueError(f"{command_name} expects one argument")
                    qubit = parse_reg(arguments[0])
                    if command_name == "rx":
                        circuit.rx(qubit, angle)
                    else:
                        circuit.rz(qubit, angle)
            case "measure":
                pass
            case _:
                raise ValueError(f"Unknown command: {command_name}")
    if circuit is None:
        raise ValueError("No circuit defined")
    return circuit
