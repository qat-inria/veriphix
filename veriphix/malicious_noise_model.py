"""Malicious noise model."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from graphix.channels import KrausChannel, dephasing_channel
from graphix.command import CommandKind
from graphix.noise_models.noise_model import ApplyNoise, CommandOrNoise, Noise, NoiseModel
from graphix.rng import ensure_rng
from graphix.utils import Probability

# override introduced in Python 3.12
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Iterable

    from graphix.command import BaseM
    from graphix.measurements import Outcome
    from numpy.random import Generator


class DephasingNoise(Noise):
    """One-qubit dephasing noise with probabibity ``prob``."""

    prob = Probability()

    def __init__(self, prob: float) -> None:
        """Initialize one-qubit dephasing noise.

        Parameters
        ----------
        prob : float
            Probability parameter of the noise, between 0 and 1.
        """
        self.prob = prob

    @property
    @override
    def nqubits(self) -> int:
        """Return the number of qubits targetted by the noise element."""
        return 1

    @override
    def to_kraus_channel(self) -> KrausChannel:
        """Return the Kraus channel describing the noise element."""
        return dephasing_channel(self.prob)


class MaliciousNoiseModel(NoiseModel):
    """Malicious noise model.

    :param NoiseModel: Parent abstract class class:`graphix.noise_model.NoiseModel`
    :type NoiseModel: class
    """

    def __init__(
        self,
        nodes: list[int],
        prob: float = 0.0,
    ) -> None:
        self.prob = prob
        self.nodes = nodes
        self.node = random.choice(self.nodes)
        self.refresh_randomness()

    def refresh_randomness(self, rng: Generator | None = None) -> None:
        rng = ensure_rng(rng)
        # self.node = random.choice(self.nodes)
        # self.target_nodes = random.sample(self.nodes, self.n_targets)
        self.attack = bool(rng.uniform() < self.prob)

    @override
    def input_nodes(self, nodes: Iterable[int], rng: Generator | None = None) -> list[CommandOrNoise]:
        """Return the noise to apply to input nodes."""
        return []

    @override
    def command(self, cmd: CommandOrNoise, rng: Generator | None = None) -> list[CommandOrNoise]:
        """Return the noise to apply to the command `cmd`."""
        if cmd.kind == CommandKind.M and cmd.node in self.nodes and self.attack:
            return [cmd, ApplyNoise(DephasingNoise(prob=1), [self.node])]
        else:
            return [cmd]

    @override
    def confuse_result(self, cmd: BaseM, result: Outcome, rng: Generator | None = None) -> Outcome:
        """Assign wrong measurement result cmd = "M"."""
        return result
