"""Malicious noise model."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from graphix.channels import KrausChannel, dephasing_channel
from graphix.command import Command, CommandKind
from graphix.noise_models.noise_model import Noise, NoiseModel
from graphix.rng import ensure_rng

if TYPE_CHECKING:
    from numpy.random import Generator


class MaliciousNoiseModel(NoiseModel):
    """Malicious noise model.

    :param NoiseModel: Parent abstract class class:`graphix.noise_model.NoiseModel`
    :type NoiseModel: class
    """

    def __init__(
        self,
        nodes: list[int],
        prob: float = 0.0,
        rng: Generator = None,
    ) -> None:
        self.prob = prob
        self.nodes = nodes
        self.node = random.choice(self.nodes)
        self.rng = ensure_rng(rng)
        self.refresh_randomness()

    def refresh_randomness(self) -> None:
        # self.node = random.choice(self.nodes)
        # self.target_nodes = random.sample(self.nodes, self.n_targets)
        self.attack = bool(self.rng.uniform() < self.prob)

    def input_nodes(self, nodes: list[int]) -> Noise:
        """Return the noise to apply to input nodes."""
        return [(KrausChannel([]), [])]

    def command(self, cmd: Command) -> Noise:
        """Return the noise to apply to the command `cmd`."""
        if cmd.kind == CommandKind.M and cmd.node in self.nodes and self.attack:
            return [(dephasing_channel(prob=1), [self.node])]
        else:
            return [(KrausChannel([]), [])]

    def confuse_result(self, result: bool) -> bool:
        """Assign wrong measurement result cmd = "M"."""
        return result
