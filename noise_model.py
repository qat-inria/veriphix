import numpy as np
from graphix.channels import KrausChannel, depolarising_channel, two_qubit_depolarising_channel
from graphix.noise_models.noise_model import NoiseModel


class VBQCNoiseModel(NoiseModel):
    """Test noise model for testing.
    Only return the identity channel.

    :param NoiseModel: Parent abstract class class:`graphix.noise_model.NoiseModel`
    :type NoiseModel: class
    """

    def __init__(
        self,
        prepare_error_prob: float = 0.0,
        x_error_prob: float = 0.0,
        z_error_prob: float = 0.0,
        entanglement_error_prob: float = 0.0,
        measure_channel_prob: float = 0.0,
        measure_error_prob: float = 0.0,
    ) -> None:
        self.prepare_error_prob = prepare_error_prob
        self.x_error_prob = x_error_prob
        self.z_error_prob = z_error_prob
        self.entanglement_error_prob = entanglement_error_prob
        self.measure_error_prob = measure_error_prob
        self.measure_channel_prob = measure_channel_prob
        # TODO: fix that somehow??
        self.rng = np.random.default_rng()

    def prepare_qubit(self) -> KrausChannel:
        """return the channel to apply after clean single-qubit preparation. Here just identity."""
        return depolarising_channel(self.prepare_error_prob)

    def entangle(self) -> KrausChannel:
        """return noise model to qubits that happens after the CZ gate"""
        # return two_qubit_depolarising_tensor_channel(self.entanglement_error_prob)
        return two_qubit_depolarising_channel(self.entanglement_error_prob)

    def measure(self) -> KrausChannel:
        """apply noise to qubit to be measured."""
        return depolarising_channel(self.measure_channel_prob)

    def confuse_result(self, result: bool) -> bool:
        """assign wrong measurement result
        cmd = "M"
        """
        if self.rng.uniform() < self.measure_error_prob:
            return not result
        else:
            return result

    def byproduct_x(self) -> KrausChannel:
        """apply noise to qubits after X gate correction"""
        return depolarising_channel(self.x_error_prob)

    def byproduct_z(self) -> KrausChannel:
        """apply noise to qubits after Z gate correction"""
        return depolarising_channel(self.z_error_prob)

    def clifford(self) -> KrausChannel:
        """apply noise to qubits that happens in the Clifford gate process"""
        # TODO list separate different Cliffords to allow customization
        return KrausChannel([{"coef": 1.0, "operator": np.eye(2)}])

    def tick_clock(self) -> None:
        """notion of time in real devices - this is where we apply effect of T1 and T2.
        we assume commands that lie between 'T' commands run simultaneously on the device.
        """
