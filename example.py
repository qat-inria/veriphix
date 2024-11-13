import graphix.command
from graphix.sim.statevec import StatevectorBackend
from graphix.sim.density_matrix import DensityMatrixBackend
from graphix.states import BasicStates
from graphix.transpiler import Circuit

from noise_model import VBQCNoiseModel
from veriphix.client import Client, Secrets

# smallest deterministic BQP comp with XY plane measurements only (gadget)
circ = Circuit(1)
circ.h(0)
circ.h(0)
circ.h(0)
circ.h(0)

pattern = circ.transpile().pattern
pattern.standardize()
print(f"pattern {list(pattern)}")
# don't forget to add in the output nodes that are not initially measured!
# for this example expect a |+> state so ok.
# TODO absorb corrections in last meas
for onode in pattern.output_nodes:
    pattern.add(graphix.command.M(node=onode))


pattern.standardize()
print(f"pattern {list(pattern)}")


print(f"output_nodes {pattern.output_nodes}")


noise_model = VBQCNoiseModel(
    prepare_error_prob=0., # 0.7
    x_error_prob=0.0,
    z_error_prob=0.0,
    entanglement_error_prob=0.0,
    measure_channel_prob=0.0,
    measure_error_prob=0.0,
)

# useless but OK.
states = [BasicStates.PLUS for _ in pattern.input_nodes]

# classical input so don't need the a bit
# actually this doesn't really work when adding the last measurement. 
# Why??? No more output nodes so weird things happening??
# be careful with no output nodes.
# Work on BQPify methiod to measure "output nodes"!
# recheck what the problem was....

secrets = Secrets(r=True, a=True, theta=True)
client = Client(pattern=pattern, input_state=states, secrets=secrets)

print(f"client secrets {client.secrets}")
test_runs = client.create_test_runs()

# reinit the backend every time
# TODO reinitialise the secrets too!!!!!!!
# modify client to allow that.
for run in test_runs:
    # backend = StatevectorBackend()
    backend = DensityMatrixBackend(pr_calc=True)
    trap_outcomes = client.delegate_test_run(backend=backend, run=run, noise_model=noise_model) 
    print("Client leasurel db ", client.measurement_db)
    print('trap', trap_outcomes)
    assert trap_outcomes == [0 for _ in run.traps_list]


# backend = StatevectorBackend()
backend = DensityMatrixBackend(pr_calc=True)
# Blinded simulation, between the client and the server

for _ in range(10):
    client.delegate_pattern(backend, noise_model=noise_model)

# add a state attribute to the client?
# no last measurement: get |+> state as expected.
    print(client.results, backend.state) # noise_model=noise_model

    assert client.results[4] == 0



# todo : add execution up to certain command to check??

# NOTE: if examples folder: "no module named veriphix"
