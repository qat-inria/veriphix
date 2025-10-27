from graphix.transpiler import Circuit
from perceval import Source

from veriphix.client import Client, Secrets
from veriphix.verifying import TrappifiedSchemeParameters
from veriphix.perceval_backend import PercevalBackend

# client computation pattern definition
circ = Circuit(1)
circ.h(0)
circ.h(0)

pattern = circ.transpile().pattern
pattern.standardize()

# client definition
secrets = Secrets(r=True, a=True, theta=True)
d = 10
t = 10
w = 1
trap_scheme_param = TrappifiedSchemeParameters(d, t, w)
client = Client(pattern=pattern, secrets=secrets, parameters=trap_scheme_param)
protocol_runs = client.sample_canvas()

source = Source(emission_probability = 1, 
                multiphoton_component = 0, 
                indistinguishability = 1)

backend = PercevalBackend(source)
outcomes = client.delegate_canvas(protocol_runs, backend)
result = client.analyze_outcomes(protocol_runs, outcomes)
print(result)