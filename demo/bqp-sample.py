import json
import random
from pathlib import Path
import numpy as np

bqp_error = 1/np.e
epsilon = 0.01


n_sample = 100
sampled_circuits=[]
with Path("circuits/table.json").open() as f:
    table = json.load(f)
    circuits = [name for name, prob in table.items() if (
        prob > bqp_error-epsilon
        and prob < bqp_error+epsilon
        ) or (
            prob > (1 - bqp_error) - epsilon
            and prob < (1-bqp_error) + epsilon
            )]

sampled_circuits = random.sample(circuits, n_sample)

with open('demo/sampled_circuits.txt', 'w') as outfile:
    for circuit in sampled_circuits:
        outfile.write(circuit+"\n")
print(sampled_circuits)

with Path("circuits/table.json").open() as f:
    table = json.load(f)
    for circuit in sampled_circuits:
        print(table[circuit])