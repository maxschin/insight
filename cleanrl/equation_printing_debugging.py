from collections import namedtuple

import time
import numpy as np
import torch
import agents.eql.pretty_print as pretty_print
from agents.agent import Agent
import sympy as sy
from hackatari_utils import save_equations
from hackatari_env import HackAtariWrapper, SyncVectorEnvWrapper

# get all meta
var_names = SyncVectorEnvWrapper.get_variable_names_hardcoded_pong()
output_names = ["NOOP_1", "NOOP_2", "UP_1", "DOWN_1", "UP_2", "DOWN_2"]

# load agent
path = "../results_saved/agents/PongNoFrameskip-v4_Agent_close_but_no_hit_rf_final.pth"
device = torch.device("cpu")
agent = torch.load(path, weights_only=False, map_location=device)

agent_in_dim = 2048
#Args = namedtuple("args", ["n_layers", "deter_action"])
#args = Args(1,False)
#agent = Agent(
#    args=args,
#    agent_in_dim=agent_in_dim,
#    skip_perception=True,
#    n_actions=len(output_names)
#)

# extract equations as strings and symbolic expressions
outputs, expra = pretty_print.extract_equations(
    agent,
    var_names,
    output_names,
    accuracy=0.001,
    threshold=0.01,
    use_multiprocessing=True
)

# save_equations(outputs, ".", "test_run")

# Number of random samples to evaluate
n_samples = 1000

# Counters to accumulate statistics over samples
argmax_agreements = 0
mean_diff_total = 0.0

# Loop over n_samples random inputs
for sample_idx in range(n_samples):
    # Sample a random input vector of shape (1, agent_in_dim)
    x = torch.randn(1, agent_in_dim)
    
    # Evaluate the symbolic expressions:
    # Build a substitution dictionary mapping each symbol to the corresponding value from x.
    subs = {sy.symbols(var_names[i]): float(x[0, i]) for i in range(agent_in_dim)}
    sym_outputs = [expr.evalf(subs=subs) for expr in expra]
    sym_vals = np.array([float(val) for val in sym_outputs])
    
    # Evaluate the network's output
    eql_outputs = agent.eql_actor(x)
    net_vals = eql_outputs.detach().numpy().flatten()
    
    # Determine the argmax indices for both outputs
    net_argmax = np.argmax(net_vals)
    sym_argmax = np.argmax(sym_vals)
    if net_argmax == sym_argmax:
        argmax_agreements += 1
    
    # Compute mean absolute difference between the two outputs for this sample
    mean_diff_total += np.mean(np.abs(net_vals - sym_vals))

# Compute overall metrics
mean_of_mean_diffs = mean_diff_total / n_samples
argmax_agreement_rate = argmax_agreements / n_samples

# Print summary results
print("Summary:")
print(f"Mean of mean absolute differences: {mean_of_mean_diffs:.5f}")
print(f"Argmax agreement rate: {argmax_agreement_rate:.2%}")


