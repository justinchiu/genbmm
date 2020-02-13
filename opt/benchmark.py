
import torch
import torch_struct as ts

import time

N, T, K = 32, 32, 128

num_trials = 20

device = torch.device("cuda:2")

log_potentials = torch.randn(N, T, K, K, device=device, dtype=torch.float32)

def get_time(fn):
    start = time.time()
    for _ in range(num_trials):
        fn(log_potentials)
    torch.cuda.synchronize(device)
    end = time.time()
    return (end - start) / num_trials

from hmm2 import fb
def ts_marginals(x):
    return ts.LinearChain().marginals(x)
def ts_fast_marginals(x):
    return ts.LinearChain(ts.FastLogSemiring).marginals(x)

#fns = [ts_marginals, ts_fast_marginals, fb]
fns = [ts_fast_marginals, fb]
#fns = [fb]

for fn in fns:
    print(f"{fn}: {get_time(fn)}s")
