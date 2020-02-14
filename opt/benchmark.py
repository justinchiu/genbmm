
import torch
import torch_struct as ts

import time

N, T, K = 64, 64, 128

num_trials = 20

device = torch.device("cuda:0")

torch.randn(5).cuda() + 1

log_potentials = torch.randn(N, T, K, K, device=device, dtype=torch.float32)

def get_time(fn, mask=None):
    fn(log_potentials)
    start = time.time()
    for _ in range(num_trials):
        fn(log_potentials)
    torch.cuda.synchronize(device)
    end = time.time()
    return (end - start) / num_trials

#from hmm2 import get_fb
from hmm3 import get_fb
def ts_marginals(x):
    return ts.LinearChain().marginals(x)
def ts_fast_marginals(x):
    return ts.LinearChain(ts.FastLogSemiring).marginals(x)

#fns = [ts_marginals, ts_fast_marginals, fb]
#fns = [ts_fast_marginals, get_fb(K)]
fns = [get_fb(K)]
for fn in fns:
    print(f"{fn}: {get_time(fn)}s")


mask = torch.randint(1, (N, T+1)).bool().to(log_potentials.device)
print(f"get_fb(mask): {get_time(fn, mask)}s")
