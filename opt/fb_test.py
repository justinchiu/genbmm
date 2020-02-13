import torch
import torch_struct as ts

def log_eye(K, dtype, device):
    x = torch.empty(K, K, dtype = dtype, device = device)
    x.fill_(float("-inf"))
    x.diagonal().fill_(0)
    return x

# just go with this one, then
def fwd(x):
    N, T, K, _ = x.shape
    alphas = torch.empty_like(x[:,:,0,:])
    alphas[:,0] = x[:,0].logsumexp(-1)
    for t in range(1, T):
        alphas[:,t] = (alphas[:,t-1].view(N, 1, K) + x[:,t]).logsumexp(-1)
    return alphas

def fwd0(x):
    N, T, K, _ = x.shape
    alphas = torch.empty_like(x[:,:,0,:])
    alphas[:,0] = x[:,0].logsumexp(-1).log_softmax(-1)
    for t in range(1, T):
        alphas[:,t] = (alphas[:,t-1].view(N, 1, K) + x[:,t]).logsumexp(-1).log_softmax(-1)
    return alphas

def fwd_wrong(x):
    N, T, K, _ = x.shape
    alphas = torch.empty_like(x[:,:,0,:])
    alphas[:,0] = x[:,0].logsumexp(-1).log_softmax(-1)
    for t in range(1, T):
        alphas[:,t] = (alphas[:,t-1].view(N, 1, K) + x[:,t]).logsumexp(-2).log_softmax(-1)
    return alphas

def bwd(x):
    N, T, K, _ = x.shape
    betas = torch.empty_like(x[:,:,0,:])
    betas[:,-1] = x[:,-1].logsumexp(-2)
    for t in range(T-2, -1, -1):
        betas[:,t] = (betas[:,t+1].view(N, K, 1) + x[:,t]).logsumexp(-2)
    return betas

def bwd0(x):
    x0 = x.flip(1).transpose(-1, -2).contiguous()
    return fwd(x0).flip(1)

def fwdbwd(x):
    N, T, K, _ = x.shape
    x = torch.cat([
        x,
        log_eye(K, dtype=x.dtype, device=x.device)
            .view(1, 1, K, K).expand(N, 1, K, K),
    ], dim=1)
    alphas = fwd(x)
    alphas0 = fwd0(x)
    betas = bwd(x)
    betas0 = bwd0(x)
    #return alphas[:,:-1] + betas[:,:-1].flip(1)
    return alphas, alphas0, betas, betas0

T = 8
N = 4
K = 16
V = 64

obs = torch.distributions.Categorical(probs = torch.ones(V)).sample((N, T))

init = torch.randn(K).double().log_softmax(0)
trans = (torch.randn(K, K) * 30).double().log_softmax(0)

emit = torch.randn(V, K).double()
emit[obs[0]] += 5
emit[obs[N // 2]] += 5
emit = emit.log_softmax(0)

# doesn't matter if you log_eye or augment first pairwise potential
# for sum or marginals. latter uses autograd so it's fine and sum is commutative.
chain = torch.cat([
    (log_eye(K, dtype=init.dtype, device=init.device)
        + init.unsqueeze(-1)).view(1, 1, K, K),
    trans.view(1, 1, K, K).expand(1, T-1, K, K),
], dim = 1)

Z = ts.LinearChain().sum(chain)
print(Z)
pz = ts.LinearChain().marginals(chain)
#print(pz)

px_z = emit[obs]

clamped_chain = px_z.view(N, T, K, 1) + chain
Zx = ts.LinearChain().sum(clamped_chain)
print(Zx)
# edge marginals
pz_x = ts.LinearChain().marginals(clamped_chain)
#print(pz_x)

ts_chain = ts.LinearChain.hmm(
    transition = trans,
    emission = emit,
    init = init,
    observations = obs,
)
print(ts.LinearChain().sum(ts_chain))
ts_marginals = ts.LinearChain().marginals(ts_chain)

alphas, alphas0, betas, betas0 = fwdbwd(clamped_chain)
alphas_wrong = fwd_wrong(
   torch.cat([
        clamped_chain,
        log_eye(K, dtype=clamped_chain.dtype, device=clamped_chain.device)
            .view(1, 1, K, K).expand(N, 1, K, K),
    ], dim=1)
)

gt_unary_marginals = pz_x.sum(-1)
unary_marginals = (alphas[:,:-1] + betas[:,1:]).softmax(-1)
#unary_marginals0 = (alphas0[:,:-1] + betas0[:,1:]).exp()
unary_marginals0 = (alphas0[:,:-1] + betas0[:,1:]).softmax(-1)
unary_wrong = (alphas_wrong[:,:-1] + betas0[:,1:]).softmax(-1)

assert torch.allclose(gt_unary_marginals, unary_marginals)
assert torch.allclose(gt_unary_marginals, unary_marginals0)
assert not torch.allclose(gt_unary_marginals, unary_wrong)

# edge marginals
edge_marginals = (
    alphas0[:,:-2].unsqueeze(-2) + clamped_chain[:, 1:] + betas[:, 2:].unsqueeze(-1)
).view(N, T-1, -1).log_softmax(-1).view(N, T-1, K, K).exp()

allclose = torch.allclose(pz_x[:,1:], edge_marginals)
print(allclose)
print((pz_x[:,1:] - edge_marginals).abs().max())

# time, batch, size, size
from hmm2 import fb, hmm_pytorch
marginals, alphas1, betas1 = fb(ts_chain.float().cuda())
print(f"Marginals match: {torch.allclose(ts_marginals.float().cuda(), marginals, rtol=1e-4, atol=1e-6)}")
print(f"Zx match: {torch.allclose(alphas1[-1].logsumexp(-1).cpu().double(), Zx)}")
