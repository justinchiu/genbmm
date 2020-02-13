from __future__ import absolute_import, print_function

import tvm
import numpy as np


# bank conflicts?
num_step = tvm.var("num_step")
batch = tvm.var("batch")
k = tvm.var("k")

k1 = tvm.reduce_axis((0, k), name='k1')
k2 = tvm.reduce_axis((0, k), name='k2')

X = tvm.placeholder((num_step, batch, k, k), name="X")

s_state = tvm.placeholder((num_step, batch, k))
s_init = tvm.compute((1, batch, k), lambda a, b, c: 0.)

M = tvm.compute(
    (num_step, batch, k),
    lambda t, bb, ii: tvm.max(s_state[t-1, bb, k1] + X[t-1, bb, k, ii], axis=k1),
    name="M")

M2 = tvm.compute(
    (num_step, batch, k),
    lambda t, bb, ii: tvm.sum(tvm.exp(s_state[t-1, bb, k2] + X[t-1, bb, k2, ii]
                                      - M[t, bb, ii]), axis=k2),
    name="M2")
C = tvm.compute(
    (num_step, batch, k),
    #lambda t, bb, ii: M[t, bb, ii] + M2[t, bb,ii],
    lambda t, bb, ii: tvm.log(M2[t, bb, ii]) + M[t, bb, ii],
    name='C')

s_scan = tvm.scan(s_init, C, s_state, inputs=[X])
s = tvm.create_schedule(s_scan.op)

num_thread = 256
block_x = tvm.thread_axis("blockIdx.x")
thread_x = tvm.thread_axis("threadIdx.x")

xo, xi = s[s_init].split(s_init.op.axis[1], factor=num_thread)
s[s_init].bind(xo, block_x)
s[s_init].bind(xi, thread_x)

xo, xi = s[M].split(M.op.axis[1], factor=num_thread)
s[M].bind(xo, block_x)
s[M].bind(xi, thread_x)

xo, xi = s[M2].split(M2.op.axis[1], factor=num_thread)
s[M2].bind(xo, block_x)
s[M2].bind(xi, thread_x)

xo, xi = s[C].split(C.op.axis[1], factor=num_thread)
s[C].bind(xo, block_x)
s[C].bind(xi, thread_x)


print(tvm.lower(s, [X, s_scan], simple_mode=True))

import pdb; pdb.set_trace()

print("Building...")

fscan = tvm.build(s, [X, s_scan], "cuda", name="myscan")
ctx = tvm.gpu(0)
bsz = 32
t = 1024
k = 10
a_np = np.random.uniform(size=(t, bsz, k, k)).astype(s_scan.dtype)
a = tvm.nd.array(a_np, ctx)
b = tvm.nd.array(np.zeros((t, bsz, k), dtype=s_scan.dtype), ctx)
fscan(a, b)
tvm.testing.assert_allclose(b.asnumpy(), np.cumsum(a_np, axis=0))
print(a_np.shape)
print(np.cumsum(a_np, axis=0).shape)
print(b.asnumpy().shape)
import pdb; pdb.set_trace()
