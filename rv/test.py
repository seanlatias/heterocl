import heterocl as hcl
import numpy as np

A = hcl.placeholder((10, 10), "A")

def kernel_raw(A):
    with hcl.for_(2, 10) as i:
        A[i, 0] = A[i-2, 0] + 10
    return hcl.compute(A.shape, lambda x, y: A[x, y], "B")

def kernel_raw2(A):
    with hcl.for_(2, 10) as i:
        with hcl.for_(2, 10) as j:
            A[i, j] = A[i-2, j-2] + 10
    return hcl.compute(A.shape, lambda x, y: A[x, y], "B")

def kernel_reuse(A):
    B = hcl.compute((10, 8), lambda x, y: A[x, y] + A[x, y+1] + A[x, y+2], "B")
    return hcl.compute(A.shape, lambda x, y: 0, "C")

def kernel_reuse2(A):
    B = hcl.compute((8, 10), lambda x, y: A[x, y] + A[x+1, y] + A[x+2, y], "B")
    return hcl.compute(A.shape, lambda x, y: 0, "C")

def kernel_reuse3(A):
    B = hcl.compute((8, 8), lambda x, y: A[x, y] + A[x+1, y+1] + A[x+2, y+2], "B")
    return hcl.compute(A.shape, lambda x, y: 0, "C")

def kernel_stream(A):
    B = hcl.compute(A.shape, lambda x, y: A[x, y], "B")
    return hcl.compute(A.shape, lambda x, y: B[x, y], "C")

def kernel_stream2(A):
    B = hcl.compute(A.shape, lambda x, y: A[x, y], "B")
    C = hcl.compute(A.shape, lambda x, y: B[x, y], "C")
    return hcl.compute(A.shape, lambda x, y: C[x, y], "D")

s = hcl.create_schedule([A], kernel_raw)
f = hcl.build(s)

a = np.random.randint(0, 100, size=(10, 10))
b = np.zeros((10, 10))

hcl_a = hcl.asarray(a)
hcl_b = hcl.asarray(b)

f(hcl_a, hcl_b)
