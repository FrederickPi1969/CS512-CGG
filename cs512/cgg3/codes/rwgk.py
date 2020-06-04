import numpy as np

## A1/A2: adjacency matrices (column normalized)
## L1/L2: attribute matrices (row normalized)
def kernel(A1, A2, L1, L2):
    nm = np.shape(L1)[0] * np.shape(L2)[0]
    l = np.shape(L1)[1]
    Lx = np.zeros((nm, nm))
    for k in range(l):
        #print(Lx)
        Lx = np.add(Lx, np.kron(np.diag(L1[:,k]), np.diag(L2[:,k])))
    y = x = np.ones(nm)
    I = np.diag(np.ones(nm))
    y /= nm
    x /= nm
    c = 0.95
    Ax = np.matmul(Lx, np.kron(np.transpose(A1), np.transpose(A2)))
    return np.transpose(y) @ np.matmul(np.linalg.inv(np.subtract(I, c * Ax)), Lx) @ x
