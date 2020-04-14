import numpy as np
import networkx as nx
import string
import random
import queue
from itertools import permutations

## A1/A2: adjacency matrices (column normalized), A2 = edit(G(z)), A1 = G(z+aw)
## L1/L2: attribute matrices (row normalized)
def kernel(A1, A2, L1, L2):
    A1 = A1/np.linalg.norm(A1, ord=1, axis=0, keepdims=True)
    A2 = A2/np.linalg.norm(A2, ord=1, axis=0, keepdims=True)
    L1 = L1 - L1.min(axis = 1)
    L2 = L2 - L2.min(axis = 1)
    L1 = L1/np.linalg.norm(L1, ord=1, axis=1, keepdims=True)
    L2 = L2/np.linalg.norm(L2, ord=1, axis=1, keepdims=True)

    c = 0.0001
    
    n1 = np.shape(A1)[0]
    n2 = np.shape(A2)[0]
    p1 = np.ones(n1) / n1
    q1 = np.ones(n1) / n1
    p2 = np.ones(n2) / n2
    q2 = np.ones(n2) / n2

    l = np.shape(L1)[1]
    Lx = np.zeros((n1 * n2, n1 * n2))
    for k in range(l):
        #print(Lx)
        Lx = np.add(Lx, np.kron(np.diag(L1[:,k]), np.diag(L2[:,k])))
    Ax = np.kron(A1,A2);
    qx = np.kron(q1,q2);
    px = np.kron(p1,p2);
    sim = np.transpose(qx).dot(np.linalg.inv(np.identity(n1 * n2) - c * np.matmul(Lx, Ax))).dot(Lx).dot(px)

    Lx_self = np.zeros((n2 * n2, n2 * n2))
    for k in range(l):
        #print(Lx)
        Lx_self = np.add(Lx_self, np.kron(np.diag(L2[:,k]), np.diag(L2[:,k])))
    Ax_self = np.kron(A2, A2)
    qx_self = np.kron(q2, q2);
    px_self = np.kron(p2, p2);
    sim_self = np.transpose(qx_self).dot(np.linalg.inv(np.identity(n2 * n2) - c * np.matmul(Lx_self, Ax_self))).dot(Lx_self).dot(px_self)

    return min(sim.item(), sim_self.item()) / max(sim.item(), sim_self.item())
