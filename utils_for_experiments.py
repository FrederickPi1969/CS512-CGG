import networkx as nx
import numpy as np
import torch

def load_all_params(operation_name):
    """
    load all pretrained model
    """
    vae = torch.load("vae.model")

    if operation_name == "density":
        w = torch.load("w_density.pt")
        a_w1 = torch.load("a_w1_density.pt")
        a_w2 = torch.load("a_w2_density.pt")
        a_b1 = torch.load("a_b1_density.pt")
        a_b2 = torch.load("a_b2_density.pt")


    else:
        raise NameError("operation model not trained")

    return vae, w, a_w1,a_w2, a_b1, a_b2



def topological_measure(batched_A):
    # calculate average
    # assert type(batched_A) is list, "Passed argument expected to be a list of batched adjacency matrix!"
    As = torch.cat(batched_A, dim=0).squeeze(-1)
    density, diameter, cluster_coef, edges, avg_degree = 0,0,0,0,0
    for i in range(len(As)):
        de,di,cl,ed,avd = topol_one_graph(As[i])
        density += de
        diameter += di
        cluster_coef += cl
        edges += ed
        avg_degree += avd

    density /= len(As)
    diameter /= len(As)
    cluster_coef /= len(As)
    edges /= len(As)
    avg_degree /= len(As)
    return density, diameter, cluster_coef, edges, avg_degree


def topol_one_graph(A):
    g = nx.from_numpy_matrix(A.numpy())
    density = nx.density(g)

    if nx.is_connected(g):
        diameter = nx.diameter(g)
    else:
        diameter = -1

    cluster_coef = nx.average_clustering(g)

    # if g.number_of_edges() > 2 and len(g) > 2:
    #    assort = nx.degree_assortativity_coefficient(g, x='out', y='in')
    # else:
    #    assort = 0

    edges = g.number_of_edges()
    avg_degree = sum(i for i in nx.degree_centrality(g).values()) / len(nx.degree_centrality(g).keys())

    topol = [density, diameter, cluster_coef, edges, avg_degree]

    return topol
