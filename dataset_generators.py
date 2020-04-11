import numpy as np
import networkx as nx
import string
import random
import queue
from itertools import permutations

def dblp():
    name_dict = {}
    nodes = open('authorDict.txt', 'r').readlines()
    count = 0
    for node in nodes:
        name_dict[count] = node.replace("\n", "")
        count += 1
    
    G = nx.null_graph()
    edges = open('dblp_edges.txt', 'r').readlines()
    for edge in edges:
        temp = edge.replace("\n", "").split() 
        G.add_edge(name_dict[int(temp[0])], name_dict[int(temp[1])])

    return G

def sample_subgraph(G, target_size = 20, start = None):
    subgraph = nx.null_graph()
    initial_node = start
    if not start or start not in G.nodes():
        initial_node = np.random.choice(list(G.nodes()))
    
    node_queue = queue.Queue()
    node_queue.put((initial_node, 1))
    prob_sum = 1
    seen = set()
    seen.add(initial_node)

    ## sample nodes
    while not node_queue.empty():
        new_node = node_queue.get()
        prob_sum -= new_node[1]
        roll = random.random()
        if roll > new_node[1]:
            continue
        subgraph.add_node(new_node[0])
        for neighbor in set(G.neighbors(new_node[0])) - seen:
            seen.add(neighbor)
            select_prob = max(0, 1 - ((len(subgraph) + prob_sum) / target_size))
            node_queue.put((neighbor, select_prob))
            prob_sum += select_prob

    ## add edges
    for edge in set(permutations(list(subgraph.nodes()), 2)).intersection(set(G.edges())):
        subgraph.add_edge(*edge)

    return subgraph

def square():
  test_graph = nx.null_graph()
  test_graph.add_node(0)
  test_graph.add_node(1)
  test_graph.add_node(2)
  test_graph.add_node(3)
  test_graph.add_edge(0, 1)
  test_graph.add_edge(0, 2)
  test_graph.add_edge(0, 3)
  test_graph.add_edge(1, 2)
  test_graph.add_edge(1, 3)
  test_graph.add_edge(2, 3)
  return test_graph

def triangle():
  test_graph = nx.null_graph()
  test_graph.add_node(0)
  test_graph.add_node(1)
  test_graph.add_node(2)
  test_graph.add_edge(0, 2)
  test_graph.add_edge(0, 1)
  test_graph.add_edge(1, 2)
  return test_graph

def line(n):
  test_graph = nx.null_graph()
  for i in range(n):
    test_graph.add_node(i)
    if i > 0:
      test_graph.add_edge(i, i - 1)
  return test_graph
