import numpy as np
import networkx as nx
import string
import random
import queue
from itertools import permutations

def sample_subgraph(G, prob = 0.8, start = None):
    subgraph = nx.null_graph()
    initial_node = start
    if not start or start not in G.nodes():
        initial_node = np.random.choice(list(G.nodes()))
    
    node_queue = queue.Queue()
    node_queue.put((initial_node, prob))
    seen = set()
    seen.add(initial_node)

    ## sample nodes
    while not node_queue.empty():
        new_node = node_queue.get()
        subgraph.add_node(new_node[0])
        for neighbor in set(G.neighbors(new_node[0])) - seen:
            roll = random.random()
            seen.add(neighbor)
            if roll < new_node[1]:
                node_queue.put((neighbor, new_node[1] * prob))

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
