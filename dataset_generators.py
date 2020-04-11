import numpy as np
import networkx as nx
import string
import random
import queue
from itertools import combinations

def sample_subgraph(G, start = None, prob = 0.8):
    subgraph = nx.null_graph()
    initial_node = start
    if not start or start not in G.nodes():
        initial_node = np.random.choice(list(G.keys()))
    
    node_queue = Queue()
    node_queue.put((initial_node), prob)

    ## sample nodes
    while not node_queue.empty():
        new_node = node_queue.pop()
        subgraph.add_node(new_node[0])
        for neighbor in G.neighbors(new_node[0]):
            if random.random() > new_node[1]:
              node_queue.put((neighbor, new_node[1] * prob))

    ## add edges
    for node in subgraph.nodes():
        for neighbor in list(intersection(set(G.neighbors(node)), set(subgraph.nodes()))):
            subgraph.add_edge(neighbor, node)

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
