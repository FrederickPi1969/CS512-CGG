import numpy as np
import networkx as nx
import string
import random
import queue
from itertools import permutations

# Python code to insert a node in AVL tree 
  
# Generic tree node class 
class TreeNode(object): 
    def __init__(self, val): 
        self.val = val 
        self.left = None
        self.right = None
        self.height = 1
  
# AVL tree class which supports the  
# Insert operation 
class AVL_Tree(object): 
  
    # Recursive function to insert key in  
    # subtree rooted with node and returns 
    # new root of subtree. 
    def insert(self, root, key): 
      
        # Step 1 - Perform normal BST 
        if not root: 
            return TreeNode(key) 
        elif key < root.val: 
            root.left = self.insert(root.left, key) 
        else: 
            root.right = self.insert(root.right, key) 
  
        # Step 2 - Update the height of the  
        # ancestor node 
        root.height = 1 + max(self.getHeight(root.left), 
                           self.getHeight(root.right)) 
  
        # Step 3 - Get the balance factor 
        balance = self.getBalance(root) 
  
        # Step 4 - If the node is unbalanced,  
        # then try out the 4 cases 
        # Case 1 - Left Left 
        if balance > 1 and key < root.left.val: 
            return self.rightRotate(root) 
  
        # Case 2 - Right Right 
        if balance < -1 and key > root.right.val: 
            return self.leftRotate(root) 
  
        # Case 3 - Left Right 
        if balance > 1 and key > root.left.val: 
            root.left = self.leftRotate(root.left) 
            return self.rightRotate(root) 
  
        # Case 4 - Right Left 
        if balance < -1 and key < root.right.val: 
            root.right = self.rightRotate(root.right) 
            return self.leftRotate(root) 
  
        return root 
  
    def leftRotate(self, z): 
  
        y = z.right 
        T2 = y.left 
  
        # Perform rotation 
        y.left = z 
        z.right = T2 
  
        # Update heights 
        z.height = 1 + max(self.getHeight(z.left), 
                         self.getHeight(z.right)) 
        y.height = 1 + max(self.getHeight(y.left), 
                         self.getHeight(y.right)) 
  
        # Return the new root 
        return y 
  
    def rightRotate(self, z): 
  
        y = z.left 
        T3 = y.right 
  
        # Perform rotation 
        y.right = z 
        z.left = T3 
  
        # Update heights 
        z.height = 1 + max(self.getHeight(z.left), 
                        self.getHeight(z.right)) 
        y.height = 1 + max(self.getHeight(y.left), 
                        self.getHeight(y.right)) 
  
        # Return the new root 
        return y 
  
    def getHeight(self, root): 
        if not root: 
            return 0
  
        return root.height 
  
    def getBalance(self, root): 
        if not root: 
            return 0
  
        return self.getHeight(root.left) - self.getHeight(root.right) 
  
    def preOrder(self, root): 
  
        if not root: 
            return
  
        print("{0} ".format(root.val), end="") 
        self.preOrder(root.left) 
        self.preOrder(root.right) 

class BST_Tree(object): 
  
    def insert(self, root, key): 
      
        # Step 1 - Perform normal BST 
        if not root: 
            return TreeNode(key) 
        elif key < root.val: 
            root.left = self.insert(root.left, key) 
        else: 
            root.right = self.insert(root.right, key) 
  
        return root 

def generate_trees(n = 20):
    sequence = np.random.permutation(n)

    a = AVL_Tree()
    b = BST_Tree()
    a_root = None
    b_root = None
  
    for i in sequence:
        a_root = a.insert(a_root, i) 
        b_root = a.insert(b_root, i) 

    avl_graph = nx.null_graph()
    avl_queue = queue.Queue()
    avl_queue.put(a_root)
    print(a_root.val)
    while not avl_queue.empty():
        current = avl_queue.get()
        if current.left:
            avl_graph.add_edge(current.val, current.left.val)
            avl_queue.put(current.left)
        if current.right:
            avl_graph.add_edge(current.val, current.right.val)
            avl_queue.put(current.right)

    bst_graph = nx.null_graph()
    bst_queue = queue.Queue()
    bst_queue.put(b_root)
    print(b_root.val)
    while not bst_queue.empty():
        current = bst_queue.get()
        if current.left:
            bst_graph.add_edge(current.val, current.left.val)
            bst_queue.put(current.left)
        if current.right:
            bst_graph.add_edge(current.val, current.right.val)
            bst_queue.put(current.right)

    return (avl_graph, bst_graph)

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

def sample_subgraph(G, target_size = 20, max_size = 24, start = None):
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
        if len(subgraph) >= max_size:
            break
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
