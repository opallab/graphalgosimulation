import random
import signal
from collections import deque
from contextlib import contextmanager
from collections import namedtuple

import torch
from ltfs import config

def track_changes(X_old, X_new, index):
    indexes = (X_old - X_new).nonzero()[:, 1].unique()
    fields = []
    for i in indexes:
        field = next(k for k, v in index.items() if v.start <= i < v.stop)
        if field not in fields:
            fields.append(field)
    return fields

def pe2num(p_i):
    num = torch.arctan2(p_i[0], p_i[1]) / config.eps
    if num < 0:
        num += 2 * torch.pi / config.eps
    return num.round().int().item()

def decode_output(X, index):
    return [pe2num(x)-1 for x in X[1:, index["OUT"]]], X[1:, index["D"]].tolist()

# Reference algorithms

def reference_dfs(adj_matrix, root):
    # Number of nodes in the graph
    num_nodes = len(adj_matrix)
    
    # Initialize parent array with None
    parent = list(range(num_nodes))
    reached = [False] * num_nodes
    
    # Helper function for recursive DFS
    def dfs_recursive(node):
        reached[node] = True
        for neighbor in range(num_nodes):
            if adj_matrix[node][neighbor] == 1 and not reached[neighbor]:
                parent[neighbor] = node
                dfs_recursive(neighbor)
    
    # Mark the root as its own parent
    parent[root] = root
    
    # Start DFS from the root node
    dfs_recursive(root)
    
    return parent


def reference_bfs(adj_matrix, root):
    # Number of nodes in the graph
    num_nodes = len(adj_matrix)
    
    # Initialize parent as itself
    parent = list(range(num_nodes))
    
    # Create a queue for BFS
    queue = deque()
    
    # Mark the root as its own parent
    parent[root] = root
    reached = [False] * num_nodes
    
    # Enqueue the root node
    queue.append(root)
    
    while queue:
        # Dequeue a node from the queue
        current_node = queue.popleft()
        reached[current_node] = True
        
        # Visit all adjacent nodes of the current node
        for neighbor in range(num_nodes):
            if adj_matrix[current_node][neighbor] == 1 and not reached[neighbor]:
                # Mark the neighbor as visited and set its parent
                reached[neighbor] = True
                parent[neighbor] = current_node
                queue.append(neighbor)
    
    return parent

def reference_dijkstra(adj_matrix, root):
    num_nodes = adj_matrix.shape[0]
    D = torch.ones(num_nodes)*config.INF
    D[root] = 0
    V = torch.zeros(num_nodes)
    Pr = torch.arange(num_nodes)

    while 0 in V:
        argmin = torch.argmin(D + V*config.INF)
        for i, v in enumerate(adj_matrix[argmin]):
            if v != 0 and v + D[argmin] < D[i]:
                D[i] = v + D[argmin]
                Pr[i] = argmin
        V[argmin] = 1
    return Pr.int().tolist(), D.tolist()

def reference_scc(adj_matrix):
    n = adj_matrix.shape[0]
    visited = [False] * n
    stack = []

    def fill_order(v, visited, stack, matrix):
        visited[v] = True
        for i in range(len(matrix)):
            if matrix[v][i] == 1 and not visited[i]:
                fill_order(i, visited, stack, matrix)
        stack.append(v)
    
    def dfs(v, visited, scc, matrix):
        visited[v] = True
        scc.append(v)
        for i in range(len(matrix)):
            if matrix[v][i] == 1 and not visited[i]:
                dfs(i, visited, scc, matrix)

    # First pass: Fill stack based on finishing times
    for i in range(n):
        if not visited[i]:
            fill_order(i, visited, stack, adj_matrix)

    # Second pass: Get strongly connected components
    transposed_matrix = adj_matrix.T
    visited = [False] * n
    sccs_indexes = list(range(n))
    sccs = []
    while stack:
        i = stack.pop()
        if not visited[i]:
            scc = []
            dfs(i, visited, scc, transposed_matrix)
            sccs.append(scc)
            for j in scc:
                sccs_indexes[j] = i

    return sccs_indexes


def generate_A(num_nodes: int, seed: int = 0, edge_ratio: float = 0.5):

    matrix = torch.zeros((num_nodes, num_nodes))

    # Step 1: Create a spanning tree to guarantee connectivity.
    nodes = list(range(num_nodes))
    generator = random.Random(seed)
    generator.shuffle(nodes)

    for i in range(1, num_nodes):
        weight = generator.randint(1, 10)
        matrix[nodes[i-1], nodes[i]] = weight
        matrix[nodes[i], nodes[i-1]] = weight

    # Step 2: Add some additional edges with weights to make the graph more interesting.
    # The exact number "m" can be adjusted as desired.
    m = int(edge_ratio*num_nodes)
    m = generator.randint(m, m*(m-1)//2)  # At max, there can be n(n-1)/2 edges in an undirected graph

    count = 0
    while count < m:
        i, j = generator.randint(0, num_nodes-1), generator.randint(0, num_nodes-1)
        if i != j and matrix[i, j] == 0:  # Avoid self-loops and overwrite existing edges
            weight = generator.randint(1, 10)
            matrix[i, j] = weight
            matrix[j, i] = weight
            count += 1

    return matrix


def evaluate_clrs(model, dataset, input_fn, time_limit: int, output_col: str = "OUT", device=None):
    device = torch.device("cpu") if device is None else device
    
    input = next(iter(dataset.as_numpy_iterator()))
    features, outputs = input.features, input.outputs
    A_base = features.inputs[0].data
    labels = outputs[0].data

    if len(features.inputs) > 3:
        s_base = features.inputs[-1].data
        s_base = [x.nonzero()[0][0] for x in s_base]
    else:
        s_base = [0] * A_base.shape[0]

    acc = 0
    num_samples = A_base.shape[0]
    wrong, exceptions, predictions = [], [], []    
    for i in range(num_samples):
        
        A = torch.from_numpy(A_base[i].copy()).double()
        s = s_base[i]
        
        X, index = input_fn(len(A), s)
        X, A = X.to(device), A.to(device)

        try:
            with timeout(time_limit):
                output = model(X, A)
                pos_pred = output[1:, index[output_col]].flatten().tolist()
                pos_pred = [round(x) for x in pos_pred]
                predictions.append(pos_pred)
                acc_i = (pos_pred == labels[i].tolist())
                if not acc_i:
                    wrong.append(i)
                acc += acc_i
        
        except RuntimeError:
            exceptions.append(i)
            predictions.append([])

        print(f"({i+1}) | acc: {acc / (i+1)} | (C/I/E): {acc}/{len(wrong)}/{len(exceptions)}", end="\r")
    
    print(f"({i+1}) | acc: {acc / (i+1)} | (C/I/E): {acc}/{len(wrong)}/{len(exceptions)}")
    return acc / num_samples, wrong, exceptions, predictions


@contextmanager
def timeout(seconds):
    def timeout_handler(signum, frame):
        raise RuntimeError("Code execution timed out")

    # Set a signal handler for SIGALRM
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Cancel the alarm and restore the default signal handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)