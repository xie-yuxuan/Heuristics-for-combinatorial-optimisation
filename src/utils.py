import networkx as nx
import numpy as np
import json
from networkx.readwrite import json_graph


"""
Utility functions including:
- calc_cost
- calc_delta_cost
- calc_delta_cost_edge (no longer used, logic included in optimise function)
"""

def load_graph_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # load graph_name and attibutes
    graph_name = data["graph_name"]
    num_groups = data["num_groups"]
    num_nodes = data["num_nodes"]
    group_mode = data["group_mode"]
    initial_node_colors = data["initial_node_colors"]
    ground_truth_log_likelihood = data["ground_truth_log_likelihood"]

    graph_data = data["graph_data"]
    graph = json_graph.node_link_graph(graph_data)

    # uncomment to get adj matrix
    # adj_matrix = nx.adjacency_matrix(graph, weight='weight').toarray()
    
    return graph, graph_name, num_nodes, num_groups, group_mode, initial_node_colors, ground_truth_log_likelihood

def compute_w2(graph, total_groups):
    w = 0
    # initialise n and m
    # n is 1D array that stores the number of nodes in each group
    # m is 2D array that stores the number of edges between groups
    n, m = np.zeros(total_groups), np.zeros((total_groups, total_groups))

    for node in graph.nodes():
        n[graph.nodes[node]['color']] += 1 # increment group count for each group
    # print(n)
    for u, v in graph.edges():
        # increment edge count between groups
        # ensures m is symmetric
        m[graph.nodes[v]['color'], graph.nodes[u]['color']] = m[graph.nodes[u]['color'], graph.nodes[v]['color']] = m[graph.nodes[u]['color'], graph.nodes[v]['color']] + 1 
    # print(m)
    # print(np.outer(n,n)-np.diag(0.5*n*(n+1)))
    # Suppress warnings and handle division safely
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.divide(
            m,
            (np.outer(n, n) - np.diag(0.5 * n * (n + 1))),
            where=(np.outer(n, n) - np.diag(0.5 * n * (n + 1))) != 0
        )
    
    
    # subtracts within-group combinations for diagonal elements

    return w

def compute_w(graph, total_groups):
    """
    Compute the symmetric w matrix for the graph based on group memberships.
    Ensures the w matrix remains of size total_groups x total_groups,
    with rows/columns for missing groups set to zero.

    Parameters:
        graph (nx.Graph): The graph with group memberships in the 'color' attribute.
        total_groups (int): The total number of groups (fixed size for w matrix).

    Returns:
        ndarray: The symmetric matrix of edge probabilities with size total_groups x total_groups.
    """
    # Get node groups (colors) and initialize counts
    groups = np.array([graph.nodes[node]['color'] for node in graph.nodes()])
    n = np.zeros(total_groups)                 # Node counts per group
    m = np.zeros((total_groups, total_groups))  # Edge counts between groups

    # Count node group sizes
    for group in groups:
        n[group] += 1

    # Count edges between groups
    for u, v in graph.edges:
        gu, gv = groups[u], groups[v]
        m[gu, gv] += 1
        if gu != gv:  # Symmetric for between-group edges
            m[gv, gu] += 1

    # Compute w matrix with fixed size NxN
    w = np.zeros((total_groups, total_groups))
    for i in range(total_groups):
        for j in range(total_groups):
            if i == j:  # Within-group
                if n[i] > 1:
                    w[i, j] = m[i, j] / (0.5 * n[i] * (n[i] - 1))
                else:
                    w[i, j] = 0  
            else:  # Between-group
                if n[i] > 0 and n[j] > 0:
                    w[i, j] = m[i, j] / (n[i] * n[j])
                else:
                    w[i, j] = 0  

    return w

def calc_log_likelihood3(graph, w):
    log_likelihood = 0

    g = np.array([graph.nodes[node]['color'] for node in graph.nodes])
    num_nodes = graph.number_of_nodes()
    A = nx.to_numpy_array(graph, nodelist=range(num_nodes))

    g_reshaped = g[:, np.newaxis]  # Shape (n, 1)
    probabilities = w[g_reshaped, g]  # Broadcasting to get the pairwise probabilities

    edge_contributions = (A*np.log(probabilities))
    non_edge_contributions = (1 - A) * np.log(1 - probabilities)

    # replace Nan with 0
    edge_contributions = np.nan_to_num(edge_contributions, nan=0.0, posinf=0.0, neginf=0.0)
    non_edge_contributions = np.nan_to_num(non_edge_contributions, nan=0.0, posinf=0.0, neginf=0.0)
    
    log_likelihood = np.sum(np.triu(edge_contributions + non_edge_contributions))

    return log_likelihood

def calc_log_likelihood2(graph, w, total_groups): # or pass g, n, m in here as argument, not graph

    n, m = np.zeros(total_groups), np.zeros((total_groups, total_groups))
    # g = np.array(color for color in graph.nodes)

    for node in graph.nodes():
        n[graph.nodes[node]['color']] += 1 # increment group count for each group
    # print(n)
    for u, v in graph.edges():
        # increment edge count between groups
        # ensures m is symmetric
        m[graph.nodes[v]['color'], graph.nodes[u]['color']] = m[graph.nodes[u]['color'], graph.nodes[v]['color']] = m[graph.nodes[u]['color'], graph.nodes[v]['color']] + 1 

    # print(n, m)
    #e = 1e-10  # Small value to avoid log(0) or log(1)
    

    log_likelihood = np.nansum(
        np.triu(
            m * np.log(w) + (np.outer(n, n) - np.diag(0.5 * n * (n + 1)) - m) * np.log(1 - w)
        )
    ) # \sum_{r>s} m_rs log w_rw + (n_r n_s -m_rs ) log(1-w_rs) + \sum_r m_rr log w_rr + (nr(nr-1)/2 - m_rr) log(1-w_rr)

    return log_likelihood



def calc_log_likelihood(graph, w):
    """
    Calculate the log-likelihood for the graph given the w matrix.
    
    Parameters:
        graph (nx.Graph): The graph with node colors assigned.
        w (ndarray): Precomputed w matrix based on group memberships.

    Returns:
        float: Log-likelihood of the current graph configuration.
    """

    groups = np.array([graph.nodes[node]['color'] for node in graph.nodes])  # Group membership of each node, len of this list is num_nodes
    log_likelihood = 0
    epsilon = 1e-10  # Small constant to avoid log(0)

    # Iterate over all pairs of nodes
    for u in graph.nodes:
        for v in graph.nodes:
            if u != v:  # Skip self-loops
                gu, gv = groups[u], groups[v]  # Group indices of the nodes
                # print(gu, gv)
                p = max(epsilon, min(1 - epsilon, w[gu, gv]))  # Clamp probabilities
                if graph.has_edge(u, v):
                    log_likelihood += np.log(p)  # Add log(w_{g_i g_j})
                else:
                    log_likelihood += np.log(1 - p)  # Add log(1 - w_{g_i g_j})
    
    return log_likelihood

# def compute_w(graph): # TODO: one for loop over list of edges, 4 lines
#     """
#     Compute the symmetric w matrix for the graph based on group memberships.
    
#     Parameters:
#         graph (nx.Graph): The graph with initial group memberships in the 'color' attribute.

#     Returns:
#         ndarray: The symmetric matrix of edge probabilities.
#     """
#     adj_matrix = nx.to_numpy_array(graph, nodelist=range(G.number_of_nodes())) # adjacency matrix
#     groups = np.array([graph.nodes[node]['color'] for node in graph.nodes]) # get list of node colors / group membership
#     unique_groups = np.unique(groups)
    
#     group_counts = {g: np.sum(groups == g) for g in unique_groups} # for 2 colors, 10 nodes we have {0:3, 1:7}
#     # print(group_counts)
#     w = np.zeros((len(unique_groups), len(unique_groups)))
    
#     for i, g1 in enumerate(unique_groups): #TODO: iterate through edges instead of groups to not double count, init m matrix of 0s, size NxN
#         for j, g2 in enumerate(unique_groups):
#             if i == j:  # Within-group
#                 m_gg = np.sum(adj_matrix[np.ix_(groups == g1, groups == g1)]) # np.ix_ to get a submatrix, m_gg is the total number of edges in group g, A is all pairwise connections in graph
#                 n_g = group_counts[g1]
#                 w[i, j] = m_gg / (0.5 * n_g * (n_g - 1)) if n_g > 1 else 0
#             else:  # Between-group
#                 m_gg = np.sum(adj_matrix[np.ix_(groups == g1, groups == g2)])
#                 n_g1, n_g2 = group_counts[g1], group_counts[g2]
#                 w[i, j] = m_gg / (n_g1 * n_g2) if n_g1 > 0 and n_g2 > 0 else 0

#     return w

def calc_cost(graph):
    cost = 0
    
    vertex_colors = nx.get_node_attributes(graph, 'color')
    
    for vertex_1, vertex_2, edge_data in graph.edges(data=True):
        if vertex_colors.get(vertex_1) == vertex_colors.get(vertex_2): # Check if connected vertices have the same color
            cost += edge_data.get('weight')

    return cost

def calc_delta_cost(graph, vertex, ori_color, new_color):
    """
    Calc cost reduction (delta) when a vertex is recolored, +ve means cost is reduced
    """
    delta = 0

    for neighbor in graph.neighbors(vertex):
        neighbor_color = graph.nodes[neighbor]['color']

        if ori_color == neighbor_color:
            delta += graph[vertex][neighbor].get('weight')

        if new_color == neighbor_color:
            delta -= graph[vertex][neighbor].get('weight')

    return delta

def calc_delta_cost_edge(graph, node, node_color_bef, node_color_aft, neighbor_node, neighbor_color_bef, neighbor_color_aft):
    """
    calc cost reduction for neighbor when a node is recolored, just calculating the difference by one edge connected to node
    possible Cases:
    1. same color before and same color after: Add 2 * edge cost
    2. same color before, different color after: Add edge cost
    3. different color before, same color after: Subtract edge cost
    4. different color before and different color after: Subtract 2 * edge cost
    """

    edge_weight = graph[node][neighbor_node].get('weight')

    if node_color_bef == neighbor_color_bef and node_color_bef != neighbor_color_aft and node_color_aft != neighbor_color_bef and node_color_aft == neighbor_color_aft:
        return 2 * edge_weight
    elif node_color_bef == neighbor_color_bef and node_color_bef != neighbor_color_aft and node_color_aft != neighbor_color_bef and node_color_aft != neighbor_color_aft:
        return edge_weight
    elif node_color_bef != neighbor_color_bef and node_color_bef == neighbor_color_aft and node_color_aft == neighbor_color_bef and node_color_aft != neighbor_color_aft:
        return -2 * edge_weight
    elif node_color_bef != neighbor_color_bef and node_color_bef != neighbor_color_aft and node_color_aft == neighbor_color_bef and node_color_aft != neighbor_color_aft:
        return -edge_weight  
    elif node_color_bef != neighbor_color_bef and node_color_bef == neighbor_color_aft and node_color_aft != neighbor_color_bef and node_color_aft != neighbor_color_aft:
        return -edge_weight
    elif node_color_bef != neighbor_color_bef and node_color_bef != neighbor_color_aft and node_color_aft != neighbor_color_bef and node_color_aft == neighbor_color_aft:
        return edge_weight
    elif node_color_bef != neighbor_color_bef and node_color_bef != neighbor_color_aft and node_color_aft != neighbor_color_bef and node_color_aft != neighbor_color_aft:
        return 0
    
    # print(node_color_bef, node_color_aft, neighbor_color_bef, neighbor_color_aft)
    # print(node_color_bef == neighbor_color_bef)
    # print(node_color_bef != neighbor_color_aft)
    # print(node_color_aft != neighbor_color_bef)
    # print(node_color_aft != neighbor_color_aft)

if __name__ == '__main__':
    file_path = r"C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\graphs\SBM(5, 2, a).json"
    graph, graph_name, num_nodes, num_groups, group_mode, initial_node_colors, ground_truth_log_likelihood = load_graph_from_json(file_path)
    total_groups = num_groups
    w = compute_w(graph, total_groups)

    # print(w)

    print(calc_log_likelihood3(graph, w))
    # print(calc_log_likelihood2(graph, w))
    print(calc_log_likelihood(graph, w))
    