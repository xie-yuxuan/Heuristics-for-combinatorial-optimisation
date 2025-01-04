import networkx as nx
import numpy as np

"""
Utility functions including:
- calc_cost
- calc_delta_cost
- calc_delta_cost_edge (no longer used, logic included in optimise function)
"""

def calc_log_likelihood(graph, w):
    """
    Calculate the log-likelihood for the graph given the w matrix.
    
    Parameters:
        graph (nx.Graph): The graph with node colors assigned.
        w (ndarray): Precomputed w matrix based on group memberships.

    Returns:
        float: Log-likelihood of the current graph configuration.
    """
    adj_matrix = nx.to_numpy_array(graph)
    groups = np.array([graph.nodes[node]['color'] for node in graph.nodes])
    log_likelihood = 0
    epsilon = 1e-10

    for u in range(len(groups)):
        for v in range(len(groups)):
            if u != v:  # Skip self-loops
                gu, gv = groups[u], groups[v]
                p = max(epsilon, min(1 - epsilon, w[gu, gv]))  # Clamp probabilities to avoid log issues
                log_likelihood += (
                    adj_matrix[u, v] * np.log(p) + (1 - adj_matrix[u, v]) * np.log(1 - p)
                )
    return log_likelihood

def compute_w(graph):
    """
    Compute the symmetric w matrix for the graph based on group memberships.
    
    Parameters:
        graph (nx.Graph): The graph with initial group memberships in the 'color' attribute.

    Returns:
        ndarray: The symmetric matrix of edge probabilities.
    """
    adj_matrix = nx.to_numpy_array(graph) # adjacency matrix
    groups = np.array([graph.nodes[node]['color'] for node in graph.nodes]) # get list of node colors / group membership
    unique_groups = np.unique(groups)
    
    group_counts = {g: np.sum(groups == g) for g in unique_groups} # for 2 colors, 10 nodes we have {0:3, 1:7}
    # print(group_counts)
    w = np.zeros((len(unique_groups), len(unique_groups)))
    
    for i, g1 in enumerate(unique_groups):
        for j, g2 in enumerate(unique_groups):
            if i == j:  # Within-group
                m_gg = np.sum(adj_matrix[np.ix_(groups == g1, groups == g1)]) # np.ix_ to get a submatrix, m_gg is the total number of edges in group g, A is all pairwise connections in graph
                n_g = group_counts[g1]
                w[i, j] = m_gg / (0.5 * n_g * (n_g - 1)) if n_g > 1 else 0
            else:  # Between-group
                m_gg = np.sum(adj_matrix[np.ix_(groups == g1, groups == g2)])
                n_g1, n_g2 = group_counts[g1], group_counts[g2]
                w[i, j] = m_gg / (n_g1 * n_g2) if n_g1 > 0 and n_g2 > 0 else 0

    return w

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

    pass