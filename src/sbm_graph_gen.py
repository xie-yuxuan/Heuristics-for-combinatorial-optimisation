import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from networkx.readwrite import json_graph
from collections import defaultdict

from visualisation import draw_graph
from utils import calc_log_likelihood, compute_w

def gen_sbm_graph(g, w):
    num_nodes = len(g)
    num_groups = w.shape[0]

    group_membership = {i: int(g[i]) for i in range(num_nodes)}
    
    # Generate the adjacency matrix
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            prob = w[group_membership[i], group_membership[j]]
            if np.random.rand() < prob:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    # Create the NetworkX graph
    graph = nx.from_numpy_array(adjacency_matrix)

    # Assign colors to nodes based on group membership
    for node in graph.nodes():
        group = group_membership[node]
        graph.nodes[node]['color'] = group

    return graph, adjacency_matrix

def analyze_graph(graph, g):
    num_edges = graph.number_of_edges()
    avg_degree = num_edges / len(graph.nodes())
    
    # unique_groups = set(g.values())  # Get unique group labels
    group_edges = defaultdict(int)
    intergroup_edges = defaultdict(int)
    
    for u, v in graph.edges():
        if g[u] == g[v]:
            group_edges[g[u]] += 1  # Count intra-group edges
        else:
            intergroup_edges[(g[u], g[v])] += 1  # Count inter-group edges
    
    print(f"Number of edges: {num_edges}")
    print(f"Average degree: {avg_degree:.2f}")
    
    for group in range(num_groups):
        print(f"Edges within group {group}: {group_edges[group]}")
    
    print("Inter-group edges:")
    for (group1, group2), count in intergroup_edges.items():
        print(f"Edges between group {group1} and group {group2}: {count}")

    isolated_nodes = [node for node in graph.nodes() if len(list(graph.neighbors(node))) == 0]
    if isolated_nodes:
        print(f"Isolated nodes: {isolated_nodes}")
    else:
        print("No isolated nodes.")

if __name__ == '__main__':

    # set parameters
    num_nodes = 5000
    num_groups = 2
    num_initial_colorings = 100

    g = []
    for group in range(num_groups):
        g.extend([group] * (num_nodes // num_groups))
    g.extend([num_groups - 1] * (num_nodes % num_groups))
    g = np.array(g)

    # group_mode = "association"
    # group_mode = "bipartite"
    # group_mode = "core-periphery"
    # group_mode = "design"
    group_mode = "t"
    for mode_number in range(1,10):
    # mode_number = 0     
        instance_number = 0

        mapped_value = np.linspace(-0.95, 0.95, 10)[mode_number]
        seed = instance_number+1

        # for mode_number in range(10):  # X values (t0 to t9)
        #     mapped_value = np.linspace(-0.95, 0.95, 10)[mode_number]
            
        #     for instance_number in range(0):  # Y values (00 to 09, 10 to 19, etc.)
        #         seed = instance_number + 1  # Ensures repeatability
        #         print(seed)

        # set random seed
        np.random.seed(seed)

        # Generate g (same for all tX with the same instance_number)
        g_copy = g.copy()
        np.random.shuffle(g_copy)

        if group_mode[0] == "t":
            graph_name = f"SBM({num_nodes}, {num_groups}, t{mode_number}{instance_number})"
        else:
            graph_name = f"SBM({num_nodes}, {num_groups}, {group_mode[0]})"

        # Generate the w matrix (edge probabilities)
        w = np.zeros((num_groups, num_groups))

        if group_mode == "association":
            w += 1  # Small baseline for non-diagonal elements
            np.fill_diagonal(w, 9)  # Large diagonal elements
        elif group_mode == "bipartite":
            w += 9  # Large baseline for non-diagonal elements
            np.fill_diagonal(w, 1)  # Small diagonal elements
        elif group_mode == "core-periphery":
            w += 9  # Large baseline
            w[0, :] = 1  # Small first row (loners have low connections to all groups)
            w[:, 0] = 1  # Small first column (low connections to loners)
            w[0, 0] = 1  # loners have low self-connections
        elif group_mode == "design": # custom design
            w += 1
            np.fill_diagonal(w, 30)
            w[0, :] = 1
            w[:, 0] = 1
            w[0, 0] = 1
        elif group_mode[0] == "t":
            w += 12*(1-mapped_value)
            np.fill_diagonal(w, 12*(1+mapped_value))

        # normalise w such that average degree remains the same 
        w /= num_nodes

        graph, adjacency_matrix = gen_sbm_graph(g_copy, w)

        analyze_graph(graph, g_copy)

        # draw_graph(graph, pos=nx.spring_layout(graph, seed=1), graph_name=graph_name, iterations_taken=0, cost_data=None,
        #         color_set_size=num_groups, 
        #         degree=None, 
        #         num_nodes=num_nodes, 
        #         gaussian_mean=None, 
        #         gaussian_variance=None,
        #         ground_truth_log_likelihood = 0
        #         )

        # save graph data
        graphs_path = "C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\graphs"

        graph_data = json_graph.node_link_data(graph)
        
        initial_node_colors = [
            [np.random.randint(0, num_groups) for _ in range(num_nodes)]
            for _ in range(num_initial_colorings)
        ]

        n, m = np.zeros(num_groups), np.zeros((num_groups, num_groups))
        for node in graph.nodes():
            n[g_copy[node]] += 1
        for u, v in graph.edges():
            m[g_copy[v], g_copy[u]] = m[g_copy[u], g_copy[v]] = m[g_copy[u], g_copy[v]] + 1

        w_json = json.dumps(w.tolist())

        data = {
            "graph_name": graph_name,
            "num_nodes": num_nodes,
            "num_groups": num_groups,
            "group_mode": f"t{mode_number}{instance_number}",
            "graph_data": graph_data,
            "ground_truth_w": w_json,
            "ground_truth_log_likelihood": calc_log_likelihood(n, m, w),
            "initial_node_colors": initial_node_colors
        }

        with open(os.path.join(graphs_path, f"{graph_name}.json"), 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved graph to {graphs_path}/{graph_name}.json")

