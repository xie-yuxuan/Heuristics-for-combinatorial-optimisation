import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from networkx.readwrite import json_graph

from visualisation import draw_graph

"""
Run this script to generate and save random regular graphs based on specified parameters (n, k, mu, sigma). 
Specified number of initial colorings are generated and saved in a list.
Uncomment draw_graph() to view graphs before saving. 
"""

def generate_random_regular_graph(degree, num_nodes,  gaussian_mean, gaussian_variance, seed):
    # specifying seed for reproducibility of random results
    graph = nx.random_regular_graph(degree, num_nodes, seed)

    for u,v in graph.edges(): # u, v are the nodes connected by each edge
        if gaussian_mean == None and gaussian_variance == None:
            edge_weight = 1
            graph[u][v]['weight'] = edge_weight
        else:
            edge_weight = np.random.normal(gaussian_mean, gaussian_variance)
            graph[u][v]['weight'] = edge_weight

    # for node in graph.nodes():
    #     graph.nodes[node]['color'] = np.random.randint(0, color_set_size)

    return graph

def generate_random_graph(num_nodes, max_degree, gaussian_mean, gaussian_variance, seed):
    """
    generate a random graph with specified number of nodes, degree is random between 1 and max_degree, connection is random
    graph creation follows the configuration process where each nodes have stubs (tentacles)
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed) #  initialise generator object
    
    # Initialize an empty graph
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    
    # Generate random degrees for each node
    degrees = rng.integers(1, max_degree + 1, size=num_nodes)
    
    # Create a list of stubs (nodes repeated according to their degree)
    stubs = []
    for node, degree in enumerate(degrees):
        stubs.extend([node] * degree) # [2, 3, 1] -> [0, 0, 1, 1, 1, 2]
    
    # Shuffle the stubs and create random edges
    rng.shuffle(stubs)
    while len(stubs) > 1:
        u = stubs.pop()
        v = stubs.pop()
        # Avoid self-loops and duplicate edges
        while u == v or graph.has_edge(u, v):
            stubs.append(v)
            rng.shuffle(stubs)
            v = stubs.pop()
        graph.add_edge(u, v)
    
    # Ensure the graph is connected
    while not nx.is_connected(graph):
        components = list(nx.connected_components(graph))
        for i in range(len(components) - 1):
            # Connect a node from one component to another
            u = rng.choice(list(components[i]))
            v = rng.choice(list(components[i + 1]))
            graph.add_edge(u, v)
    
    # Assign edge weights
    for u, v in graph.edges():
        if gaussian_mean is None and gaussian_variance is None:
            edge_weight = 1
        else:
            edge_weight = np.random.normal(gaussian_mean, gaussian_variance)
        graph[u][v]['weight'] = edge_weight

    return graph

if __name__ == '__main__':
    # set parameters
    num_nodes = 20
    degree = 10
    color_set_size = 4
    gaussian_mean = None
    gaussian_variance = None
    random_regular = False
    num_initial_colorings = 100
    if gaussian_mean == None and gaussian_variance == None and random_regular:
        graph_name = f"{num_nodes, degree, color_set_size, 'uniform'}"
    elif random_regular:
        graph_name = f"{num_nodes, degree, color_set_size}"
    elif gaussian_mean == None and gaussian_variance == None and not random_regular:
        graph_name = f"{num_nodes, degree, color_set_size, 'uniform', 'not regular'}"
    else:
        graph_name = f"{num_nodes, degree, color_set_size, 'not regular'}"

    # generate graph, get J
    if random_regular:
        graph = generate_random_regular_graph(degree, num_nodes, gaussian_mean, gaussian_variance, seed=1)
    else:
        graph = generate_random_graph(num_nodes, degree, gaussian_mean, gaussian_variance, seed=1)


    # create a list of initial color states (list of lists)
    initial_node_colors = [
        [np.random.randint(0, color_set_size) for _ in range(num_nodes)]
        for _ in range(num_initial_colorings)
    ]

    # print(initial_node_colors)

    # uncomment to view graphs before saving
    # draw_graph(graph, pos=nx.spring_layout(graph, seed=1), graph_name=graph_name, iterations_taken=0, cost_data=None,
    #            color_set_size=color_set_size, 
    #            degree=degree, 
    #            num_nodes=num_nodes, 
    #            gaussian_mean=gaussian_mean, 
    #            gaussian_variance=gaussian_variance,
    #            ground_truth_log_likelihood = None
    #            )

    graphs_path = "C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\graphs"

    graph_data = json_graph.node_link_data(graph) # node_link_data converts graph into dictionary to be serialised to JSON

    data = {
        "graph_name": graph_name,
        "degree" : degree,
        "num_nodes" : num_nodes,
        "color_set_size" : color_set_size,
        "gaussian_mean" : gaussian_mean,
        "gaussian_variance" : gaussian_variance,
        "initial_node_colors" : initial_node_colors,
        "graph_data": graph_data
    }

    with open(os.path.join(graphs_path, f"{graph_name}.json"), 'w') as f:
        json.dump(data, f, indent = 2)

    print(f"Saved graph to {graphs_path}/{graph_name}.json")
