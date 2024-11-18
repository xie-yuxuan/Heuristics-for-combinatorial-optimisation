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
        edge_weight = np.random.normal(gaussian_mean, gaussian_variance)
        graph[u][v]['weight'] = edge_weight

    # for node in graph.nodes():
    #     graph.nodes[node]['color'] = np.random.randint(0, color_set_size)

    return graph

if __name__ == '__main__':
    # set parameters
    num_nodes = 1000
    degree = 20
    color_set_size = 6
    gaussian_mean = 0
    gaussian_variance = 1
    num_initial_colorings = 100
    graph_name = f"{num_nodes, degree, color_set_size}"

    # generate graph, get J
    graph = generate_random_regular_graph(degree, num_nodes, gaussian_mean, gaussian_variance, seed=1)

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
    #            gaussian_variance=gaussian_variance
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
