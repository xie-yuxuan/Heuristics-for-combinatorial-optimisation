import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from networkx.readwrite import json_graph

from analysis import draw_graph
from graph import load_color_map

"""
Run this script to generate and save random graphs based on specified parameters. Uncomment draw_graph() to view graphs before saving. 
"""

def generate_random_regular_graph(degree, num_nodes, color_set_size, gaussian_mean, gaussian_variance, seed=None):
    # specifying seed for reproducibility of random results
    graph = nx.random_regular_graph(degree, num_nodes, seed=seed)

    for u,v in graph.edges(): # u, v are the nodes connected by each edge
        edge_weight = np.random.normal(gaussian_mean, gaussian_variance)
        graph[u][v]['weight'] = edge_weight

    for node in graph.nodes():
        graph.nodes[node]['color'] = np.random.randint(0, color_set_size)

    return graph

if __name__ == '__main__':
    # set parameters
    degree = 3
    num_nodes = 10
    color_set_size = 4
    gaussian_mean = 0
    gaussian_variance = 1
    seed = 1
    graph_name = "test2"

    graph = generate_random_regular_graph(degree, num_nodes, color_set_size, gaussian_mean, gaussian_variance, seed)

    # Uncomment line below to view graphs before saving
    # draw_graph(graph, pos=nx.spring_layout(graph, seed=1), graph_name="test", iterations_taken=0, cost_data=None)

    graphs_path = "C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\graphs"

    attributes = {
        "degree" : degree,
        "num_nodes" : num_nodes,
        "color_set_size" : color_set_size,
        "gaussian_mean" : gaussian_mean,
        "gaussian_variance" : gaussian_variance,
    }

    graph_data = json_graph.node_link_data(graph) # node_link_data converts graph into dictionary to be serialieed to JSON
    graph_data['attributes'] = attributes
    graph_data['name'] = graph_name
    # print(graph_data)

    with open(os.path.join(graphs_path, f"{graph_name}.json"), 'w') as f:
        json.dump(graph_data, f, indent = 2)

    print(f"Saved graph to {graphs_path}/{graph_name}.json")

