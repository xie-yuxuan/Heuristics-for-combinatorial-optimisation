import json
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph

from visualisation import draw_graph

def load_graph_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # load graph_name and attibutes
    graph_name = data["graph_name"]
    color_set_size = data["color_set_size"]
    degree = data["degree"]
    num_nodes = data["num_nodes"]
    gaussian_mean = data["gaussian_mean"]
    gaussian_variance = data["gaussian_variance"]
    
    graph_data = data["graph_data"]
    graph = json_graph.node_link_graph(graph_data)

    # uncomment to get adj matrix
    # adj_matrix = nx.adjacency_matrix(graph, weight='weight').toarray()
    
    return graph, graph_name, color_set_size, degree, num_nodes, gaussian_mean, gaussian_variance

if __name__ == '__main__':
    file_path = r"C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\graphs\test3.json"
    graph, graph_name, color_set_size, degree, num_nodes, gaussian_mean, gaussian_variance = load_graph_from_json(file_path)
    
    # uncomment to visualise graph plot
    # draw_graph(graph, pos=nx.spring_layout(graph, seed=1), graph_name=graph_name, iterations_taken=0, cost_data=None)

