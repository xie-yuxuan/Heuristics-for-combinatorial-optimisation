import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import pydot_layout

def load_json_graph(json_file):
    '''
    construct graph from adj matrix in the json file. Update vertex colors
    '''
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    adj_matrix = np.array(data['adjacency_matrix'])
    graph = nx.from_numpy_array(adj_matrix)

    vertex_colors = data.get('vertex_colors')
    nx.set_node_attributes(graph, vertex_colors, 'color')

    return graph

def draw_graph(graph):
    '''
    Visualise graph and save image file
    '''
    # TODO use pyvis or another library to visualise rather than networkx
    
    with open('C:/Users/Yuxuan Xie/Desktop/Heuristics for combinatorial optimisation/Heuristics-for-combinatorial-optimisation/data/color_map.json', 'r') as f:
        data = json.load(f)
    color_map = data['color_map']
    
    pos = pydot_layout(graph, prog='dot')
    node_colors = [color_map.get(str(graph.nodes[node].get('color', 0)), 'gray') for node in graph.nodes]
    
    plt.figure(figsize=(10, 10))

    nx.draw(graph, pos, with_labels=True, node_color=node_colors, node_size=500, edge_color='black', font_color='white', font_size=10)

    plt.savefig(str(graph))
    plt.show()

graph_1 = load_json_graph('C:/Users/Yuxuan Xie/Desktop/Heuristics for combinatorial optimisation/Heuristics-for-combinatorial-optimisation/data/sample_graphs/graph1.json')
draw_graph(graph_1)