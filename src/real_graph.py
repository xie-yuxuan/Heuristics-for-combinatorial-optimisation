import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import json

from visualisation import draw_graph

if __name__ == '__main__':

    real_world_data_path = r'C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\real_world_data\spanish_high_school'

    edges_df = pd.read_csv(real_world_data_path + '/edges.csv')
    nodes_df = pd.read_csv(real_world_data_path + '/nodes.csv')

    nodes_df.columns = nodes_df.columns.str.strip()
    edges_df.columns = edges_df.columns.str.strip()

    # Initialize graph
    G = nx.Graph()

    # class_to_color = {'Applied': 0, 'Formal': 1, 'Natural': 2, 'Social': 3}  # Adjust according to your actual class names
    # class_to_color = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}  # Adjust according to your actual class names
    class_to_color = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}  # Adjust according to your actual class names
    # class_to_color = {'Female': 0, 'Male': 1}  # Adjust according to your actual class names

    # Add nodes with attributes
    for _, row in nodes_df.iterrows():
        node_id = row['# index']
        # position_str = row['_pos']
        class_name = row['Curso']
        
        # Remove 'array(' and ')' and convert the rest to a list
        # position_list = eval(position_str[6:-1])  # Remove 'array(' and ')'
        # position = np.array(position_list)  # Convert list to a NumPy array

        G.add_node(node_id, color=class_to_color.get(class_name, 0))
        # G.add_node(node_id, pos=position)

    # Add edges with attributes
    for _, row in edges_df.iterrows():
        # if row['weight'] > 0:
        source = row['# source']
        target = row['target']
        weight = 1  # You can use other attributes like fiber_length_mean, etc.
        G.add_edge(source, target, weight=weight)

    # pos = nx.get_node_attributes(G, 'pos')

    num_nodes = len(G.nodes)

    # print(num_nodes)

    # Plot the graph
    draw_graph(graph=G, pos=nx.spring_layout(G, seed=1), graph_name="spanish_high_school", iterations_taken=None, cost_data=None, 
               color_set_size=None, 
               degree=None, 
               num_nodes=num_nodes,
               gaussian_mean=None, 
               gaussian_variance=None, 
               ground_truth_log_likelihood=None)