import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
from networkx.readwrite import json_graph
from collections import defaultdict
import numpy as np
import json
import time
from sklearn.metrics import normalized_mutual_info_score

from visualisation import draw_graph
from algorithms import optimise_sbm, optimise_sbm2, optimise_sbm3, SBMState


if __name__ == '__main__':

    np.random.seed(seed=1)

    real_world_data_path = r'C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\real_world_data\spanish_high_school'
    num_groups = 2 # Set number of groups
    graph_name = "SBM_spanish" 

    # construct network -------------------------------------------------------
    edges_df = pd.read_csv(real_world_data_path + '/edges.csv')
    nodes_df = pd.read_csv(real_world_data_path + '/nodes.csv')

    nodes_df.columns = nodes_df.columns.str.strip()
    edges_df.columns = edges_df.columns.str.strip()

    # Initialize graph
    graph = nx.Graph()

    # class_to_color = {'Applied': 0, 'Formal': 1, 'Natural': 2, 'Social': 3}  # Adjust according to your actual class names
    # class_to_color = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}  # Adjust according to your actual class names
    class_to_color = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}  # Adjust according to your actual class names
    # class_to_color = {'Female': 0, 'Male': 1}  # Adjust according to your actual class names

    # Add nodes with attributes
    for _, row in nodes_df.iterrows():
        node_id = row['# index']
        # position_str = row['_pos']
        class_name = row['Curso']
        
        # position_list = eval(position_str[6:-1])  # Remove 'array(' and ')'
        # position = np.array(position_list)  # Convert list to a NumPy array

        # G.add_node(node_id, color=class_to_color.get(class_name, 0))
        graph.add_node(node_id)

    # Add edges with attributes
    for _, row in edges_df.iterrows():
        # if row['weight'] > 0:
        source = row['# source']
        target = row['target']
        weight = 1  # You can use other attributes like fiber_length_mean, etc.
        graph.add_edge(source, target, weight=weight)

    # pos = nx.get_node_attributes(G, 'pos')

    # Plot the graph
    # draw_graph(graph=G, pos=nx.spring_layout(G, seed=1), graph_name="spanish_high_school", iterations_taken=None, cost_data=None, 
    #            color_set_size=None, 
    #            degree=None, 
    #            num_nodes=num_nodes,
    #            gaussian_mean=None, 
    #            gaussian_variance=None, 
    #            ground_truth_log_likelihood=None)

    # determine graph parameters ------------------------------------------
    num_nodes = len(graph.nodes)

    initial_node_colors = [
            [np.random.randint(0, num_groups) for _ in range(num_nodes)]
            for _ in range(100)
        ]
    
    g = []
    for group in range(num_groups):
        g.extend([group] * (num_nodes // num_groups))
    g.extend([num_groups - 1] * (num_nodes % num_groups))
    g = np.array(g)

    g_copy = g.copy()
    np.random.shuffle(g_copy)

    n, m = np.zeros(num_groups), np.zeros((num_groups, num_groups))
    for node in graph.nodes():
        n[g_copy[node]] += 1
    for u, v in graph.edges():
        m[g_copy[v], g_copy[u]] = m[g_copy[u], g_copy[v]] = m[g_copy[u], g_copy[v]] + 1
    
    # w_json = json.dumps(w.tolist())

    # print(g)
    # print(g_copy)
    # print(n)
    # print(m)
    # print(num_nodes)


    # optimisation ------------------------------------------------
    
    w = np.zeros((num_groups, num_groups))
    # set an estimate of w
    w += 1
    np.fill_diagonal(w, 9)
    w /= num_nodes

    random_prob = 0.05 # Set random prob

    color_map_path = 'C:/Projects/Heuristics for combinatorial optimisation/Heuristics-for-combinatorial-optimisation/data/color_map.json'
    with open(color_map_path, 'r') as f:
        color_map = json.load(f)['color_map']
    color_to_int = {name: int(key) for key, name in color_map.items()}

    results = {
        "graph_name": graph_name,
        "num_nodes" : num_nodes,
        "num_groups" : num_groups,
        # "ground_truth_w" : ground_truth_w,
        # "ground_truth_log_likelihood" : ground_truth_log_likelihood,
        "cost_data" : {}
    }

    # replace all colors with initial colorings
    start_time = time.time()
    for i, initial_coloring in enumerate(initial_node_colors):
        for node, color in enumerate(initial_coloring):
            graph.nodes[node]['color'] = color

        graph_copy = graph.copy()
        graph_copy2 = graph.copy()
        graph_copy3 = graph.copy()

        # **Instantiate class**
        greedy_state = SBMState(graph, num_groups, w)
        graph_g, log_likelihood_data_g, final_w_g, g_optimised_g = greedy_state.optimise(algo_func="greedy", random_prob=random_prob, max_iterations=None)
        reluctant_state = SBMState(graph_copy, num_groups, w)
        graph_r, log_likelihood_data_r, final_w_r, g_optimised_r = reluctant_state.optimise(algo_func="reluctant", random_prob=random_prob, max_iterations=None)
        greedy_random_state = SBMState(graph_copy2, num_groups, w)
        graph_gr, log_likelihood_data_gr, final_w_gr, g_optimised_gr = greedy_random_state.optimise(algo_func="greedy_random", random_prob=random_prob, max_iterations=None)
        reluctant_random_state = SBMState(graph_copy3, num_groups, w)
        graph_rr, log_likelihood_data_rr, final_w_rr, g_optimised_rr = reluctant_random_state.optimise(algo_func="reluctant_random", random_prob=random_prob, max_iterations=None)

        results["cost_data"][f"initial_coloring_{i}"] = {
            "cost_data_g": log_likelihood_data_g,
            "cost_data_r": log_likelihood_data_r,
            "cost_data_gr" : log_likelihood_data_gr,
            "cost_data_rr" : log_likelihood_data_rr
            # "nmi_g" : normalized_mutual_info_score(g_copy, g_optimised_g),
            # "nmi_r" : normalized_mutual_info_score(g_copy, g_optimised_r),
            # "nmi_gr" : normalized_mutual_info_score(g_copy, g_optimised_gr),
            # "nmi_rr" : normalized_mutual_info_score(g_copy, g_optimised_rr)
        }

        print(f"{i} initial coloring optimisation complete")

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    # save results --------------------------------------------------
    graphs_path = r"C:\Projects\Heuristics for combinatorial optimisation\results"

    # if random_prob is not None:
    #     graph_name = graph_name[:-1] + f", {random_prob})"

    with open(os.path.join(graphs_path, f"{graph_name}_results.json"), 'w') as f:
        json.dump(results, f, indent = 2)

    print(f"Saved results to {graphs_path}/{graph_name}_results.json")  
    
    
    
