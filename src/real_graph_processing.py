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
from utils import calc_log_likelihood, compute_w

from visualisation import draw_graph
from algorithms import optimise_sbm, optimise_sbm2, optimise_sbm3, SBMState

def load_graph_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # load graph_name and attibutes
    graph_name = data["graph_name"]
    num_groups = data["num_groups"]
    num_nodes = data["num_nodes"]
    group_mode = data["group_mode"]
    initial_node_colors = data["initial_node_colors"]
    ground_truth_w = data["ground_truth_w"]
    ground_truth_log_likelihood = data["ground_truth_log_likelihood"]

    graph_data = data["graph_data"]
    graph = json_graph.node_link_graph(graph_data)

    # uncomment to get adj matrix
    # adj_matrix = nx.adjacency_matrix(graph, weight='weight').toarray()
    
    return graph, graph_name, num_nodes, num_groups, group_mode, initial_node_colors, ground_truth_w, ground_truth_log_likelihood

if __name__ == '__main__':

    np.random.seed(seed=1)
    random_prob = 0.05
    use_dynamic_w = False

    file_path = rf"C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\graphs\within_organisation_facebook_friendships_L2, a.json"
    graph, graph_name, num_nodes, num_groups, group_mode, initial_node_colors, ground_truth_w, ground_truth_log_likelihood = load_graph_from_json(file_path)

    g = []
    for group in range(num_groups):
        g.extend([group] * (num_nodes // num_groups))
    g.extend([num_groups - 1] * (num_nodes % num_groups))
    g = np.array(g)
    g_copy = g.copy()
    np.random.shuffle(g_copy)

    w = np.array(json.loads(ground_truth_w), dtype=float)

    # Prepare output path
    graphs_path = r"C:\Projects\Heuristics for combinatorial optimisation\results"
    os.makedirs(graphs_path, exist_ok=True)

    suffix = "dynamic_w" if use_dynamic_w else "static_w"
    graph_name_full = graph_name + f", {random_prob}, {suffix}"
    output_path = os.path.join(graphs_path, f"{graph_name_full}_results.json")

    # Load existing results if any
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            results = json.load(f)
    else:
        results = {
            "graph_name": graph_name,
            "num_nodes": num_nodes,
            "num_groups": num_groups,
            "group_mode": group_mode,
            "ground_truth_w": ground_truth_w,
            "ground_truth_log_likelihood": ground_truth_log_likelihood,
            "cost_data": {}
        }

    start_time = time.time()

    print(f"Processing graph: {graph_name_full}")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of groups: {num_groups}")
    print(f"Group mode: {group_mode}")

    for i, initial_coloring in enumerate(initial_node_colors):

        key = f"initial_coloring_{i}"
        if key not in results["cost_data"]:
            results["cost_data"][key] = {}

        entry = results["cost_data"][key]

        for node, color in enumerate(initial_coloring):
            graph.nodes[node]['color'] = color

        def optimise_and_update(algorithm_name):
            graph_copy = graph.copy()
            current_w = w.copy()
            data_all = []
            g_final = None
            while True:
                state = SBMState(graph_copy, num_groups, current_w)
                _, data_step, _, g_final = state.optimise(algorithm_name, random_prob, None)
                data_all.extend(data_step[1:] if data_all else data_step)
                if not use_dynamic_w:
                    break
                # compute new w
                n = np.zeros(num_groups)
                m = np.zeros((num_groups, num_groups))
                for node in graph_copy.nodes():
                    n[graph_copy.nodes[node]['color']] += 1
                for u, v in graph_copy.edges():
                    u = int(u)
                    v = int(v)
                    m[graph_copy.nodes[u]['color'], graph_copy.nodes[v]['color']] += 1
                    m[graph_copy.nodes[v]['color'], graph_copy.nodes[u]['color']] += 1
                new_w = compute_w(n, m)
                if np.allclose(new_w, current_w, atol=1e-6):
                    break
                current_w = new_w
            return data_all, g_final

        if "cost_data_g" not in entry:
            data_g, g_optimised = optimise_and_update("greedy")
            entry["cost_data_g"] = data_g
            entry["nmi_g"] = normalized_mutual_info_score(g_copy, g_optimised)

        if "cost_data_r" not in entry:
            data_r, g_optimised = optimise_and_update("reluctant")
            entry["cost_data_r"] = data_r
            entry["nmi_r"] = normalized_mutual_info_score(g_copy, g_optimised)

        if "cost_data_gr" not in entry:
            data_gr, g_optimised = optimise_and_update("greedy_random")
            entry["cost_data_gr"] = data_gr
            entry["nmi_gr"] = normalized_mutual_info_score(g_copy, g_optimised)

        if "cost_data_rr" not in entry:
            data_rr, g_optimised = optimise_and_update("reluctant_random")
            entry["cost_data_rr"] = data_rr
            entry["nmi_rr"] = normalized_mutual_info_score(g_copy, g_optimised)

        # if "cost_data_gsa" not in entry:
        #     data_gsa, g_optimised = optimise_and_update("greedy_sa")
        #     entry["cost_data_gsa"] = data_gsa
        #     entry["nmi_gsa"] = normalized_mutual_info_score(g_copy, g_optimised)

        # if "cost_data_rsa" not in entry:
        #     data_rsa, g_optimised = optimise_and_update("reluctant_sa")
        #     entry["cost_data_rsa"] = data_rsa
        #     entry["nmi_rsa"] = normalized_mutual_info_score(g_copy, g_optimised)

        results["cost_data"][key] = entry
        print(f"{i} initial coloring optimisation complete")

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {output_path}")

