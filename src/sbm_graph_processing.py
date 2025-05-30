import json
import os
import numpy as np
import networkx as nx
import time
import copy
import seaborn as sns
from networkx.readwrite import json_graph
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from sklearn.metrics import normalized_mutual_info_score

import matplotlib.pyplot as plt

from algorithms import optimise_sbm, optimise_sbm2, optimise_sbm3, SBMState
from visualisation import draw_graph, generate_heatmap_of_color_changes



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

# if __name__ == "__main__":

#     # for mode_number in range(10):  
#     #     mapped_value = np.linspace(-0.95, 0.95, 10)[mode_number]
        
#     #     # Loop through Y values (instance_number = 00 to 09, 10 to 19, etc.)
#     #     for instance_number in range(10):  
#     #         seed = instance_number + 1  # Matches the graph generation seed
#     #         print(seed)
#     #         np.random.seed(seed)  

#             # Load graph
#             # file_path = rf"C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\graphs\SBM(1000, 2, t{mode_number}{instance_number}).json"


#     mode_number = 7
#     instance_number = 0

#     random_prob = 0.0

#     seed = 1+instance_number
#     np.random.seed(seed)
    
#     file_path = rf"C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\graphs\SBM(10000, 2, t{mode_number}{instance_number}).json"

#     graph, graph_name, num_nodes, num_groups, group_mode, initial_node_colors, ground_truth_w, ground_truth_log_likelihood = load_graph_from_json(file_path)


#     g = []
#     for group in range(num_groups):
#         g.extend([group] * (num_nodes // num_groups))
#     g.extend([num_groups - 1] * (num_nodes % num_groups))
#     g = np.array(g)
#     g_copy = g.copy()

#     color_map_path = 'C:/Projects/Heuristics for combinatorial optimisation/Heuristics-for-combinatorial-optimisation/data/color_map.json'
#     with open(color_map_path, 'r') as f:
#         color_map = json.load(f)['color_map']
#     color_to_int = {name: int(key) for key, name in color_map.items()}

#     results = {
#         "graph_name": graph_name,
#         "num_nodes" : num_nodes,
#         "num_groups" : num_groups,
#         "group_mode" : group_mode,
#         "ground_truth_w" : ground_truth_w,
#         "ground_truth_log_likelihood" : ground_truth_log_likelihood,
#         "cost_data" : {}
#     }

#     w = np.array(json.loads(ground_truth_w), dtype=float)  # Using dtype=float to handle None as NaN
    
#     np.random.shuffle(g_copy)


#     # ISOLATE TEST FOR ONE SPECIFIC INITIAL COLORING ------------------------

#     # initial coloring 2nd eigenvector of A
#     # mean_degree = 0
#     # for n in range(num_nodes):
#     #     mean_degree += graph.degree(n)
#     # mean_degree /= num_nodes
#     # # print(mean_degree)


#     # D = np.diag([graph.degree(n) for n in range(num_nodes)])  # Degree matrix
#     # # print(D)
#     # A = nx.adjacency_matrix(graph, nodelist=range(num_nodes)).toarray()  # Adjacency matrix
#     # L = D - A  # Laplacian matrix

#     # # Compute the two smallest eigenvalues and eigenvectors
#     # eigvals, eigvecs = eigsh(A, k=2, which='LM')

#     # sorted_indices = np.argsort(eigvals)[::-1]  # Indices of eigenvalues sorted from largest to smallest
#     # second_largest_index = sorted_indices[1]  # Index of second-largest eigenvalue

#     # # Get the second largest eigenvector
#     # second_eigenvector = eigvecs[:, second_largest_index] 

#     # # # Assign colors based on the sign of the second-largest eigenvector
#     # for i, node in enumerate(range(num_nodes)):
#     #     graph.nodes[node]['color'] = 0 if second_eigenvector[i] < 0 else 1

#     # graph_copy = graph.copy()


#     # specific_coloring = initial_node_colors[0]
#     # for node, color in enumerate(specific_coloring):
#     #     graph.nodes[node]['color'] = color

#     # # # **Instantiate GreedyState**
#     # greedy_state = SBMState(graph, num_groups, w)
#     # graph_g, log_likelihood_data_g, final_w_g, g_optimised_g, final_LL = greedy_state.optimise(algo_func="greedy_random")


#     # print(final_LL)
#     # print(log_likelihood_data_g[-1][-1])

#     # # # print(normalized_mutual_info_score(g_copy, g_optimised_g))
#     # # # print(g)
#     # # # print(g_optimised_g)
#     # # # print(normalized_mutual_info_score(g, g_optimised_g))

#     # # reluctant_state = SBMState(graph_copy, num_groups, w)
#     # # graph_r, log_likelihood_data_r, final_w_r, g_optimised_r = reluctant_state.optimise(algo_func="reluctant")


#     # # # print(normalized_mutual_info_score(g, g_optimised_g))
    
#     # # results["cost_data"] = {
#     # #     "cost_data_g": log_likelihood_data_g,
#     # #     "cost_data_r": log_likelihood_data_r,
#     # #     "nmi_g" : normalized_mutual_info_score(g_copy, g_optimised_g),
#     # #     "nmi_r" : normalized_mutual_info_score(g_copy, g_optimised_r)
#     # #     }

#     # # graphs_path = r"C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\results"

#     # # with open(os.path.join(graphs_path, f"{graph_name}ce_results.json"), 'w') as f:
#     # #     json.dump(results, f, indent = 2)

#     # # print(f"Saved results to {graphs_path}/{graph_name}ce_results.json")  

#     # # graph_g, log_likelihood_data_g, final_w_g, changes_g = optimise_sbm4(graph, num_groups, group_mode, algo_func=fr)
#     # # graph_r, log_likelihood_data_r, final_w_r, changes_g = optimise_sbm4(graph, num_groups, group_mode, algo_func=fr)

#     # # Visualise graph after optimisation
#     # draw_graph(graph_g, pos=nx.spring_layout(graph, seed=1), graph_name=graph_name, iterations_taken=0, cost_data=log_likelihood_data_g,
#     #         color_set_size=num_groups, 
#     #         degree=None, 
#     #         num_nodes=num_nodes, 
#     #         gaussian_mean=None, 
#     #         gaussian_variance=None,
#     #         ground_truth_log_likelihood = ground_truth_log_likelihood
#     #         )

#     # Visualise heatmap of node changes against iterations
#     # generate_heatmap_of_color_changes(specific_coloring, changes_g, num_nodes, num_groups, color_map)

#     # Visualise histogram of number of colorings per node index
#     # recolor_counts = [0] * num_nodes
#     # for change in changes_g:
#     #     node_index, _ = change  # We don't need the color, just the node index
#     #     recolor_counts[node_index] += 1
#     # plt.bar(range(num_nodes), recolor_counts)
#     # plt.xlabel('Node Index')
#     # plt.ylabel('Number of Recolors')
#     # plt.title('Histogram of Node Recolors')
#     # plt.show()
#     # ------------------------------------------------------------------------S

#     # # make all colors red which is 0
#     # for node in graph_copy.nodes:
#     #     graph_copy.nodes[node]['color'] = 0

#     # replace all colors with initial colorings
#     start_time = time.time()
#     for i, initial_coloring in enumerate(initial_node_colors):
#         for node, color in enumerate(initial_coloring):
#             graph.nodes[node]['color'] = color

#         graph_copy = graph.copy()
#         graph_copy2 = graph.copy()
#         graph_copy3 = graph.copy()
#         graph_copy4 = graph.copy()
#         graph_copy5 = graph.copy()

#         # **Instantiate class**
#         greedy_state = SBMState(graph, num_groups, w)
#         graph_g, log_likelihood_data_g, final_w_g, g_optimised_g = greedy_state.optimise(algo_func="greedy", random_prob=random_prob, max_iterations=None)
#         reluctant_state = SBMState(graph_copy, num_groups, w)
#         graph_r, log_likelihood_data_r, final_w_r, g_optimised_r = reluctant_state.optimise(algo_func="reluctant", random_prob=random_prob, max_iterations=None)
#         greedy_random_state = SBMState(graph_copy2, num_groups, w)
#         graph_gr, log_likelihood_data_gr, final_w_gr, g_optimised_gr = greedy_random_state.optimise(algo_func="greedy_random", random_prob=random_prob, max_iterations=None)
#         reluctant_random_state = SBMState(graph_copy3, num_groups, w)
#         graph_rr, log_likelihood_data_rr, final_w_rr, g_optimised_rr = reluctant_random_state.optimise(algo_func="reluctant_random", random_prob=random_prob, max_iterations=None)
#         greedy_sa_state = SBMState(graph_copy4, num_groups, w)
#         graph_gsa, log_likelihood_data_gsa, final_w_gsa, g_optimised_gsa = greedy_sa_state.optimise(algo_func="greedy_sa", random_prob=random_prob, max_iterations=None)
#         reluctant_sa_state = SBMState(graph_copy5, num_groups, w)
#         graph_rsa, log_likelihood_data_rsa, final_w_rsa, g_optimised_rsa = reluctant_sa_state.optimise(algo_func="reluctant_sa", random_prob=random_prob, max_iterations=None)

#         results["cost_data"][f"initial_coloring_{i}"] = {
#             "cost_data_g": log_likelihood_data_g,
#             "cost_data_r": log_likelihood_data_r,
#             "cost_data_gr" : log_likelihood_data_gr,
#             "cost_data_rr" : log_likelihood_data_rr,
#             "cost_data_gsa" : log_likelihood_data_gsa,
#             "cost_data_rsa" : log_likelihood_data_rsa,
#             "nmi_g" : normalized_mutual_info_score(g_copy, g_optimised_g),
#             "nmi_r" : normalized_mutual_info_score(g_copy, g_optimised_r),
#             "nmi_gr" : normalized_mutual_info_score(g_copy, g_optimised_gr),
#             "nmi_rr" : normalized_mutual_info_score(g_copy, g_optimised_rr),
#             "nmi_gsa" : normalized_mutual_info_score(g_copy, g_optimised_gsa),
#             "nmi_rsa" : normalized_mutual_info_score(g_copy, g_optimised_r)
#         }

#         print(f"{i} initial coloring optimisation complete")

#     end_time = time.time()
#     print(f"Execution time: {end_time - start_time:.4f} seconds")

#     # save results

#     graphs_path = r"C:\Projects\Heuristics for combinatorial optimisation\results"

#     if random_prob is not None:
#         graph_name = graph_name[:-1] + f", {random_prob})"

#     with open(os.path.join(graphs_path, f"{graph_name}_results.json"), 'w') as f:
#         json.dump(results, f, indent = 2)

#     print(f"Saved results to {graphs_path}/{graph_name}_results.json")  

if __name__ == "__main__":

    # mode_number = 7
    instance_number = 0
    random_prob = 0

    seed = 1 + instance_number
    np.random.seed(seed)

    file_path = rf"C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\graphs\SBM(10000, 2, c).json"
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

    if random_prob is not None:
        graph_name = graph_name[:-1] + f", {random_prob})"
    output_path = os.path.join(graphs_path, f"{graph_name}_results.json")

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

    for i, initial_coloring in enumerate(initial_node_colors):
        key = f"initial_coloring_{i}"
        if key not in results["cost_data"]:
            results["cost_data"][key] = {}

        entry = results["cost_data"][key]

        for node, color in enumerate(initial_coloring):
            graph.nodes[node]['color'] = color

        # ==== Greedy ====
        if "cost_data_g" not in entry:
            graph_copy_g = graph.copy()
            greedy_state = SBMState(graph_copy_g, num_groups, w)
            graph_g, data_g, _, g_optimised = greedy_state.optimise("greedy", random_prob, None)
            entry["cost_data_g"] = data_g
            entry["nmi_g"] = normalized_mutual_info_score(g_copy, g_optimised)

        # ==== Reluctant ====
        if "cost_data_r" not in entry:
            graph_copy_r = graph.copy()
            reluctant_state = SBMState(graph_copy_r, num_groups, w)
            graph_r, data_r, _, g_optimised = reluctant_state.optimise("reluctant", random_prob, None)
            entry["cost_data_r"] = data_r
            entry["nmi_r"] = normalized_mutual_info_score(g_copy, g_optimised)

        # # ==== Greedy Random ====
        # if "cost_data_gr" not in entry:
        #     graph_copy_gr = graph.copy()
        #     greedy_random_state = SBMState(graph_copy_gr, num_groups, w)
        #     graph_gr, data_gr, _, g_optimised = greedy_random_state.optimise("greedy_random", random_prob, None)
        #     entry["cost_data_gr"] = data_gr
        #     entry["nmi_gr"] = normalized_mutual_info_score(g_copy, g_optimised)

        # # ==== Reluctant Random ====
        # if "cost_data_rr" not in entry:
        #     graph_copy_rr = graph.copy()
        #     reluctant_random_state = SBMState(graph_copy_rr, num_groups, w)
        #     graph_rr, data_rr, _, g_optimised = reluctant_random_state.optimise("reluctant_random", random_prob, None)
        #     entry["cost_data_rr"] = data_rr
        #     entry["nmi_rr"] = normalized_mutual_info_score(g_copy, g_optimised)

        # # ==== Greedy SA ====
        # if "cost_data_gsa" not in entry:
        #     graph_copy_gsa = graph.copy()
        #     greedy_sa_state = SBMState(graph_copy_gsa, num_groups, w)
        #     graph_gsa, data_gsa, _, g_optimised = greedy_sa_state.optimise("greedy_sa", random_prob, None)
        #     entry["cost_data_gsa"] = data_gsa
        #     entry["nmi_gsa"] = normalized_mutual_info_score(g_copy, g_optimised)

        # # ==== Reluctant SA ====
        # if "cost_data_rsa" not in entry:
        #     graph_copy_rsa = graph.copy()
        #     reluctant_sa_state = SBMState(graph_copy_rsa, num_groups, w)
        #     graph_rsa, data_rsa, _, g_optimised = reluctant_sa_state.optimise("reluctant_sa", random_prob, None)
        #     entry["cost_data_rsa"] = data_rsa
        #     entry["nmi_rsa"] = normalized_mutual_info_score(g_copy, g_optimised)

        results["cost_data"][key] = entry
        print(f"{i} initial coloring optimisation complete")

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {output_path}")
