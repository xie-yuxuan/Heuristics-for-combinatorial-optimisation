import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

from sbm_analysis_single_graph import sbm_plot_cost_data, sbm_plot_final_costs

def compute_sbm_summary_table(cost_data, num_nodes, graph_name):
    greedy_random_costs = []
    reluctant_random_costs = []
    greedy_random_iters = []
    reluctant_random_iters = []
    # greedy_sa_iters = []
    # reluctant_sa_iters = []

    for run in cost_data.values():
        try:
            if "cost_data_g" in run:
                gr_final = run["cost_data_g"][-1]
                if isinstance(gr_final, (list, np.ndarray)):
                    gr_final = gr_final[-1]
                greedy_random_costs.append(float(gr_final))
                greedy_random_iters.append(len(run["cost_data_g"]))

            if "cost_data_r" in run:
                rr_final = run["cost_data_r"][-1]
                if isinstance(rr_final, (list, np.ndarray)):
                    rr_final = rr_final[-1]
                reluctant_random_costs.append(float(rr_final))
                reluctant_random_iters.append(len(run["cost_data_r"]))

            # if "cost_data_gsa" in run:
            #     greedy_sa_iters.append(len(run["cost_data_gsa"]))
            # if "cost_data_rsa" in run:
            #     reluctant_sa_iters.append(len(run["cost_data_rsa"]))

        except Exception as e:
            print(f"Skipping malformed entry due to error: {e}")

    greedy_random_costs = np.array(greedy_random_costs)
    reluctant_random_costs = np.array(reluctant_random_costs)
    greedy_random_iters = np.array(greedy_random_iters)
    reluctant_random_iters = np.array(reluctant_random_iters)
    # greedy_sa_iters = np.array(greedy_sa_iters)
    # reluctant_sa_iters = np.array(reluctant_sa_iters)

    def mean_ci(arr):
        mean = np.mean(arr)
        ci = 1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0
        return mean, ci

    # print(f"--- SBM Summary for {graph_name} (n={num_nodes}) ---")

    if len(greedy_random_costs):
        gr_mean, gr_ci = mean_ci(greedy_random_costs)
        print(f"Mean Final Cost (Greedy Random): {gr_mean:.2f} ± {gr_ci:.2f}")
    if len(reluctant_random_costs):
        rr_mean, rr_ci = mean_ci(reluctant_random_costs)
        print(f"Mean Final Cost (Reluctant Random): {rr_mean:.2f} ± {rr_ci:.2f}")
    # if len(greedy_random_costs) and len(reluctant_random_costs):
    #     cost_diff = (greedy_random_costs - reluctant_random_costs) / num_nodes
    #     diff_mean, diff_ci = mean_ci(cost_diff)
    #     print(f"Normalised Cost Diff (GR - RR) / n: {diff_mean:.6f} ± {diff_ci:.6f}")

    # if len(greedy_random_iters):
    #     gr_iter_mean, gr_iter_ci = mean_ci(greedy_random_iters)
    #     print(f"Mean Iterations (Greedy Random): {gr_iter_mean:.2f} ± {gr_iter_ci:.2f}")
    # if len(reluctant_random_iters):
    #     rr_iter_mean, rr_iter_ci = mean_ci(reluctant_random_iters)
    #     print(f"Mean Iterations (Reluctant Random): {rr_iter_mean:.2f} ± {rr_iter_ci:.2f}")

    # if len(greedy_sa_iters):
    #     gsa_mean, gsa_ci = mean_ci(greedy_sa_iters)
    #     print(f"Mean Iterations (Greedy SA): {gsa_mean:.2f} ± {gsa_ci:.2f}")
    # if len(reluctant_sa_iters):
    #     rsa_mean, rsa_ci = mean_ci(reluctant_sa_iters)
    #     print(f"Mean Iterations (Reluctant SA): {rsa_mean:.2f} ± {rsa_ci:.2f}")


if __name__ == '__main__':

    

    file_path = r"C:\Projects\Heuristics for combinatorial optimisation\results\within_organisation_facebook_friendships_L2, 0.05, dynamic_w_results.json"
    
    with open(file_path, 'r') as f:
        data = json.load(f)

    graph_name = data['graph_name']
    # if random_prob is not None:
    #     graph_name = graph_name[:-1] + f", {random_prob})"
    num_nodes = data['num_nodes']
    num_groups = data['num_groups']
    # group_mode = data['group_mode']
    # ground_truth_w = data['ground_truth_w']
    # ground_truth_log_likelihood = data['ground_truth_log_likelihood']
    all_cost_data = data['cost_data']

    # plot log likelihood against iterations for all initial colorings / specific coloring
    # sbm_plot_cost_data(all_cost_data, graph_name, num_groups, num_nodes, group_mode=None, ground_truth_w=None, ground_truth_log_likelihood=None, specific_coloring=None)

    # plot scatter of final log likelihood against initial colorings 
    # sbm_plot_final_costs(all_cost_data, graph_name, num_nodes, num_groups, group_mode=None, ground_truth_w=None, ground_truth_log_likelihood=None)

    # compute summary statistics
    compute_sbm_summary_table(all_cost_data, num_nodes, graph_name)