import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt

def extract_final_ll_vs_random_prob_random_greedy_and_reluctant(results_folder, num_nodes, num_groups, mode_number):
    pattern = re.compile(rf"SBM\({num_nodes}, {num_groups}, t{mode_number}0, ([0-9.]+)\)_results\.json")
    
    random_probs = []
    ll_values = {"gr": [], "rr": []}
    
    for file in os.listdir(results_folder):
        match = pattern.match(file)
        if match:
            random_prob = match.group(1)
            random_probs.append(float(random_prob))
            
            with open(os.path.join(results_folder, file), 'r') as f:
                data = json.load(f)
            
            final_lls = {"gr": [], "rr": []}
            
            for key in data["cost_data"]:
                if "cost_data_gr" in data["cost_data"][key]:
                    final_lls["gr"].append(data["cost_data"][key]["cost_data_gr"][-1])
                if "cost_data_rr" in data["cost_data"][key]:
                    final_lls["rr"].append(data["cost_data"][key]["cost_data_rr"][-1])
            
            for method in ll_values:
                if final_lls[method]:
                    ll_values[method].append(np.mean(final_lls[method]))
                else:
                    ll_values[method].append(np.nan)  # Handle missing data with NaN
    
    sorted_indices = np.argsort(random_probs)
    random_probs = np.array(random_probs)[sorted_indices]
    
    for method in ll_values:
        ll_values[method] = np.array(ll_values[method])[sorted_indices]
    
    plt.figure(figsize=(10, 6))
    colors = {"gr": "orange", "rr": "purple"}
    for method, label in zip(["gr", "rr"], ["Greedy Random", "Reluctant Random"]):
        plt.plot(random_probs, ll_values[method], marker='o', label=label, color=colors[method])

    plt.xlim(0, 1)
    # include horizontal line from reluctant randomS
    last_rr_idx = next(i for i in reversed(range(len(ll_values["rr"]))) if not np.isnan(ll_values["rr"][i]))
    plt.axhline(y=ll_values["rr"][last_rr_idx], color=colors["rr"], linestyle='--', xmin=0.15, xmax=1)
    
    plt.xlabel("Random Prob")
    plt.ylabel("Average Final Log-Likelihood")
    plt.title(f"Average Final LL vs Random Prob (Nodes: {num_nodes}, Groups: {num_groups}, Mode: {mode_number})")
    plt.legend()
    plt.grid()
    plt.savefig(f"plots/SBM({num_nodes}, {num_groups}, t{mode_number}0, X)_cost.png")
    plt.show()


def extract_final_nmi_vs_random_prob_greedy_and_reluctant(results_folder, num_nodes, num_groups, mode_number):
    pattern = re.compile(rf"SBM\({num_nodes}, {num_groups}, t{mode_number}0, ([0-9.]+)\)_results\.json")
    
    random_probs = []
    nmi_values = {"gr": [], "rr": []}
    
    for file in os.listdir(results_folder):
        match = pattern.match(file)
        if match:
            random_prob = match.group(1)
            random_probs.append(float(random_prob))
            
            with open(os.path.join(results_folder, file), 'r') as f:
                data = json.load(f)
            
            final_nmis = {"gr": [], "rr": []}
            
            for key in data["cost_data"]:
                if "nmi_gr" in data["cost_data"][key]:
                    final_nmis["gr"].append(data["cost_data"][key]["nmi_gr"])
                if "nmi_rr" in data["cost_data"][key]:
                    final_nmis["rr"].append(data["cost_data"][key]["nmi_rr"])
            
            for method in nmi_values:
                if final_nmis[method]:
                    nmi_values[method].append(np.mean(final_nmis[method]))
                else:
                    nmi_values[method].append(np.nan)

    sorted_indices = np.argsort(random_probs)
    random_probs = np.array(random_probs)[sorted_indices]
    
    for method in nmi_values:
        nmi_values[method] = np.array(nmi_values[method])[sorted_indices]
    
    plt.figure(figsize=(10, 6))
    colors = {"gr": "orange", "rr": "purple"}
    for method, label in zip(["gr", "rr"], ["Greedy Random", "Reluctant Random"]):
        plt.plot(random_probs, nmi_values[method], marker='o', label=label, color=colors[method])
    
    # include horizontal line from reluctant random
    plt.xlim(0, 1)
    # include horizontal line from reluctant randomS
    last_rr_idx = next(i for i in reversed(range(len(nmi_values["rr"]))) if not np.isnan(nmi_values["rr"][i]))
    plt.axhline(y=nmi_values["rr"][last_rr_idx], color=colors["rr"], linestyle='--', xmin=0.15, xmax=1)
    
    plt.xlabel("Random Prob")
    plt.ylabel("Average Final NMI")
    plt.title(f"Average Final NMI vs Random Prob (Nodes: {num_nodes}, Groups: {num_groups}, Mode: {mode_number})")
    plt.legend()
    plt.grid()
    plt.savefig(f"plots/SBM({num_nodes}, {num_groups}, t{mode_number}0, X)_nmi.png")
    plt.show()


if __name__ == "__main__":
    results_folder = r'C:\Projects\Heuristics for combinatorial optimisation\results'

    extract_final_ll_vs_random_prob_random_greedy_and_reluctant(results_folder, num_nodes = 10000, num_groups = 2, mode_number = 7)

    extract_final_nmi_vs_random_prob_greedy_and_reluctant(results_folder, num_nodes = 10000, num_groups = 2, mode_number = 7)