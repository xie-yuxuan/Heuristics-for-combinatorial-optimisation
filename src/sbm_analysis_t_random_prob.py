import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt

def sbm_plot_final_ll_vs_random_prob(results_folder, num_nodes, num_groups, mode_number):
    pattern = re.compile(rf"SBM\({num_nodes}, {num_groups}, t{mode_number}0, (\d+\.\d+)\)_results\.json")
    
    random_probs = []
    ll_values = {"g": [], "r": [], "gr": [], "rr": []}
    
    for file in os.listdir(results_folder):
        match = pattern.match(file)
        if match:
            random_prob = float(match.group(1))
            random_probs.append(random_prob)
            
            with open(os.path.join(results_folder, file), 'r') as f:
                data = json.load(f)
            
            final_lls = {"g": [], "r": [], "gr": [], "rr": []}
            
            for key in data["cost_data"]:
                final_lls["g"].append(data["cost_data"][key]["cost_data_g"][-1])
                final_lls["r"].append(data["cost_data"][key]["cost_data_r"][-1])
                final_lls["gr"].append(data["cost_data"][key]["cost_data_gr"][-1])
                final_lls["rr"].append(data["cost_data"][key]["cost_data_rr"][-1])
            
            for method in ll_values:
                ll_values[method].append(np.mean(final_lls[method]))
    
    sorted_indices = np.argsort(random_probs)
    random_probs = np.array(random_probs)[sorted_indices]
    
    for method in ll_values:
        ll_values[method] = np.array(ll_values[method])[sorted_indices]
    
    plt.figure(figsize=(10, 6))
    colors = {"g": "red", "r": "green", "gr": "orange", "rr": "purple"}
    for method, label in zip(["g", "r", "gr", "rr"], ["Greedy", "Reluctant", "Greedy Random", "Reluctant Random"]):
        plt.plot(random_probs, ll_values[method], marker='o', label=label, color=colors[method])
    
    plt.xlabel("Random Prob")
    plt.ylabel("Average Final Log-Likelihood")
    plt.title(f"Average Final LL vs Random Prob (Nodes: {num_nodes}, Groups: {num_groups}, Mode: {mode_number})")
    plt.legend()
    plt.grid()
    plt.savefig(f"plots/SBM({num_nodes}, {num_groups}, t{mode_number}0, random_prob)_cost.png")
    plt.show()

def sbm_plot_final_nmi_vs_random_prob(results_folder, num_nodes, num_groups, mode_number):
    pattern = re.compile(rf"SBM\({num_nodes}, {num_groups}, t{mode_number}0, ([0-9.]+)\)_results\.json")
    
    random_probs = []
    nmi_values = {"g": [], "r": [], "gr": [], "rr": []}
    colors = {"g": "red", "r": "green", "gr": "orange", "rr": "purple"}
    
    for file in os.listdir(results_folder):
        match = pattern.match(file)
        if match:
            random_prob = match.group(1)
            random_probs.append(float(random_prob))
            
            with open(os.path.join(results_folder, file), 'r') as f:
                data = json.load(f)
            
            final_nmis = {"g": [], "r": [], "gr": [], "rr": []}
            
            for key in data["cost_data"]:
                final_nmis["g"].append(data["cost_data"][key]["nmi_g"])
                final_nmis["r"].append(data["cost_data"][key]["nmi_r"])
                final_nmis["gr"].append(data["cost_data"][key]["nmi_gr"])
                final_nmis["rr"].append(data["cost_data"][key]["nmi_rr"])
            
            for method in nmi_values:
                nmi_values[method].append(np.mean(final_nmis[method]))
    
    sorted_indices = np.argsort(random_probs)
    random_probs = np.array(random_probs)[sorted_indices]
    
    for method in nmi_values:
        nmi_values[method] = np.array(nmi_values[method])[sorted_indices]
    
    plt.figure(figsize=(10, 6))
    for method, label in zip(["g", "r", "gr", "rr"], ["Greedy", "Reluctant", "Greedy Random", "Reluctant Random"]):
        plt.plot(random_probs, nmi_values[method], marker='o', label=label, color=colors[method])
    
    plt.xlabel("Random Prob")
    plt.ylabel("Average Final NMI")
    plt.title(f"Average Final NMI vs Random Prob (Nodes: {num_nodes}, Groups: {num_groups}, Mode: {mode_number})")
    plt.legend()
    plt.grid()
    plt.savefig(f"plots/SBM({num_nodes}, {num_groups}, t{mode_number}0, random_prob)_nmi.png")
    plt.show()

def plot_iterations_vs_random_prob_sbm_instance(results_folder, num_nodes_fixed=10000, num_groups_fixed=2, mode_number=7, instance_number=0):
    """
    Plots average number of iterations to convergence vs random probability for GR and RR
    on SBM graphs with fixed num_nodes, num_groups, mode_number, and instance_number.
    Includes ±2 standard deviation shading.
    """


    random_probs = [0, 0.05, 0.1, 0.15, 0.3, 0.4, 0.5, 0.51]
    results = {
        'gr': {p: [] for p in random_probs},
        'rr': {p: [] for p in random_probs}
    }

    mode_instance_prefix = f"t{mode_number}{instance_number}"

    for filename in os.listdir(results_folder):
        if not filename.endswith("_results.json") or not filename.startswith("SBM"):
            continue

        try:
            name = filename.replace("SBM(", "").replace(")_results.json", "")
            parts = name.split(', ')
            num_nodes = int(parts[0])
            num_groups = int(parts[1])
            mode_instance = parts[2]
            random_prob = float(parts[3])
        except:
            continue

        if (num_nodes != num_nodes_fixed or
            num_groups != num_groups_fixed or
            mode_instance != mode_instance_prefix or
            random_prob not in random_probs):
            continue

        file_path = os.path.join(results_folder, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)

        cost_data = data.get("cost_data", {})

        for run in cost_data.values():
            if "cost_data_gr" in run:
                results['gr'][random_prob].append(len(run["cost_data_gr"]))
            if "cost_data_rr" in run:
                results['rr'][random_prob].append(len(run["cost_data_rr"]))

    # Compute average iterations and std dev
    gr_avg = [np.mean(results['gr'][p]) if results['gr'][p] else np.nan for p in random_probs]
    rr_avg = [np.mean(results['rr'][p]) if results['rr'][p] else np.nan for p in random_probs]
    gr_std = [np.std(results['gr'][p], ddof=1) if results['gr'][p] else np.nan for p in random_probs]
    rr_std = [np.std(results['rr'][p], ddof=1) if results['rr'][p] else np.nan for p in random_probs]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(random_probs, gr_avg, color='orange', marker='o', label='Greedy Random')
    ax.plot(random_probs, rr_avg, color='purple', marker='s', label='Reluctant Random')
    ax.fill_between(random_probs, np.array(gr_avg) - 2 * np.array(gr_std), np.array(gr_avg) + 2 * np.array(gr_std),
                    color='orange', alpha=0.2, label='GR ±2 SD')
    ax.fill_between(random_probs, np.array(rr_avg) - 2 * np.array(rr_std), np.array(rr_avg) + 2 * np.array(rr_std),
                    color='purple', alpha=0.2, label='RR ±2 SD')

    ax.set_xlabel("Random Probability")
    ax.set_ylabel("Average Number of Iterations to Convergence")
    ax.set_title(f"Iterations vs Random Probability\nSBM(N={num_nodes_fixed}, Groups={num_groups_fixed}, t{mode_number}{instance_number})")
    ax.legend()
    ax.grid(True)

    os.makedirs("plots", exist_ok=True)
    save_path = f"plots/sbm_avg_iterations_vs_random_prob_N{num_nodes_fixed}_G{num_groups_fixed}_t{mode_number}{instance_number}.png"
    # plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results_folder = r'C:\Projects\Heuristics for combinatorial optimisation\results'

    # sbm_plot_final_ll_vs_random_prob(results_folder, num_nodes=10000, num_groups=2, mode_number=2)

    # sbm_plot_final_nmi_vs_random_prob(results_folder, num_nodes=10000, num_groups=2, mode_number=7)

    plot_iterations_vs_random_prob_sbm_instance(
        results_folder,
        num_nodes_fixed=10000,
        num_groups_fixed=2,
        mode_number=7,
        instance_number=0
    )