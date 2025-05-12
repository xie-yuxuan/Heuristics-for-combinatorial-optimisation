import json
import os
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.sankey import Sankey
from collections import Counter
from networkx.readwrite import json_graph

from utils import calc_cost

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
    initial_node_colors = data["initial_node_colors"]
    
    graph_data = data["graph_data"]
    graph = json_graph.node_link_graph(graph_data)

    # uncomment to get adj matrix
    # adj_matrix = nx.adjacency_matrix(graph, weight='weight').toarray()
    
    return graph, graph_name, color_set_size, degree, num_nodes, gaussian_mean, gaussian_variance, initial_node_colors

def sankey_basin_data(Sg, Sr):
    plt.figure(figsize=(10, 6))
    for key, final_g in Sg.items():
        plt.plot([0, 1], [int(key), final_g], color='red', alpha=0.01)
    for key, final_r in Sr.items():
        plt.plot([0, 1], [int(key), final_r], color='green', alpha=0.01)
    
    unique_final_g = set(Sg.values())
    unique_final_r = set(Sr.values())
    print(f"Unique greedy final colorings: {len(unique_final_g)}")
    print(f"Unique reluctant final colorings: {len(unique_final_r)}")

    plt.title('Initial to Final Coloring Mapping')
    plt.xlabel('Initial Coloring (Left) → Final Coloring (Right)')
    plt.xticks([0, 1], ['Initial', 'Final'])
    plt.ylabel('Initial Coloring')
    plt.yticks(range(0, 2 ** num_nodes, 5000))  # Label every x intervals

    # Analysis Text Box
    # analysis_text = (
    #                  f"Number of final colors (greedy): {len(unique_final_g)}\n"
    #                  f"Number of final colors (reluctant): {len(unique_final_r)}")
    # plt.gca().text(0.65, 0.9, analysis_text, fontsize=10, va='center', ha='left', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black'))
    
    # Show the plot

    plt.savefig(f"plots/{graph_name}_basin_sankey.png")
    plt.show()

def heatmap_basin_data(Sg, Sr):
    matrix_size = 2 ** num_nodes
    matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    # Populate the matrix based on Sg and Sr
    for key, final_g in Sg.items():
        matrix[matrix_size - 1 - int(final_g), int(key)] |= 1  # Mark red for greedy (Sg)

    for key, final_r in Sr.items():
        matrix[matrix_size - 1 - int(final_r), int(key)] |= 2  # Mark green for reluctant (Sr)

    # Create the color map
    color_map = np.zeros((matrix_size, matrix_size, 3))  # RGB values

    # Red for greedy (Sg)
    color_map[matrix == 1] = [1, 0, 0]
    # Green for reluctant (Sr)
    color_map[matrix == 2] = [0, 1, 0]
    # Yellow for both (Sg and Sr)
    color_map[matrix == 3] = [1, 1, 0]

    # Plotting the heatmap
    plt.figure(figsize=(10, 10))
    plt.imshow(color_map, aspect='auto', interpolation='nearest')
    plt.title('Heatmap: Initial to Final Coloring Mapping')
    plt.xlabel('Initial Coloring')
    plt.ylabel('Final Coloring')

    # Add ticks for better readability
    plt.xticks(range(0, matrix_size, 50))
    plt.yticks(range(0, matrix_size, 50))

    # Invert the y-axis to have 0 (bottom) and 1000 (top)
    plt.gca().invert_yaxis()

    plt.savefig(f"plots/{graph_name}_basin_heatmap.png")

    plt.show()

def plot_color_mapping(Sg, Sr):
    """
    Plots a bar graph showing the number of initial colorings mapped to each final coloring
    using the greedy (red) and reluctant (green) algorithms, sorted by final coloring.
    
    Args:
    - Sg (dict): Dictionary containing initial colorings as keys and final colorings as values for the greedy algorithm.
    - Sr (dict): Dictionary containing initial colorings as keys and final colorings as values for the reluctant algorithm.
    """
    
    # Count the occurrences of each final coloring in both Sg and Sr
    final_colorings_greedy = Counter(Sg.values())
    final_colorings_reluctant = Counter(Sr.values())
    
    # Create a list of unique final colorings (keys from both final_colorings_greedy and final_colorings_reluctant)
    all_final_colorings = set(final_colorings_greedy.keys()).union(set(final_colorings_reluctant.keys()))
    
    # Prepare the data for plotting
    greedy_counts = []
    reluctant_counts = []
    final_colorings = []
    
    for final_coloring in sorted(all_final_colorings):
        # Get the count from final_colorings_greedy and final_colorings_reluctant, default to 0 if not present
        greedy_counts.append(final_colorings_greedy.get(final_coloring, 0))
        reluctant_counts.append(final_colorings_reluctant.get(final_coloring, 0))
        final_colorings.append(final_coloring)
    
    # Set up the bar width and positions for the bars
    bar_width = 0.35
    index = range(len(final_colorings))
    
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar1 = ax.bar(index, greedy_counts, bar_width, label='Greedy', color='red')
    bar2 = ax.bar([i + bar_width for i in index], reluctant_counts, bar_width, label='Reluctant', color='green')
    
    # Add labels and title
    ax.set_xlabel('Final Coloring')
    ax.set_ylabel('Number of Initial Colorings')
    ax.set_title('Number of Initial Colorings Mapped to Each Final Coloring')
    
    # Set the x-ticks to the final colorings
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(final_colorings, rotation=90)
    
    # Add legend
    ax.legend()
    
    # Display the plot
    plt.tight_layout()

    # plt.savefig(f"plots/{graph_name}_basin_bar.png")
    plt.show()

def plot_hist_color_mapping(Sg, Sr, Sgr1, Srr1, seed):
    Sum_count_greedy = [0] * (2 ** num_nodes)
    Sum_count_reluctant = [0] * (2 ** num_nodes)
    Sum_count_greedy_random = [0] * (2 ** num_nodes)
    Sum_count_reluctant_random = [0] * (2 ** num_nodes)

    for final_g in Sg.values():
        Sum_count_greedy[int(final_g)] += 1
    for final_r in Sr.values():
        Sum_count_reluctant[int(final_r)] += 1
    for final_gr in Sgr1.values():
        Sum_count_greedy_random[int(final_gr)] += 1
    for final_rr in Srr1.values():
        Sum_count_reluctant_random[int(final_rr)] += 1

    # Filter out zeros
    Sum_count_greedy = [x for x in Sum_count_greedy if x != 0]
    Sum_count_reluctant = [x for x in Sum_count_reluctant if x != 0]
    Sum_count_greedy_random = [x for x in Sum_count_greedy_random if x != 0]
    Sum_count_reluctant_random = [x for x in Sum_count_reluctant_random if x != 0]

    plt.figure(figsize=(10, 6))

    plt.hist(Sum_count_greedy, bins=len(Sum_count_greedy), color='red', alpha=0.5, label='Greedy')
    plt.hist(Sum_count_reluctant, bins=len(Sum_count_reluctant), color='green', alpha=0.5, label='Reluctant')
    plt.hist(Sum_count_greedy_random, bins=len(Sum_count_greedy_random), color='purple', alpha=0.5, label='Greedy Random 0.1')
    plt.hist(Sum_count_reluctant_random, bins=len(Sum_count_reluctant_random), color='orange', alpha=0.5, label='Reluctant Random 0.1')

    plt.xlabel("Basin Size")
    plt.ylabel("Count")
    plt.title(f"Histogram of Basin Sizes for N={num_nodes}, k={regular_degree}, c={color_set_size}, seed={seed}")
    plt.legend()

    plt.savefig(f"plots/{graph_name}_basin_hist.png")
    
    plt.show()

def compute_basin_costs(basin_dict, graph):
    basin_sizes = Counter(basin_dict.values())
    costs = {}
    for final_coloring in basin_sizes:
        binary_coloring = format(int(final_coloring), f'0{len(graph.nodes)}b')
        for node_idx, bit in enumerate(binary_coloring):
            graph.nodes[node_idx]['color'] = int(bit)
        cost = calc_cost(graph)
        costs[final_coloring] = cost
    return list(basin_sizes.values()), list(costs.values())

def plot_scatter_basin_cost(Sg, Sr, Sgr1, Srr1, Sgr3, Srr3, graph, seed):
# def plot_scatter_basin_cost(Sg, Sr, graph, seed):
    plt.figure(figsize=(10, 6))

    # Compute and plot deterministic greedy (circle)
    sizes_g, costs_g = compute_basin_costs(Sg, graph)
    plt.scatter(sizes_g, costs_g, color='red', marker='o', alpha=0.5, label='Greedy')

    # Compute and plot deterministic reluctant (circle)
    sizes_r, costs_r = compute_basin_costs(Sr, graph)
    plt.scatter(sizes_r, costs_r, color='green', marker='o', alpha=0.5, label='Reluctant')

    # Compute and plot greedy random 0.1 (cross)
    sizes_gr1, costs_gr1 = compute_basin_costs(Sgr1, graph)
    plt.scatter(sizes_gr1, costs_gr1, color='purple', marker='x', alpha=0.5, label='Greedy Random 0.1')

    # Compute and plot reluctant random 0.1 (cross)
    sizes_rr1, costs_rr1 = compute_basin_costs(Srr1, graph)
    plt.scatter(sizes_rr1, costs_rr1, color='orange', marker='x', alpha=0.5, label='Reluctant Random 0.1')

    # Compute and plot greedy random 0.3 (triangle)
    sizes_gr3, costs_gr3 = compute_basin_costs(Sgr3, graph)
    plt.scatter(sizes_gr3, costs_gr3, color='purple', marker='^', alpha=0.5, label='Greedy Random 0.3')

    # Compute and plot reluctant random 0.3 (triangle)
    sizes_rr3, costs_rr3 = compute_basin_costs(Srr3, graph)
    plt.scatter(sizes_rr3, costs_rr3, color='orange', marker='^', alpha=0.5, label='Reluctant Random 0.3')

    plt.xlabel("Basin Size")
    plt.ylabel("Cost Function")
    plt.title(f"Cost Function against basin size for N={num_nodes}, d={regular_degree}, c={color_set_size}, seed={seed}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"plots/{graph_name}_basin_scatter.png")
    plt.show()

def plot_scatter_all_seeds_for_config(results_folder, graph_folder, num_nodes, regular_degree, color_set_size):
    marker_map = {
        1: 'o',  # circle
        2: 's',  # square
        3: '^',  # triangle
        4: 'v',
        5: 'D',
        6: 'P',
        7: '*',
    }

    plt.figure(figsize=(10, 6))

    for file in os.listdir(results_folder):
        if not file.endswith("_basin_results.json"):
            continue

        try:
            basename = file.replace("_basin_results.json", "")
            config = eval(basename)
            if (
                not isinstance(config, tuple) or
                len(config) != 4 or
                config[0] != num_nodes or
                config[1] != regular_degree or
                config[2] != color_set_size
            ):
                continue  # Skip files not matching config
            _, _, _, seed = config
        except Exception:
            continue  # Skip malformed filenames

        marker = marker_map.get(seed, 'x')
        result_path = os.path.join(results_folder, file)
        graph_path = os.path.join(graph_folder, f"{config}.json")

        try:
            graph, *_ = load_graph_from_json(graph_path)
        except Exception as e:
            print(f"Failed to load graph {graph_path}: {e}")
            continue

        with open(result_path, 'r') as f:
            data = json.load(f)

        Sg = data["basin_data"].get("Sg", {})
        Sr = data["basin_data"].get("Sr", {})

        sizes_g, costs_g = compute_basin_costs(Sg, graph)
        sizes_r, costs_r = compute_basin_costs(Sr, graph)

        plt.scatter(np.log(sizes_g), costs_g, color='red', marker=marker, alpha=0.5, label=f'Greedy Init {seed}')
        plt.scatter(np.log(sizes_r), costs_r, color='green', marker=marker, alpha=0.5, label=f'Reluctant Init {seed}')

    plt.xlabel("Basin Size")
    plt.ylabel("Cost Function")
    plt.title(f"Cost Function against Log Basin size for N={num_nodes}, d={regular_degree}, c={color_set_size}")

    # Deduplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize='small')

    plt.tight_layout()
    filename = f"plots/({num_nodes}, {regular_degree}, {color_set_size})_basin_results_all_init.png"
    # plt.savefig(filename)
    plt.show()

if __name__ == "__main__":

    num_nodes = 20
    regular_degree = 10
    color_set_size = 2
    init = 1

    graph_path = f"C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\graphs\({num_nodes}, {regular_degree}, {color_set_size}, {init}).json"
    graph_folder = r'C:\Projects\Heuristics for combinatorial optimisation\Heuristics-for-combinatorial-optimisation\data\graphs'

    graph, graph_name, color_set_size, degree, num_nodes, gaussian_mean, gaussian_variance, initial_node_colors = load_graph_from_json(graph_path)

    # results_path = f"C:\Projects\Heuristics for combinatorial optimisation\results\({num_nodes}, {regular_degree}, {color_set_size})_basin_results.json"

    # with open(results_path, 'r') as f:
    #     data = json.load(f)

    results_folder = r"C:\Projects\Heuristics for combinatorial optimisation\results"
    results_path = f"({num_nodes}, {regular_degree}, {color_set_size}, {init})_basin_results.json"
    results_path = os.path.join(results_folder, results_path)

    with open(results_path, 'r') as f:
        data = json.load(f)

    basin_data = data['basin_data']
    Sg = basin_data['Sg']
    Sr = basin_data['Sr']
    Sgr1 = basin_data['Sgr1']
    Srr1 = basin_data['Srr1']
    Sgr3 = basin_data['Sgr3']
    Srr3 = basin_data['Srr3']

    # sankey_basin_data(Sg, Sr)

    # heatmap_basin_data(Sg, Sr)

    # plot_color_mapping(Sg, Sr)

    # plot_hist_color_mapping(Sg, Sr, Sgr1, Srr1, init)

    # plot_scatter_basin_cost(Sg, Sr, Sgr1, Srr1, Sgr3, Srr3, graph, init)
    # plot_scatter_basin_cost(Sg, Sr, graph, init)

    plot_scatter_all_seeds_for_config(results_folder, graph_folder, num_nodes, regular_degree, color_set_size)




