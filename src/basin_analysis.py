import json
import os
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.sankey import Sankey
from collections import Counter

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

    plt.savefig(f"plots/{graph_name}_basin_bar.png")
    plt.show()

def plot_hist_color_mapping(Sg, Sr):
    Sum_count_greedy = [0] * (2 ** num_nodes)
    Sum_count_reluctant = [0] * (2 ** num_nodes)

    for key, final_g in Sg.items():
        Sum_count_greedy[int(final_g)] += 1

    for key, final_r in Sr.items():
        Sum_count_reluctant[int(final_r)] += 1

    # remove all 0s from the lists
    Sum_count_greedy = [x for x in Sum_count_greedy if x != 0]
    Sum_count_reluctant = [x for x in Sum_count_reluctant if x != 0]

    num_bins_greedy = len(Sum_count_greedy)
    num_bins_reluctant = len(Sum_count_reluctant)

    plt.figure(figsize=(10, 6))
    print(Sum_count_greedy)
    print(Sum_count_reluctant)
    plt.hist(Sum_count_greedy, bins = num_bins_greedy, color='red', alpha=0.5, label='Greedy')
    plt.hist(Sum_count_reluctant, bins = num_bins_reluctant, color='green', alpha=0.5, label='Reluctant')
    
    plt.xlabel("Basin Size")
    plt.ylabel("Count")
    plt.title(f"Histogram of Basin Sizes for N={num_nodes}, k={regular_degree}, c={color_set_size}")
    plt.legend()

    plt.savefig(f"plots/{graph_name}_basin_hist.png")
    
    plt.show()



if __name__ == "__main__":

    num_nodes = 16
    regular_degree = 8
    color_set_size = 2

    base_path = r"C:\Projects\Heuristics for combinatorial optimisation\results"

    file_path = f"({num_nodes}, {regular_degree}, {color_set_size})_basin_results.json"
    file_path = os.path.join(base_path, file_path)

    with open(file_path, 'r') as f:
        data = json.load(f)

    graph_name = data['graph_name']
    basin_data = data['basin_data']
    Sg = basin_data['Sg']
    Sr = basin_data['Sr']

    # sankey_basin_data(Sg, Sr)

    # heatmap_basin_data(Sg, Sr)

    # plot_color_mapping(Sg, Sr)

    plot_hist_color_mapping(Sg, Sr)


