from sortedcontainers import SortedList
from collections import defaultdict

# SortedList to store (cost, node_index, color_index) tuples sorted by cost
sorted_list = SortedList()
# dictionary to map node_index to a set of (cost, node_index, color_index) tuples
node_to_tuples = defaultdict(set)

def add_entry(cost, node_index, color_index):
    # Add the tuple to the SortedList
    sorted_list.add((cost, node_index, color_index))
    # Add the tuple to the set in the dictionary
    node_to_tuples[node_index].add((cost, node_index, color_index))

def remove_by_node_index(node_index):   
    # Retrieve all tuples for the given node_index
    entries = node_to_tuples.pop(node_index)
    # Remove each tuple from the SortedList
    for entry in entries:
        sorted_list.remove(entry)

add_entry(-5, 1, 2)
add_entry(0, 1, 1)
add_entry(2, 1, 3)
add_entry(-2, 2, 1)

print("Before removal:", sorted_list)
print(node_to_tuples)
remove_by_node_index(1)
print("After removal:", sorted_list) 
print(node_to_tuples)

for x in range(1):
    print('yesy')