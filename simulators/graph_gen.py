import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def barabasi_albert_with_pareto(n=10, pshape=10, power=1, zero_appeal=0.1, to_plot=False):
    """
    Generate a Barabási-Albert graph with power and zero_appeal parameters.

    Parameters:
    - n: Total number of nodes.
    - m: Number of edges to attach from a new node to existing nodes.
    - power: The exponent applied to node degree during attachment (default is 1).
    - zero_appeal: Baseline probability for nodes with zero degree (default is 0.1).

    Returns:
    - A dictionary representing the graph (adjacency list format).
    """

    # Initialize the graph with m nodes fully connected
    graph = {0: set()}

    # Add remaining nodes, preferentially attaching them
    for new_node in range(1, n):
        graph[new_node] = set()

        # Calculate the attachment probabilities
        targets = set()
        fanin = np.round(min(np.random.pareto(pshape, 1) + 1,len(graph)-1))
        #print(f'node {new_node} fanin: {fanin}')
        while len(targets) < fanin:
            potential_target = random.choice(list(graph.keys()))
            if potential_target in targets or potential_target == new_node:
                continue
            degree = len(graph[potential_target])

            # Apply power and zero_appeal to the attachment probability
            attachment_prob = (degree) ** power + zero_appeal            
            total_weights = sum((len(graph[node])) ** power + zero_appeal for node in graph)
            normalized_prob = attachment_prob / total_weights

            # Add target if it passes the probabilistic selection
            if random.random() < normalized_prob:
                targets.add(potential_target)  # Set ensures no duplicates

        # Add edges from the new node to the selected targets
        for target in targets:
            #graph[new_node].add(target)
            graph[target].add(new_node)

    # Convert the adjacency list to a NetworkX graph
    nx_graph = nx.DiGraph()
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            nx_graph.add_edge(node, neighbor)
    
    if to_plot:
        plot_graph(nx_graph)
    return nx_graph

def plot_graph(nx_graph):
    """
    Plot the graph using matplotlib and networkx.

    Parameters:
    - graph: A dictionary representing the graph (adjacency list format).
    """

    # Draw the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(nx_graph)  # Use spring layout for better visualization
    #pos = nx.planar_layout(nx_graph)
    nx.draw(
        nx_graph, pos, node_size=300, node_color="skyblue", edge_color="darkgray", with_labels=True
    )
    plt.title("Barabási-Albert Graph", fontsize=14)
    plt.show()

# Example usage
if __name__ == "__main__":
    n = 20  # Total number of nodes
    shape = 2   # Pareto shape for edges per new node
    power = 0.9  # Adjust preferential attachment strength
    zero_appeal = 3.25  # Baseline appeal for low-degree nodes

    graph = barabasi_albert_with_pareto(n, shape, power, zero_appeal,to_plot=True)
