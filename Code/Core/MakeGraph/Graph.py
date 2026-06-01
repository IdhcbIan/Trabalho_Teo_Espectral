"""
---------------------------------------

// Ian Bezerra - 2026 //

This is a networkx based aproach and you can also just visualize it 
    localy, but loading your graph in the gcnvis system allows you 
    to have a interactie view and also share it with your friends!!

"""

import os
import json
import numpy as np


#-----------------------------------------------

"""
This is the main function for the KNN Graph!

just call it with rankings being your rks from run_ball_tree 
    and k_list being the list of k values [10, 20, ...]
"""

def plot_and_export(rankings, k_list, model_name):
    for k in k_list:

        G, pos =  visualize_knn_graph(rankings, k=k) 
        export_graph_positions(G, pos, model_name, k)


def visualize_knn_graph(rankings, k=16, figsize=(15, 10), title=None, 
                        show_labels=False, save_path=None):
    """
    Visualize a KNN graph based on rankings using spring layout.
    """

    import networkx as nx
    import matplotlib.pyplot as plt
    
    # Create a graph
    G = nx.DiGraph()
    
    # Add nodes to the graph
    for i in range(len(rankings)):
        G.add_node(i)
    
    # Add edges based on k-nearest neighbors
    for i in range(len(rankings)):
        # Skip the first neighbor as it's the point itself
        for j in range(1, min(k+1, len(rankings[i]))):
            if i != rankings[i][j]: 
                G.add_edge(i, rankings[i][j])
    
    # Create spring layout with increased spacing between nodes
    pos = nx.spring_layout(G, k=0.3, iterations=100, seed=42)
    #print(pos)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Draw nodes with colors
    node_sizes = [100 if i < 200 else 150 for i in range(len(rankings))]  # Larger nodes for test data
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, arrows=False)
    
    if show_labels:
        nx.draw_networkx_labels(G, pos, font_size=8)
    
    
    # Set title
    if title:
        plt.title(title)
    else:
        plt.title(f'KNN Graph (k={k})')
    
    plt.axis('off')
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print some graph statistics
    print(f"Graph info: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G, pos



def export_graph_positions(G, pos,model_name, k):
    """
    Export the graph node positions AND edges to a JSON file.
    """

    # Create export data structure
    export_data = {
        "nodes": {},
        "edges": {},  
        "graph_info": {
            "num_nodes": len(G.nodes),
            "num_edges": len(G.edges),
            "k": k
        }
    }
    
    # Add node positions
    for node_id in G.nodes():
        node_pos = pos[node_id]
        position = [float(node_pos[0]), float(node_pos[1])]
        
        node_attrs = dict(G.nodes[node_id])
        export_data["nodes"][str(node_id)] = {
            "position": position,
            "attributes": node_attrs
        }
    
    for node_id in G.nodes():
        # Get all neighbors (outgoing edges for directed graph)
        neighbors = list(G.neighbors(node_id))
        # Store as list of neighbor IDs
        export_data["edges"][str(node_id)] = [int(n) for n in neighbors]
    
    path = model_name + "_output_"+ str(k) + ".json"
    print(f"Path: {path}")
    
    # Save to JSON file
    with open(path, 'w') as f:
        json.dump(export_data, f, indent=2)

    total_edges = sum(len(e) for e in export_data["edges"].values())
    print(f"Graph exported: {len(export_data['nodes'])} nodes, {total_edges} edges")
    print(f"Saved to: {path}")
    return path

"""
-------------------------------------------------------------------------------------------
// Exaple KNN //

Lets say you have your Features and you want to create 
    your own KNN graph to upload.

OBS: You can modify the networkx code in the function
   visualize_knn_graph to generate you other types of graph.
   this. Or replave it entirely and then call export_graph_positions
   on it, this inside plot_and_export. 

   For how to extract features and a more full aproach I recoment looking at the 
   extactor inference from: https://github.com/IdhcbIan/Visualizing_GCNs
   there you will have a Graph.py just like this one with the same visualize_knn_graph
   function you can modify to produce the json plots to upload!

   You can also find some extracted features inside: 
   https://github.com/IdhcbIan/Visualizing_GCNs/tree/master/Extractor_Inference_Code/Emb


   This will produce the .json to upload in the plataform!!
"""

model_name = 'alexnet'
features = np.load('alexnet_emb.npy')


import numpy as np
from sklearn.neighbors import BallTree


def run_ball_tree(features, k=100):
    """
    Constrói uma estrutura BallTree a partir das features e retorna os rankings dos vizinhos mais próximos.
    """

    # Verifica se as features são válidas
    if not isinstance(features, np.ndarray):
        raise ValueError("As 'features' devem ser um array do tipo numpy.ndarray.")
    if features.ndim != 2:
        raise ValueError("As 'features' devem ser um array 2D no formato (n_samples, n_features).")

    # Cria a estrutura BallTree
    tree = BallTree(features)

    # Realiza a consulta para encontrar os k vizinhos mais próximos
    _, rks = tree.query(features, k=k)

    return rks


rks = run_ball_tree(features)


# Then Create the Graph!

k_list = [10, 20, 30, 40, 60, 80]

plot_and_export(rks, k_list, model_name)


#------------// End of the program //--------------------------

