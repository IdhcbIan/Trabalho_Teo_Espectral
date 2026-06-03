"""
Plotting helpers for the Ng-Jordan-Weiss spectral clustering script.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_graph_by_cluster(graph, positions, clusters, output_path):
    """
    Plot the graph colored by spectral cluster.
    """
    plt.figure(figsize=(15, 10))
    nx.draw_networkx_edges(
        graph,
        positions,
        width=0.28,
        alpha=0.16,
        edge_color="#c9c9c9",
    )
    nodes = nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=clusters,
        cmap="tab20",
        node_size=28,
        linewidths=0.0,
        alpha=0.9,
    )
    plt.colorbar(nodes, shrink=0.75, label="Spectral cluster")
    plt.title("Ng-Jordan-Weiss Spectral Clustering")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_graph_by_correctness(graph, positions, true_labels, mapped_labels, output_path):
    """
    Plot correct nodes in green and wrong nodes in red.
    """
    correct = true_labels == mapped_labels
    node_colors = np.where(correct, "#2e7d32", "#c62828")

    plt.figure(figsize=(15, 10))
    nx.draw_networkx_edges(
        graph,
        positions,
        width=0.28,
        alpha=0.16,
        edge_color="#c9c9c9",
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=node_colors,
        node_size=28,
        linewidths=0.0,
        alpha=0.9,
    )
    plt.title("Ng-Jordan-Weiss Correctness")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_first_two_embedding_coordinates(embedding, clusters, output_path):
    """
    Plot the first two coordinates of the spectral embedding.
    """
    if embedding.shape[1] < 2:
        return

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=clusters,
        cmap="tab20",
        s=18,
        alpha=0.9,
    )
    plt.colorbar(scatter, shrink=0.75, label="Spectral cluster")
    plt.title("First Two NJW Spectral Coordinates")
    plt.xlabel("Eigenvector coordinate 1")
    plt.ylabel("Eigenvector coordinate 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.show()
    plt.close()
