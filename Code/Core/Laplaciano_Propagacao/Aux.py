"""
      ____
 ____|    \
(____|     `._____
 ____|       _|___
(____|     .'
     |____/


---------------------------------

Auxiliary plotting functions for Laplacian label propagation script.

---------------------------------
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_graph_by_cluster(graph, positions, clusters, output_path):
    """
    Plot the KNN graph colored by propagated cluster.
    """
    plt.figure(figsize=(15, 10))
    nx.draw_networkx_edges(
        graph,
        positions,
        width=0.35,
        alpha=0.22,
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
    plt.colorbar(nodes, shrink=0.75, label="Laplacian cluster")
    plt.title("Laplacian Label Propagation on KNN Graph")
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
        width=0.35,
        alpha=0.22,
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
    plt.title("Laplacian Label Propagation Correctness")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_first_two_embedding_coordinates(embedding, clusters, output_path):
    """
    Plot the first two coordinates of the Laplacian embedding.
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
    plt.colorbar(scatter, shrink=0.75, label="Laplacian cluster")
    plt.title("First Two Laplacian Embedding Coordinates")
    plt.xlabel("Eigenvector coordinate 1")
    plt.ylabel("Eigenvector coordinate 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.show()
    plt.close()



def load_labels_from_manifest(manifest_path, n_samples):
    """
    Carrega os rotulos reais do manifest do dataset limpo.

    O manifest do CUB_Cleaned50 guarda class_id no padrao original do CUB:
    1, 2, ..., 50. Para comparar com KMeans, convertemos para 0, 1, ..., 49.
    """
    labels = []

    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row["class_id"]) - 1)

    labels = np.array(labels, dtype=int)

    if len(labels) != n_samples:
        raise ValueError(
            f"Manifest has {len(labels)} labels, but ranking has {n_samples} samples."
        )

    return labels
