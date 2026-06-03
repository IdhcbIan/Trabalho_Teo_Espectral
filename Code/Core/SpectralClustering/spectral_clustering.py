"""
Spectral Clustering on the DINOv2 KNN Graph
-------------------------------------------

Goal:
    Cluster the Flowers images using a graph built from DINOv2 nearest-neighbor
    rankings.

Pipeline:

    1. Load the ranking JSON produced by Extract.py.
    2. Build a weighted KNN graph W.
    3. Compute the normalized graph Laplacian:

           L = I - D^(-1/2) W D^(-1/2)

    4. Take the smallest eigenvectors of L.
    5. Normalize each row of the eigenvector matrix.
    6. Run KMeans in spectral space.
    7. Evaluate clusters with Flowers labels.

Labels are used only at the end for evaluation. They are not used to create
the spectral clusters.
"""

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from Aux import (
    plot_first_two_embedding_coordinates,
    plot_graph_by_cluster,
    plot_graph_by_correctness,
)


ROOT_DIR = Path(__file__).resolve().parents[3]
DATASETS_DIR = ROOT_DIR / "DataSets"


def main():
    parser = argparse.ArgumentParser(description="Spectral clustering on a DINOv2 KNN graph.")
    parser.add_argument("--ranking", type=Path, default=DATASETS_DIR / "Runs" / "dinov2_vits14_output.json")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--num-clusters", type=int, default=17)
    parser.add_argument("--samples-per-class", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layout-k", type=float, default=0.34)
    parser.add_argument("--layout-iterations", type=int, default=180)
    parser.add_argument("--layout-scale", type=float, default=1.35)
    parser.add_argument("--output-dir", type=Path, default=DATASETS_DIR / "Spectral")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1. Load nearest-neighbor rankings
    # ---------------------------------------------------------------------
    print("\n[1] Loading nearest-neighbor rankings")

    with open(args.ranking, "r") as f:
        rankings = np.array(json.load(f), dtype=int)

    if rankings.ndim != 2:
        raise ValueError("Ranking JSON must be a 2D list.")

    n_samples, n_ranked_neighbors = rankings.shape
    print(f"Rankings shape: {rankings.shape}")

    if args.k >= n_ranked_neighbors:
        raise ValueError(
            f"k={args.k} is too large. Ranking file has only {n_ranked_neighbors} entries per row."
        )

    # ---------------------------------------------------------------------
    # 2. Build the weighted KNN graph W
    # ---------------------------------------------------------------------
    print("\n[2] Building weighted KNN graph W")

    # Row i of the ranking file contains neighbors of image i.
    # The first entry is usually i itself, so we skip it.
    #
    # Edge weights are simple and readable:
    #
    #     rank 1 neighbor -> weight 1.0
    #     rank 2 neighbor -> weight 0.5
    #     rank r neighbor -> weight 1/r
    W = np.zeros((n_samples, n_samples), dtype=float)

    for i in range(n_samples):
        neighbors = rankings[i, 1 : args.k + 1]
        for rank, j in enumerate(neighbors, start=1):
            if i == j:
                continue
            W[i, j] = max(W[i, j], 1.0 / rank)

    # Spectral clustering expects an undirected graph.
    # If either i links to j or j links to i, keep the strongest edge.
    W = np.maximum(W, W.T)

    graph = nx.Graph()
    graph.add_nodes_from(range(n_samples))
    rows, cols = np.nonzero(np.triu(W, k=1))
    for i, j in zip(rows, cols):
        graph.add_edge(int(i), int(j), weight=float(W[i, j]))

    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    # ---------------------------------------------------------------------
    # 3. Compute the normalized graph Laplacian
    # ---------------------------------------------------------------------
    print("\n[3] Computing normalized graph Laplacian")

    # SciPy computes:
    #
    #     L = I - D^(-1/2) W D^(-1/2)
    #
    # where D is the diagonal degree matrix.
    L = csgraph.laplacian(W, normed=True)

    # ---------------------------------------------------------------------
    # 4. Take the smallest eigenvectors of L
    # ---------------------------------------------------------------------
    print("\n[4] Computing smallest eigenvectors")

    # For the normalized Laplacian, cluster structure appears in the
    # eigenvectors associated with the smallest eigenvalues.
    eigenvalues, U = eigsh(L, k=args.num_clusters, which="SM")

    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    U = U[:, order]

    print(f"U shape: {U.shape}")

    # ---------------------------------------------------------------------
    # 5. Normalize each row of U
    # ---------------------------------------------------------------------
    print("\n[5] Row-normalizing eigenvectors")

    row_norms = np.linalg.norm(U, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    Y = U / row_norms

    # ---------------------------------------------------------------------
    # 6. Run KMeans in spectral space
    # ---------------------------------------------------------------------
    print("\n[6] Running KMeans")

    clusters = KMeans(
        n_clusters=args.num_clusters,
        n_init=50,
        random_state=args.seed,
    ).fit_predict(Y)

    # ---------------------------------------------------------------------
    # 7. Evaluate with Flowers labels
    # ---------------------------------------------------------------------
    print("\n[7] Evaluating clusters")

    expected = args.num_clusters * args.samples_per_class
    if n_samples != expected:
        raise ValueError(f"Expected {expected} Flowers samples, found {n_samples}.")

    labels = np.repeat(np.arange(args.num_clusters), args.samples_per_class)

    # KMeans cluster ids are arbitrary. We use the Hungarian algorithm to find
    # the best cluster->class mapping for accuracy reporting.
    confusion = np.zeros((args.num_clusters, args.num_clusters), dtype=int)

    for cluster_id in range(args.num_clusters):
        for class_id in range(args.num_clusters):
            confusion[cluster_id, class_id] = np.sum(
                (clusters == cluster_id) & (labels == class_id)
            )

    rows, cols = linear_sum_assignment(confusion.max() - confusion)
    cluster_to_class = {int(row): int(col) for row, col in zip(rows, cols)}
    mapped_labels = np.array([cluster_to_class[c] for c in clusters])

    accuracy = np.mean(mapped_labels == labels)
    ari = adjusted_rand_score(labels, clusters)
    nmi = normalized_mutual_info_score(labels, clusters)

    print(f"Accuracy after cluster-label matching: {accuracy:.4f}")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")
    print(f"Eigenvalues: {[round(float(v), 6) for v in eigenvalues]}")
    print(f"Cluster mapping: {cluster_to_class}")

    # ---------------------------------------------------------------------
    # 8. Compute graph layout and export JSON
    # ---------------------------------------------------------------------
    print("\n[8] Computing graph layout and exporting JSON")

    positions = nx.spring_layout(
        graph,
        k=args.layout_k,
        iterations=args.layout_iterations,
        seed=args.seed,
        weight="weight",
        scale=args.layout_scale,
    )

    graph_json = {
        "nodes": {},
        "edges": {},
        "graph_info": {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "k": args.k,
            "method": "knn_graph_spectral_clustering",
            "eigenvalues": [float(v) for v in eigenvalues],
        },
    }

    for node_id in graph.nodes:
        graph_json["nodes"][str(node_id)] = {
            "position": [
                float(positions[node_id][0]),
                float(positions[node_id][1]),
            ],
            "attributes": {
                "true_label": int(labels[node_id]),
                "cluster": int(clusters[node_id]),
                "mapped_label": int(mapped_labels[node_id]),
                "correct": bool(labels[node_id] == mapped_labels[node_id]),
            },
        }

    for node_id in graph.nodes:
        graph_json["edges"][str(node_id)] = [
            int(neighbor)
            for neighbor in graph.neighbors(node_id)
        ]

    output_stem = f"spectral_clustering_k{args.k}_c{args.num_clusters}"

    with open(args.output_dir / f"{output_stem}.json", "w") as f:
        json.dump(graph_json, f, indent=2)

    # ---------------------------------------------------------------------
    # 9. Save arrays and plots
    # ---------------------------------------------------------------------
    print("\n[9] Saving outputs")

    np.save(args.output_dir / f"{output_stem}_clusters.npy", clusters)
    np.save(args.output_dir / f"{output_stem}_mapped_labels.npy", mapped_labels)
    np.save(args.output_dir / f"{output_stem}_embedding.npy", Y)
    np.save(args.output_dir / f"{output_stem}_eigenvalues.npy", eigenvalues)
    np.save(args.output_dir / f"{output_stem}_confusion.npy", confusion)

    plot_graph_by_cluster(
        graph,
        positions,
        clusters,
        args.output_dir / f"{output_stem}.png",
    )
    plot_graph_by_correctness(
        graph,
        positions,
        labels,
        mapped_labels,
        args.output_dir / f"{output_stem}_correctness.png",
    )
    plot_first_two_embedding_coordinates(
        Y,
        clusters,
        args.output_dir / f"{output_stem}_embedding.png",
    )

    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
