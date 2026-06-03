"""
Ng-Jordan-Weiss Spectral Clustering
-----------------------------------

Paper:
    Andrew Y. Ng, Michael I. Jordan, Yair Weiss.
    "On Spectral Clustering: Analysis and an algorithm" (NeurIPS 2001)

Goal:
    Cluster the Flowers image embeddings using the original NJW idea:

        embeddings -> cosine affinity -> normalized affinity
        -> eigenvectors -> row normalization -> KMeans

The known Flowers labels are used only at the end to evaluate the clusters.
They are not used to create the clustering.
"""

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from Aux import (
    plot_first_two_embedding_coordinates,
    plot_graph_by_cluster,
    plot_graph_by_correctness,
)


ROOT_DIR = Path(__file__).resolve().parents[3]
DATASETS_DIR = ROOT_DIR / "DataSets"


def main():
    parser = argparse.ArgumentParser(description="Ng-Jordan-Weiss spectral clustering on Flowers.")
    parser.add_argument("--embeddings", type=Path, default=DATASETS_DIR / "Emb" / "Flowers_emb.pt")
    parser.add_argument("--num-clusters", type=int, default=17)
    parser.add_argument("--samples-per-class", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-neighbors", type=int, default=12)
    parser.add_argument("--layout-k", type=float, default=0.48)
    parser.add_argument("--layout-iterations", type=int, default=300)
    parser.add_argument("--layout-scale", type=float, default=2.4)
    parser.add_argument("--output-dir", type=Path, default=DATASETS_DIR / "Spectral_NJW")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1. Load the feature vectors
    # ---------------------------------------------------------------------
    print("\n[1] Loading Flowers embeddings")

    X = torch.load(args.embeddings, map_location="cpu")
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)

    X = X.float()
    n_samples = X.shape[0]

    print(f"X shape: {tuple(X.shape)}")

    # ---------------------------------------------------------------------
    # 2. Build the affinity matrix A with cosine similarity
    # ---------------------------------------------------------------------
    print("\n[2] Building cosine affinity matrix A")

    # Cosine similarity is just a dot product after L2-normalizing each row.
    X = torch.nn.functional.normalize(X, dim=1)
    A = X @ X.T

    # Affinity matrices represent positive connection strength.
    # Negative cosine values mean anti-similarity, so we remove those edges.
    A = torch.clamp(A, min=0.0)

    # A node should not vote for itself in the graph.
    A.fill_diagonal_(0.0)
    A = A.numpy()

    print(f"A shape: {A.shape}")

    # ---------------------------------------------------------------------
    # 3. Normalize the affinity matrix
    # ---------------------------------------------------------------------
    print("\n[3] Normalizing affinity matrix")

    # Degree of node i:
    #
    #     D_ii = sum_j A_ij
    #
    # Ng-Jordan-Weiss uses:
    #
    #     L = D^(-1/2) A D^(-1/2)
    #
    # This balances the graph so high-degree nodes do not dominate.
    degrees = A.sum(axis=1)
    degrees[degrees == 0] = 1.0

    D_inv_sqrt = 1.0 / np.sqrt(degrees)
    L = D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :]

    # ---------------------------------------------------------------------
    # 4. Take the top k eigenvectors
    # ---------------------------------------------------------------------
    print("\n[4] Computing top eigenvectors")

    # In the NJW version, we take the largest eigenvectors of the normalized
    # affinity matrix L. Each image becomes one row in this eigenvector matrix.
    eigenvalues, U = eigsh(L, k=args.num_clusters, which="LA")

    # Sort eigenvectors from largest to smallest eigenvalue for readability.
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    U = U[:, order]

    print(f"U shape: {U.shape}")

    # ---------------------------------------------------------------------
    # 5. Normalize each row of U
    # ---------------------------------------------------------------------
    print("\n[5] Row-normalizing eigenvectors")

    # Row i is the spectral coordinate of image i.
    # We normalize rows so KMeans clusters directions, not vector magnitudes.
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

    # Current Flowers layout:
    #     class 0 -> first 80 images
    #     class 1 -> next 80 images
    #     ...
    #     class 16 -> last 80 images
    expected = args.num_clusters * args.samples_per_class
    if n_samples != expected:
        raise ValueError(f"Expected {expected} Flowers samples, found {n_samples}.")

    labels = np.repeat(np.arange(args.num_clusters), args.samples_per_class)

    # KMeans cluster ids are arbitrary. Cluster 0 is not necessarily class 0.
    # We find the best cluster->class assignment using the Hungarian algorithm.
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
    print(f"Top eigenvalues: {[round(float(v), 6) for v in eigenvalues]}")
    print(f"Cluster mapping: {cluster_to_class}")

    # ---------------------------------------------------------------------
    # 8. Build a sparse graph for visualization
    # ---------------------------------------------------------------------
    print("\n[8] Building graph for plots")

    # The NJW algorithm clusters the rows of Y, not the original cosine matrix.
    # So the visualization graph should also be built in spectral space.
    #
    # We draw mutual nearest-neighbor links in Y:
    #     i -- j is drawn only when i is among j's closest spectral neighbors
    #     and j is among i's closest spectral neighbors.
    graph = nx.Graph()
    graph.add_nodes_from(range(n_samples))

    distances = pairwise_distances(Y)
    np.fill_diagonal(distances, np.inf)

    nearest = []
    for i in range(n_samples):
        closest = np.argsort(distances[i])[:args.plot_neighbors]
        nearest.append(set(int(j) for j in closest))

    for i in range(n_samples):
        for j in nearest[i]:
            if i in nearest[j]:
                # Stronger spring attraction for closer spectral neighbors.
                graph.add_edge(i, j, weight=float(1.0 / (1.0 + distances[i, j])))

    print(f"Plot graph edges: {graph.number_of_edges()}")

    # A random spring initialization can hide cluster structure.
    # PCA(Y) gives the layout a meaningful starting point from the spectral
    # embedding, and spring_layout then relaxes the graph visually.
    initial_xy = PCA(n_components=2, random_state=args.seed).fit_transform(Y)
    initial_xy = initial_xy / np.max(np.linalg.norm(initial_xy, axis=1))
    initial_positions = {
        i: initial_xy[i]
        for i in range(n_samples)
    }

    positions = nx.spring_layout(
        graph,
        k=args.layout_k,
        pos=initial_positions,
        iterations=args.layout_iterations,
        seed=args.seed,
        weight="weight",
        scale=args.layout_scale,
    )

    # ---------------------------------------------------------------------
    # 9. Save outputs
    # ---------------------------------------------------------------------
    print("\n[9] Saving outputs")

    np.save(args.output_dir / "clusters.npy", clusters)
    np.save(args.output_dir / "mapped_labels.npy", mapped_labels)
    np.save(args.output_dir / "spectral_points.npy", Y)
    np.save(args.output_dir / "eigenvalues.npy", eigenvalues)
    np.save(args.output_dir / "confusion.npy", confusion)

    plot_graph_by_cluster(
        graph,
        positions,
        clusters,
        args.output_dir / "clusters.png",
    )
    plot_graph_by_correctness(
        graph,
        positions,
        labels,
        mapped_labels,
        args.output_dir / "correctness.png",
    )
    plot_first_two_embedding_coordinates(
        Y,
        clusters,
        args.output_dir / "spectral_points.png",
    )

    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
