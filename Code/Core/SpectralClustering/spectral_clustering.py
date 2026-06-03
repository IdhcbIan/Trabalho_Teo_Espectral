"""
Spectral clustering on the DINOv2 KNN graph.

This script implements the classical spectral clustering pipeline:

    1. Load a nearest-neighbor ranking.
    2. Build a weighted KNN graph.
    3. Compute the normalized graph Laplacian.
    4. Use the smallest Laplacian eigenvectors as a new representation.
    5. Run KMeans in that spectral representation.
    6. Evaluate the clusters against known labels.

Labels are not used to form the clusters. They are used only at the end,
to measure how well the unsupervised clusters recover the known classes.
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


# ---------------------------------------------------------------------------
# Project defaults
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[3]
DATASETS_DIR = ROOT_DIR / "DataSets"

DEFAULT_RANKING_PATH = DATASETS_DIR / "Runs" / "dinov2_vits14_output.json"
DEFAULT_OUTPUT_DIR = DATASETS_DIR / "Spectral"


# ---------------------------------------------------------------------------
# 0. Data loading and graph construction
# ---------------------------------------------------------------------------

def load_rankings(path):
    """
    Load the nearest-neighbor ranking matrix.

    The expected file is a JSON list of lists. Row i contains the ranked
    neighbor indices for image i. The first element is usually i itself.
    """
    with open(path, "r") as f:
        rankings = np.array(json.load(f), dtype=int)

    if rankings.ndim != 2:
        raise ValueError("Ranking JSON must be a 2D list with shape (n_samples, n_neighbors).")

    return rankings


def load_labels(path, num_samples, num_classes, samples_per_class):
    """
    Load labels for evaluation.

    If no label file is provided, we use the Oxford-17 Flowers convention:
    17 classes, 80 images per class, ordered by filename.
    """
    if path is not None:
        labels = np.loadtxt(path, dtype=int)
        if labels.ndim == 2:
            labels = labels[:, -1]
        if len(labels) != num_samples:
            raise ValueError(f"Label file has {len(labels)} labels, but rankings have {num_samples} samples.")
        return labels

    expected_samples = num_classes * samples_per_class
    if num_samples != expected_samples:
        raise ValueError(
            "No label file was provided and the default Oxford-17 assumption does not match: "
            f"{num_samples} samples != {num_classes} classes * {samples_per_class} samples."
        )

    return np.repeat(np.arange(num_classes), samples_per_class)


def build_knn_adjacency(rankings, k):
    """
    Build a weighted, symmetric KNN adjacency matrix from rankings.

    The closer a neighbor appears in the ranking, the stronger its edge:

        rank 1 -> weight 1.0
        rank 2 -> weight 0.5
        rank r -> weight 1/r
    """
    num_samples, num_neighbors = rankings.shape
    if k >= num_neighbors:
        raise ValueError(f"k={k} is too large for ranking file with {num_neighbors} neighbors per sample.")

    adjacency = np.zeros((num_samples, num_samples), dtype=float)

    for source in range(num_samples):
        neighbors = rankings[source, 1 : k + 1]
        for rank, target in enumerate(neighbors, start=1):
            if source == target:
                continue
            weight = 1.0 / rank
            adjacency[source, target] = max(adjacency[source, target], weight)

    return np.maximum(adjacency, adjacency.T)


def graph_from_adjacency(adjacency):
    """
    Convert the adjacency matrix into a NetworkX graph.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(adjacency.shape[0]))

    rows, cols = np.nonzero(np.triu(adjacency, k=1))
    for source, target in zip(rows, cols):
        graph.add_edge(
            int(source),
            int(target),
            weight=float(adjacency[source, target]),
        )

    return graph


# ---------------------------------------------------------------------------
# 1. Spectral embedding
# ---------------------------------------------------------------------------

def compute_normalized_laplacian(adjacency):
    """
    Compute the normalized graph Laplacian.

    If W is the weighted adjacency matrix and D is the degree matrix, the
    normalized Laplacian is:

        L_norm = I - D^(-1/2) W D^(-1/2)

    This normalization makes the method less sensitive to degree variation.
    """
    return csgraph.laplacian(adjacency, normed=True)


def compute_smallest_eigenvectors(laplacian, num_vectors):
    """
    Compute the eigenvectors associated with the smallest eigenvalues.

    In spectral clustering, these low-frequency eigenvectors reveal the broad
    connected structure of the graph. Nodes in the same natural group tend to
    receive similar coordinates in this eigenvector space.
    """
    eigenvalues, eigenvectors = eigsh(laplacian, k=num_vectors, which="SM")

    # eigsh does not guarantee sorted output. Sorting makes the result easier
    # to inspect and keeps plots/exports stable.
    order = np.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


def normalize_rows(matrix):
    """
    Normalize each row to unit length.

    This is the standard Ng-Jordan-Weiss step. After normalization, KMeans
    clusters directions in spectral space instead of being dominated by row
    magnitude.
    """
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    return matrix / row_norms


def build_spectral_embedding(adjacency, num_clusters):
    """
    Convert a graph adjacency matrix into a spectral embedding.

    Returns:
        eigenvalues: the selected Laplacian eigenvalues.
        embedding: one row per image, used as input to KMeans.
    """
    laplacian = compute_normalized_laplacian(adjacency)
    eigenvalues, eigenvectors = compute_smallest_eigenvectors(laplacian, num_clusters)
    embedding = normalize_rows(eigenvectors)
    return eigenvalues, embedding


# ---------------------------------------------------------------------------
# 2. Clustering
# ---------------------------------------------------------------------------

def cluster_embedding_with_kmeans(embedding, num_clusters, seed):
    """
    Run KMeans on the spectral embedding.

    KMeans is applied to the Laplacian eigenvector representation, not to the
    original DINOv2 feature vectors.
    """
    kmeans = KMeans(
        n_clusters=num_clusters,
        n_init=50,
        random_state=seed,
    )
    return kmeans.fit_predict(embedding)


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------

def build_cluster_label_confusion(true_labels, clusters):
    """
    Build a matrix where rows are discovered clusters and columns are labels.

    Entry (i, j) counts how many samples from cluster i belong to true class j.
    """
    label_values = np.unique(true_labels)
    cluster_values = np.unique(clusters)

    confusion = np.zeros((len(cluster_values), len(label_values)), dtype=int)
    cluster_to_row = {cluster: row for row, cluster in enumerate(cluster_values)}
    label_to_col = {label: col for col, label in enumerate(label_values)}

    for label, cluster in zip(true_labels, clusters):
        row = cluster_to_row[cluster]
        col = label_to_col[label]
        confusion[row, col] += 1

    return confusion, cluster_values, label_values


def match_clusters_to_labels(true_labels, clusters):
    """
    Match arbitrary KMeans cluster IDs to class IDs.

    KMeans cluster numbers have no semantic meaning: cluster 0 is not
    necessarily class 0. For accuracy, we choose the one-to-one mapping that
    maximizes agreement with true labels. This is solved with the Hungarian
    algorithm.
    """
    confusion, cluster_values, label_values = build_cluster_label_confusion(
        true_labels,
        clusters,
    )

    # linear_sum_assignment minimizes cost. A large confusion count is good,
    # so we convert counts to costs.
    cost_matrix = confusion.max() - confusion
    rows, cols = linear_sum_assignment(cost_matrix)

    cluster_to_label = {
        cluster_values[row]: label_values[col]
        for row, col in zip(rows, cols)
    }
    mapped_labels = np.array([cluster_to_label[cluster] for cluster in clusters])

    return mapped_labels, cluster_to_label, confusion


def compute_clustering_metrics(true_labels, clusters, mapped_labels):
    """
    Compute evaluation metrics for the clustering result.
    """
    return {
        "accuracy": np.mean(mapped_labels == true_labels),
        "ari": adjusted_rand_score(true_labels, clusters),
        "nmi": normalized_mutual_info_score(true_labels, clusters),
    }


def print_report(true_labels, clusters, mapped_labels, eigenvalues, cluster_to_label):
    """
    Print a compact, presentation-friendly clustering report.
    """
    metrics = compute_clustering_metrics(true_labels, clusters, mapped_labels)

    print("\n=== Spectral Clustering Report ===")
    print(f"Clusters: {len(np.unique(clusters))}")
    print(f"Accuracy after optimal cluster-to-label mapping: {metrics['accuracy']:.4f}")
    print(f"Adjusted Rand Index: {metrics['ari']:.4f}")
    print(f"Normalized Mutual Information: {metrics['nmi']:.4f}")
    print(f"Eigenvalues: {[round(float(value), 6) for value in eigenvalues]}")
    print(
        "Cluster mapping:",
        {int(cluster): int(label) for cluster, label in cluster_to_label.items()},
    )

    print("\nPer-class accuracy after mapping:")
    for label in np.unique(true_labels):
        class_mask = true_labels == label
        class_accuracy = np.mean(mapped_labels[class_mask] == true_labels[class_mask])
        print(f"  Class {int(label):02d}: {class_accuracy:.4f} ({class_mask.sum()} samples)")


# ---------------------------------------------------------------------------
# 4. Visualization and export
# ---------------------------------------------------------------------------

def compute_graph_layout(graph, args):
    """
    Compute 2D graph coordinates for plots and JSON export.
    """
    return nx.spring_layout(
        graph,
        k=args.layout_k,
        iterations=args.layout_iterations,
        seed=args.seed,
        weight="weight",
        scale=args.layout_scale,
    )


def export_graph_json(graph, positions, true_labels, clusters, mapped_labels, output_path, k, eigenvalues):
    """
    Export the graph in the same JSON style used by the project visualizer.
    """
    export_data = {
        "nodes": {},
        "edges": {},
        "graph_info": {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "k": k,
            "method": "spectral_clustering",
            "eigenvalues": [float(value) for value in eigenvalues],
        },
    }

    for node_id in graph.nodes:
        export_data["nodes"][str(node_id)] = {
            "position": [
                float(positions[node_id][0]),
                float(positions[node_id][1]),
            ],
            "attributes": {
                "true_label": int(true_labels[node_id]),
                "cluster": int(clusters[node_id]),
                "mapped_label": int(mapped_labels[node_id]),
                "correct": bool(true_labels[node_id] == mapped_labels[node_id]),
            },
        }

    for node_id in graph.nodes:
        export_data["edges"][str(node_id)] = [
            int(neighbor)
            for neighbor in graph.neighbors(node_id)
        ]

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)


def save_numpy_outputs(output_dir, output_stem, clusters, mapped_labels, embedding, eigenvalues, confusion):
    """
    Save reusable NumPy artifacts for later analysis.
    """
    np.save(output_dir / f"{output_stem}_clusters.npy", clusters)
    np.save(output_dir / f"{output_stem}_mapped_labels.npy", mapped_labels)
    np.save(output_dir / f"{output_stem}_embedding.npy", embedding)
    np.save(output_dir / f"{output_stem}_eigenvalues.npy", eigenvalues)
    np.save(output_dir / f"{output_stem}_confusion.npy", confusion)


def save_visual_outputs(output_dir, output_stem, graph, positions, true_labels, clusters, mapped_labels, embedding):
    """
    Save the three visual outputs generated by the script.
    """
    plot_graph_by_cluster(
        graph,
        positions,
        clusters,
        output_dir / f"{output_stem}.png",
    )
    plot_graph_by_correctness(
        graph,
        positions,
        true_labels,
        mapped_labels,
        output_dir / f"{output_stem}_correctness.png",
    )
    plot_first_two_embedding_coordinates(
        embedding,
        clusters,
        output_dir / f"{output_stem}_embedding.png",
    )


def print_saved_files(output_dir, output_stem):
    """
    Print the generated output paths.
    """
    print("\nSaved files:")
    print(f"  Clusters: {output_dir / f'{output_stem}_clusters.npy'}")
    print(f"  Mapped labels: {output_dir / f'{output_stem}_mapped_labels.npy'}")
    print(f"  Embedding: {output_dir / f'{output_stem}_embedding.npy'}")
    print(f"  Eigenvalues: {output_dir / f'{output_stem}_eigenvalues.npy'}")
    print(f"  Confusion matrix: {output_dir / f'{output_stem}_confusion.npy'}")
    print(f"  Graph JSON: {output_dir / f'{output_stem}.json'}")
    print(f"  Cluster plot: {output_dir / f'{output_stem}.png'}")
    print(f"  Correctness plot: {output_dir / f'{output_stem}_correctness.png'}")
    print(f"  Embedding plot: {output_dir / f'{output_stem}_embedding.png'}")


# ---------------------------------------------------------------------------
# 5. Command-line interface
# ---------------------------------------------------------------------------

def parse_args():
    """
    Define all experiment parameters exposed from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Run true spectral clustering: normalized Laplacian eigenvectors plus KMeans."
    )
    parser.add_argument("--ranking", type=Path, default=DEFAULT_RANKING_PATH)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--num-classes", type=int, default=17)
    parser.add_argument("--samples-per-class", type=int, default=80)
    parser.add_argument("--num-clusters", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layout-k", type=float, default=0.34)
    parser.add_argument("--layout-iterations", type=int, default=180)
    parser.add_argument("--layout-scale", type=float, default=1.35)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 6. Main experiment
# ---------------------------------------------------------------------------

def main():
    """
    Run the full spectral clustering experiment.
    """
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading data ===")
    rankings = load_rankings(args.ranking)
    true_labels = load_labels(
        args.labels,
        len(rankings),
        args.num_classes,
        args.samples_per_class,
    )
    print(f"Ranking matrix: {rankings.shape}")
    print(f"Number of labels: {len(true_labels)}")

    print("\n=== Building KNN graph ===")
    adjacency = build_knn_adjacency(rankings, args.k)
    graph = graph_from_adjacency(adjacency)
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    print("\n=== Computing spectral embedding ===")
    eigenvalues, embedding = build_spectral_embedding(
        adjacency,
        args.num_clusters,
    )
    print(f"Spectral embedding shape: {embedding.shape}")

    print("\n=== Running KMeans ===")
    clusters = cluster_embedding_with_kmeans(
        embedding,
        args.num_clusters,
        args.seed,
    )

    print("\n=== Matching clusters to labels for evaluation ===")
    mapped_labels, cluster_to_label, confusion = match_clusters_to_labels(
        true_labels,
        clusters,
    )

    print("\n=== Computing graph layout ===")
    positions = compute_graph_layout(graph, args)

    output_stem = f"spectral_clustering_k{args.k}_c{args.num_clusters}"

    print("\n=== Saving outputs ===")
    save_numpy_outputs(
        args.output_dir,
        output_stem,
        clusters,
        mapped_labels,
        embedding,
        eigenvalues,
        confusion,
    )
    export_graph_json(
        graph,
        positions,
        true_labels,
        clusters,
        mapped_labels,
        args.output_dir / f"{output_stem}.json",
        args.k,
        eigenvalues,
    )
    save_visual_outputs(
        args.output_dir,
        output_stem,
        graph,
        positions,
        true_labels,
        clusters,
        mapped_labels,
        embedding,
    )

    print_report(
        true_labels,
        clusters,
        mapped_labels,
        eigenvalues,
        cluster_to_label,
    )
    print_saved_files(args.output_dir, output_stem)


if __name__ == "__main__":
    main()
