"""
Final ablation runner.

This script keeps propagando_rotulos.py as the main pipeline and only changes
its fixed configuration variables before each run.
"""

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import propagando_rotulos as lap


ROOT_DIR = Path(__file__).resolve().parents[3]
DATASETS_DIR = ROOT_DIR / "DataSets"
RESULTS_DIR = DATASETS_DIR / "Laplacian" / "FinalAblation"
KMEANS_DIR = RESULTS_DIR / "KMeansOnly"

EXPERIMENTS = [
    {
        "name": "DINOv2 CUB50",
        "dataset_name": "CUB_Cleaned50",
        "output_name": "CUB_Cleaned50_dinov2_vits14",
        "embedding": DATASETS_DIR / "Emb" / "CUB_Cleaned50_emb.npy",
        "label_mode": "manifest",
        "num_clusters": 50,
        "samples_per_class": None,
        "laplacian_output_dir": DATASETS_DIR / "Laplacian",
    },
    {
        "name": "DINOv2 Flowers",
        "dataset_name": "Flowers",
        "output_name": "Flowers_dinov2_vits14",
        "embedding": DATASETS_DIR / "Emb" / "Flowers_emb.npy",
        "label_mode": "fixed",
        "num_clusters": 17,
        "samples_per_class": 80,
        "laplacian_output_dir": DATASETS_DIR / "Laplacian",
    },
    {
        "name": "AlexNet Flowers",
        "dataset_name": "Flowers",
        "output_name": "AlexNet_Flowers",
        "embedding": DATASETS_DIR / "Emb" / "AlexNet_Flowers_emb.npy",
        "label_mode": "fixed",
        "num_clusters": 17,
        "samples_per_class": 80,
        "laplacian_output_dir": DATASETS_DIR / "Laplacian" / "AlexNet",
    },
]


def labels_for_experiment(exp, n_samples):
    if exp["label_mode"] == "manifest":
        labels_path = DATASETS_DIR / exp["dataset_name"] / "manifest.csv"
        labels = []

        with open(labels_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(int(row["class_id"]) - 1)

        labels = np.array(labels, dtype=int)
    else:
        labels = np.repeat(
            np.arange(exp["num_clusters"]),
            exp["samples_per_class"],
        )

    if len(labels) != n_samples:
        raise ValueError(f"{exp['name']}: {len(labels)} labels for {n_samples} samples.")

    return labels


def evaluate_clusters(labels, clusters, num_clusters):
    confusion = np.zeros((num_clusters, num_clusters), dtype=int)

    for cluster_id in range(num_clusters):
        for class_id in range(num_clusters):
            confusion[cluster_id, class_id] = np.sum(
                (clusters == cluster_id) & (labels == class_id)
            )

    rows, cols = linear_sum_assignment(confusion.max() - confusion)
    cluster_to_class = {int(row): int(col) for row, col in zip(rows, cols)}
    mapped_labels = np.array([cluster_to_class[int(c)] for c in clusters])

    return {
        "matched_accuracy": float(np.mean(mapped_labels == labels)),
        "ari": float(adjusted_rand_score(labels, clusters)),
        "nmi": float(normalized_mutual_info_score(labels, clusters)),
        "confusion": confusion,
        "mapped_labels": mapped_labels,
        "cluster_to_class": cluster_to_class,
    }


def configure_laplacian(exp):
    lap.DATASET_NAME = exp["dataset_name"]
    lap.OUTPUT_NAME = exp["output_name"]
    lap.LABEL_MODE = exp["label_mode"]
    lap.NUM_CLUSTERS = exp["num_clusters"]
    lap.SAMPLES_PER_CLASS = exp["samples_per_class"]
    lap.OUTPUT_DIR = exp["laplacian_output_dir"]
    lap.RANKING_PATH = DATASETS_DIR / "Runs" / f"{exp['output_name']}_output.json"
    lap.LABELS_PATH = DATASETS_DIR / exp["dataset_name"] / "manifest.csv"


def laplacian_output_stem(exp):
    return f"{exp['output_name']}_laplacian_label_propagation_k{lap.K}_c{exp['num_clusters']}"


def run_laplacian_experiment(exp):
    configure_laplacian(exp)
    lap.main()

    stem = laplacian_output_stem(exp)
    clusters = np.load(exp["laplacian_output_dir"] / f"{stem}_clusters.npy")
    labels = labels_for_experiment(exp, len(clusters))
    metrics = evaluate_clusters(labels, clusters, exp["num_clusters"])

    return {
        "experiment": exp["name"],
        "method": "laplacian",
        "output_stem": stem,
        "accuracy": metrics["matched_accuracy"],
        "ari": metrics["ari"],
        "nmi": metrics["nmi"],
    }


def run_direct_kmeans(exp):
    X = np.load(exp["embedding"])
    labels = labels_for_experiment(exp, len(X))

    clusters = KMeans(
        n_clusters=exp["num_clusters"],
        n_init=50,
        random_state=lap.SEED,
    ).fit_predict(X)

    metrics = evaluate_clusters(labels, clusters, exp["num_clusters"])

    KMEANS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{exp['output_name']}_direct_kmeans_c{exp['num_clusters']}"

    np.save(KMEANS_DIR / f"{stem}_clusters.npy", clusters)
    np.save(KMEANS_DIR / f"{stem}_mapped_labels.npy", metrics["mapped_labels"])
    np.save(KMEANS_DIR / f"{stem}_confusion.npy", metrics["confusion"])

    return {
        "experiment": exp["name"],
        "method": "direct_kmeans",
        "output_stem": stem,
        "accuracy": metrics["matched_accuracy"],
        "ari": metrics["ari"],
        "nmi": metrics["nmi"],
    }


def save_results(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(rows, f, indent=2)

    with open(RESULTS_DIR / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["experiment", "method", "output_stem", "accuracy", "ari", "nmi"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = []

    for exp in EXPERIMENTS:
        print(f"\n=== Laplacian: {exp['name']} ===")
        rows.append(run_laplacian_experiment(exp))

        print(f"\n=== Direct KMeans: {exp['name']} ===")
        rows.append(run_direct_kmeans(exp))

    save_results(rows)

    print("\nSaved metrics:")
    for row in rows:
        print(
            f"{row['experiment']} | {row['method']} | "
            f"acc={row['accuracy']:.4f} ari={row['ari']:.4f} nmi={row['nmi']:.4f}"
        )


if __name__ == "__main__":
    main()
