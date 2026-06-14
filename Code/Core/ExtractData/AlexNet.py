"""
AlexNet feature extraction pipeline for Flowers.

This follows the same project flow as Extract.py:

    images -> embeddings -> BallTree rankings -> graph JSON exports
"""

import json
import os
import time
from pathlib import Path

import natsort
import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import BallTree
from torchvision import models, transforms

from Graph import plot_and_export


BASE_DIR = Path(__file__).resolve().parent
CORE_DIR = BASE_DIR.parent
DATASETS_DIR = CORE_DIR.parent.parent / "DataSets"

DATASET_NAME = "Flowers"
MODEL_NAME = "AlexNet"
OUTPUT_NAME = f"{MODEL_NAME}_{DATASET_NAME}"


def alexnet_inference(image_paths):
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.AlexNet_Weights.DEFAULT
    model = models.alexnet(weights=weights)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    feature_model = torch.nn.Sequential(
        model.features,
        model.avgpool,
        torch.nn.Flatten(),
        *list(model.classifier.children())[:-1],
    )
    feature_model.to(device)
    feature_model.eval()

    batch_size = max(1, len(image_paths) // 10)
    all_features = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        input_tensors = []

        for img_path in batch_paths:
            img = Image.open(img_path).convert("RGB")
            input_tensors.append(transform(img))

        input_batch = torch.stack(input_tensors).to(device)

        with torch.no_grad():
            features = feature_model(input_batch)

        all_features.append(features.cpu().numpy())

        del input_batch, features
        torch.cuda.empty_cache()

        print(f"Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images")

    return np.vstack(all_features)


def run_ball_tree(features, k=100):
    """
    Constroi uma estrutura BallTree a partir das features e retorna rankings.
    """
    if not isinstance(features, np.ndarray):
        raise ValueError("As 'features' devem ser um array do tipo numpy.ndarray.")
    if features.ndim != 2:
        raise ValueError("As 'features' devem ser um array 2D no formato (n_samples, n_features).")

    tree = BallTree(features)
    k = min(k, len(features))
    _, rks = tree.query(features, k=k)

    return rks


def main():
    imgs_path = DATASETS_DIR / DATASET_NAME / "imgs"
    images = natsort.natsorted(os.listdir(imgs_path))

    image_paths = []

    ini_p = time.time()
    for i, img in enumerate(images):
        if ".jpg" not in img:
            continue

        image_paths.append(os.path.join(imgs_path, img))

        if i % 250 == 0:
            print(f"{i} images collected!")

    end_p = time.time()
    print(f"Collected {len(image_paths)} images in {end_p - ini_p:.2f}s")

    features = alexnet_inference(image_paths)

    emb_dir = DATASETS_DIR / "Emb"
    emb_dir.mkdir(exist_ok=True)
    emb_path = emb_dir / f"{OUTPUT_NAME}_emb.npy"
    emb_pt_path = emb_dir / f"{OUTPUT_NAME}_emb.pt"

    np.save(emb_path, features)
    torch.save(torch.from_numpy(features), emb_pt_path)
    print(f"Saved embeddings to {emb_path} and {emb_pt_path}")

    rks = run_ball_tree(features)

    runs_dir = DATASETS_DIR / "Runs"
    runs_dir.mkdir(exist_ok=True)
    ranking_path = runs_dir / f"{OUTPUT_NAME}_output.json"

    with open(ranking_path, "w") as json_file:
        json.dump(rks.tolist(), json_file, indent=4)
        json_file.flush()
        print(f"Data successfully exported to {ranking_path}")

    plots_dir = DATASETS_DIR / "Plots"
    plots_dir.mkdir(exist_ok=True)
    os.chdir(plots_dir)

    k_list = [10, 20, 30, 40, 60, 80]
    plot_and_export(rks, k_list, OUTPUT_NAME)


if __name__ == "__main__":
    main()
