#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Means 聚类（硬件Embedding）
- 输入: 默认 Embedding/hardware_embeddings_v5_draft.json（可用 --emb 指定）
- 流程: 标准化 -> k∈[2..8] 评估(inertia/silhouette/CH/DB) -> 选 silhouette 最优 k
- 输出: kmeans_metrics.csv, kmeans_labels_k{best}.csv, pca_k{best}.png, umap_k{best}.png(可选)
重复可运行、可复用。
"""
import json
import csv
import sys
import argparse
from pathlib import Path
import numpy as np

# sklearn 可选：没有则走 numpy 版 fallback
try:
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from sklearn.cluster import KMeans  # type: ignore
    from sklearn.metrics import (
        silhouette_score,  # type: ignore
        calinski_harabasz_score,  # type: ignore
        davies_bouldin_score,  # type: ignore
    )
    from sklearn.decomposition import PCA  # type: ignore
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# 可选 UMAP
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

ROOT = Path(__file__).parent

ap = argparse.ArgumentParser()
ap.add_argument(
    "--emb",
    default=str(ROOT / "hardware_embeddings_v5_draft.json"),
    help="硬件 embedding JSON 路径（list 格式）",
)
args = ap.parse_args()

emb_file = Path(args.emb)
if not emb_file.is_absolute():
    emb_file = ROOT / emb_file
if not emb_file.exists():
    print(f"错误: 未找到 {emb_file}")
    sys.exit(1)

data = json.load(open(emb_file, "r", encoding="utf-8"))
names = [d["hardware_name"] for d in data]
X = np.array([d["vector"] for d in data], dtype=float)

# 标准化
def _standardize(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std


def _kmeans_numpy(x: np.ndarray, k: int, n_init: int = 20, max_iter: int = 300, seed: int = 0):
    rng = np.random.RandomState(seed)
    best_inertia = float("inf")
    best_labels = None
    best_centroids = None
    n = x.shape[0]
    for _ in range(n_init):
        idx = rng.choice(n, k, replace=False)
        centroids = x[idx].copy()
        for _ in range(max_iter):
            dists = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            labels = dists.argmin(axis=1)
            new_centroids = []
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    new_centroids.append(x[mask].mean(axis=0))
                else:
                    new_centroids.append(centroids[j])
            new_centroids = np.stack(new_centroids, axis=0)
            if np.allclose(new_centroids, centroids):
                break
            centroids = new_centroids
        inertia = ((x - centroids[labels]) ** 2).sum()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids
    return best_labels, best_centroids, best_inertia


def _pca_numpy(x: np.ndarray, n_components: int = 2):
    xc = x - x.mean(axis=0)
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    comps = vt[:n_components]
    proj = xc @ comps.T
    var = (s ** 2) / max(x.shape[0] - 1, 1)
    var_ratio = var / var.sum() if var.sum() > 0 else var
    return proj, var_ratio


Xs = StandardScaler().fit_transform(X) if HAS_SKLEARN else _standardize(X)

# 多k评估
k_list = list(range(2, 9))
metrics = []
for k in k_list:
    if HAS_SKLEARN:
        km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(Xs)
        y = km.labels_
        try:
            sil = silhouette_score(Xs, y)
        except Exception:
            sil = float("nan")
        ch = calinski_harabasz_score(Xs, y)
        db = davies_bouldin_score(Xs, y)
        inertia = km.inertia_
    else:
        y, _, inertia = _kmeans_numpy(Xs, k, n_init=10, max_iter=200, seed=0)
        sil = float("nan")
        ch = float("nan")
        db = float("nan")
    metrics.append({"k": k, "inertia": inertia, "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db})

# 保存指标
metrics_csv = ROOT / "kmeans_metrics.csv"
with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["k","inertia","silhouette","calinski_harabasz","davies_bouldin"])
    w.writeheader()
    for m in metrics:
        w.writerow(m)
print(f"已保存: {metrics_csv}")

# 选最佳k（silhouette最大，若全NaN则按inertia最小）
valid = [m for m in metrics if not np.isnan(m["silhouette"]) ]
if valid:
    best = max(valid, key=lambda m: m["silhouette"]) 
else:
    best = min(metrics, key=lambda m: m["inertia"]) 

best_k = best["k"]
print(f"最佳k={best_k} (silhouette={best.get('silhouette')})")

# 重聚类并保存标签
if HAS_SKLEARN:
    km = KMeans(n_clusters=best_k, n_init=20, random_state=0).fit(Xs)
    labels = km.labels_
else:
    labels, _, _ = _kmeans_numpy(Xs, best_k, n_init=10, max_iter=200, seed=0)
labels_csv = ROOT / f"kmeans_labels_k{best_k}.csv"
with open(labels_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["hardware_name","cluster_label"])
    for n, lab in zip(names, labels):
        w.writerow([n, int(lab)])
print(f"已保存: {labels_csv}")

# PCA 可视化
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
if HAS_SKLEARN:
    P = PCA(n_components=2, random_state=0).fit_transform(Xs)
else:
    P, _ = _pca_numpy(Xs, n_components=2)
plt.figure(figsize=(6,5))
plt.scatter(P[:,0], P[:,1], c=labels, s=10, cmap="tab10")
plt.title(f"PCA + KMeans (k={best_k})")
plt.tight_layout()
_pca_path = ROOT / f"pca_k{best_k}.png"
plt.savefig(_pca_path, dpi=160)
plt.close()
print(f"已保存: {_pca_path}")

# UMAP 可视化（可选）
if HAS_UMAP:
    U = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(Xs)
    plt.figure(figsize=(6,5))
    plt.scatter(U[:,0], U[:,1], c=labels, s=10, cmap="tab10")
    plt.title(f"UMAP + KMeans (k={best_k})")
    plt.tight_layout()
    _umap_path = ROOT / f"umap_k{best_k}.png"
    plt.savefig(_umap_path, dpi=160)
    plt.close()
    print(f"已保存: {_umap_path}")
else:
    print("未安装 umap-learn，跳过 UMAP 图。")
