#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Means 聚类（硬件Embedding）
- 输入: Embedding/hardware_embeddings_v2.json
- 流程: 标准化 -> k∈[2..8] 评估(inertia/silhouette/CH/DB) -> 选 silhouette 最优 k
- 输出: kmeans_metrics.csv, kmeans_labels_k{best}.csv, pca_k{best}.png, umap_k{best}.png(可选)
重复可运行、可复用。
"""
import json
import csv
import sys
from pathlib import Path
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

# 可选 UMAP
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

ROOT = Path(__file__).parent
emb_file = ROOT / "hardware_embeddings_v2.json"
if not emb_file.exists():
    print(f"错误: 未找到 {emb_file}")
    sys.exit(1)

data = json.load(open(emb_file, "r", encoding="utf-8"))
names = [d["hardware_name"] for d in data]
X = np.array([d["vector"] for d in data], dtype=float)

# 标准化
Xs = StandardScaler().fit_transform(X)

# 多k评估
k_list = list(range(2, 9))
metrics = []
for k in k_list:
    km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(Xs)
    y = km.labels_
    try:
        sil = silhouette_score(Xs, y)
    except Exception:
        sil = float("nan")
    ch = calinski_harabasz_score(Xs, y)
    db = davies_bouldin_score(Xs, y)
    metrics.append({"k":k, "inertia":km.inertia_, "silhouette":sil, "calinski_harabasz":ch, "davies_bouldin":db})

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
km = KMeans(n_clusters=best_k, n_init=20, random_state=0).fit(Xs)
labels = km.labels_
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
P = PCA(n_components=2, random_state=0).fit_transform(Xs)
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
