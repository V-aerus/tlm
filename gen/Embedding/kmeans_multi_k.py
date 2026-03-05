#!/usr/bin/env python3
"""
批量生成 k=3/4/5/6 的 K-Means 聚类结果
导出标签文件和 PCA/UMAP 可视化图像
默认读取 hardware_embeddings_v5_draft.json（可用 --emb 指定）
"""
import json
import numpy as np
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")  # 非GUI环境
import matplotlib.pyplot as plt

try:
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from sklearn.cluster import KMeans  # type: ignore
    from sklearn.decomposition import PCA  # type: ignore
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("警告: 未安装 umap-learn，将跳过 UMAP 可视化")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--emb",
        default=str(Path(__file__).parent / "hardware_embeddings_v5_draft.json"),
        help="硬件 embedding JSON 路径（list 格式）",
    )
    args = ap.parse_args()

    # 设置路径
    root = Path(__file__).parent
    json_file = Path(args.emb)
    if not json_file.is_absolute():
        json_file = root / json_file
    
    # 读取数据
    print("读取硬件嵌入数据...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 提取特征向量和硬件名称
    X = np.array([d["vector"] for d in data], dtype=float)
    names = [d["hardware_name"] for d in data]
    
    print(f"数据形状: {X.shape}")
    print(f"硬件数量: {len(names)}")
    
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
        return best_labels

    def _pca_numpy(x: np.ndarray, n_components: int = 2):
        xc = x - x.mean(axis=0)
        u, s, vt = np.linalg.svd(xc, full_matrices=False)
        comps = vt[:n_components]
        proj = xc @ comps.T
        var = (s ** 2) / max(x.shape[0] - 1, 1)
        var_ratio = var / var.sum() if var.sum() > 0 else var
        return proj, var_ratio

    X_scaled = StandardScaler().fit_transform(X) if HAS_SKLEARN else _standardize(X)
    
    # 降维用于可视化
    print("计算 PCA...")
    if HAS_SKLEARN:
        pca = PCA(n_components=2, random_state=0)
        X_pca = pca.fit_transform(X_scaled)
        pca_var = pca.explained_variance_ratio_
    else:
        X_pca, pca_var = _pca_numpy(X_scaled, n_components=2)
    
    if HAS_UMAP:
        print("计算 UMAP...")
        umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0)
        X_umap = umap_reducer.fit_transform(X_scaled)
    
    # 批量生成 k=3,4,5,6 的结果
    k_list = [3, 4, 5, 6]
    
    for k in k_list:
        print(f"\n=== 生成 k={k} 聚类结果 ===")
        
        # K-Means 聚类
        if HAS_SKLEARN:
            kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
            labels = kmeans.fit_predict(X_scaled)
        else:
            labels = _kmeans_numpy(X_scaled, k, n_init=10, max_iter=200, seed=0)
        
        # 保存标签文件
        labels_file = root / f"kmeans_labels_k{k}.csv"
        with open(labels_file, 'w', newline='') as f:
            f.write("hardware_name,cluster_label\n")
            for name, label in zip(names, labels):
                f.write(f"{name},{int(label)}\n")
        print(f"标签文件已保存: {labels_file}")
        
        # 生成 PCA 可视化
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=20, 
                            cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        plt.title(f'PCA + K-Means (k={k})')
        plt.xlabel(f'PC1 (解释方差: {pca_var[0]:.1%})')
        plt.ylabel(f'PC2 (解释方差: {pca_var[1]:.1%})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        pca_file = root / f"pca_k{k}.png"
        plt.savefig(pca_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"PCA 图已保存: {pca_file}")
        
        # 生成 UMAP 可视化（如果可用）
        if HAS_UMAP:
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, s=20, 
                                cmap='tab10', alpha=0.7)
            plt.colorbar(scatter)
            plt.title(f'UMAP + K-Means (k={k})')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            umap_file = root / f"umap_k{k}.png"
            plt.savefig(umap_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"UMAP 图已保存: {umap_file}")
        
        # 输出聚类统计信息
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"聚类分布: {dict(zip(unique_labels, counts))}")
    
    print(f"\n=== 批量生成完成 ===")
    print("生成的文件:")
    for k in k_list:
        print(f"  k={k}: kmeans_labels_k{k}.csv, pca_k{k}.png" + 
              (", umap_k{k}.png" if HAS_UMAP else ""))

if __name__ == "__main__":
    main()
