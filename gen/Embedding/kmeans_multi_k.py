#!/usr/bin/env python3
"""
批量生成 k=3/4/5/6 的 K-Means 聚类结果
导出标签文件和 PCA/UMAP 可视化图像
"""
import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")  # 非GUI环境
import matplotlib.pyplot as plt

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("警告: 未安装 umap-learn，将跳过 UMAP 可视化")

def main():
    # 设置路径
    root = Path("Embedding")
    json_file = root / "hardware_embeddings_v2.json"
    
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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 降维用于可视化
    print("计算 PCA...")
    pca = PCA(n_components=2, random_state=0)
    X_pca = pca.fit_transform(X_scaled)
    
    if HAS_UMAP:
        print("计算 UMAP...")
        umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0)
        X_umap = umap_reducer.fit_transform(X_scaled)
    
    # 批量生成 k=3,4,5,6 的结果
    k_list = [3, 4, 5, 6]
    
    for k in k_list:
        print(f"\n=== 生成 k={k} 聚类结果 ===")
        
        # K-Means 聚类
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
        labels = kmeans.fit_predict(X_scaled)
        
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
        plt.xlabel(f'PC1 (解释方差: {pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 (解释方差: {pca.explained_variance_ratio_[1]:.1%})')
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

