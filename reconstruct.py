import numpy as np
import pandas as pd
import os
import nibabel as nib
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

# 参数列表
lamda_list = [1, 5, 10, 50, 100, 500, 1000, 5000]
c_list = [1, 5, 10, 50, 100, 500, 1000, 5000]
file_path_2 = 'precluster_sum.txt'
file_path_3 = 'sum.txt'
output_dir = f'reconstruct'
os.makedirs(output_dir, exist_ok=True)
# 初始化结果矩阵
silhouette_scores = np.zeros((len(c_list), len(lamda_list)))

# 遍历参数组合并计算 silhouette score
for i, lamda_num in enumerate(lamda_list):
    for j, c in enumerate(c_list):
        try:
            # 加载矩阵
            matrix = np.load(f"block_diag_lrt/d/distance_{lamda_num}_{c}_100.npy")
            normalized_matrix = np.zeros_like(matrix)
            block_sizes = [286, 206, 222, 126, 200, 116, 74]

            # 归一化每个对角块
            start = 0
            for size in block_sizes:
                end = start + size
                block = matrix[start:end, start:end]
                min_val = block.min()
                max_val = block.max()
                if max_val > min_val:
                    norm_block = (block - min_val) / (max_val - min_val)
                else:
                    norm_block = np.zeros_like(block)
                normalized_matrix[start:end, start:end] = norm_block
                start = end

            # 聚类
            model = SpectralClustering(
                n_clusters=18,
                affinity='precomputed',
                assign_labels='discretize',
                random_state=42,
                n_neighbors=5,
                n_jobs=-1
            )
            labels = model.fit_predict(normalized_matrix)
            sil_score = silhouette_score(normalized_matrix, labels)
            silhouette_scores[j, i] = sil_score
            print(f"λ={lamda_num}, τ={c} → Silhouette Score = {sil_score:.4f}")

            volume_shape = (182, 218, 182)

            # 读取索引信息和坐标
            with open(file_path_2, 'r') as f2:
                indices_all = [int(line.strip()) for line in f2]
            with open(file_path_3, 'r') as f3:
                coords_all = [tuple(map(int, line.strip().split())) for line in f3]

            volume = np.zeros(volume_shape, dtype=np.int16)
            for k, (x, y, z) in enumerate(coords_all):
                label_idx = indices_all[k]
                if 0 <= label_idx < len(labels):
                    volume[x, y, z] = labels[label_idx] + 1
                else:
                    volume[x, y, z] = -1

            # 保存为 NIfTI 图像
            nii_path = os.path.join(output_dir, f'{lamda_num}_{c}.nii.gz')
            nifti_img = nib.Nifti1Image(volume, affine=np.eye(4))
            nib.save(nifti_img, nii_path)

            # 计算 silhouette score

        except Exception as e:
            print(f"Error with λ={lamda_num}, τ={c}: {e}")
            silhouette_scores[j, i] = np.nan

# 保存为 CSV
score_df = pd.DataFrame(silhouette_scores, index=c_list, columns=lamda_list)
score_df.index.name = "tau"
score_df.columns.name = "lambda"
csv_path = "silhouette_scores_real.csv"
score_df.to_csv(csv_path)
