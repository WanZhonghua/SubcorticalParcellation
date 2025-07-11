import math
import numpy as np
import nibabel as nib
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity


def slic_like_clustering(D, coords, n_clusters=143, max_iter=10, S=5):
    print(f"S:{S}")
    # print(S)
    n_points = D.shape[0]
    centers = np.linspace(0, n_points - 1, n_clusters, dtype=int)
    labels = np.full(n_points, -1)
    distance_to_center = np.full(n_points, np.inf)

    for it in range(max_iter):
        print(f"迭代 {it + 1}/{max_iter}")
        for i, center_idx in enumerate(centers):
            center_coord = coords[center_idx]  # (x, y, z)
            dists_to_center = np.linalg.norm(coords - center_coord, axis=1)
            region = np.where(dists_to_center <= S)[0]

            dist = D[region, center_idx]
            better = dist < distance_to_center[region]
            distance_to_center[region[better]] = dist[better]
            labels[region[better]] = i

        for i in range(n_clusters):
            cluster_points = np.where(labels == i)[0]
            if len(cluster_points) > 0:
                mean_coord = np.mean(coords[cluster_points], axis=0)
                new_center = np.argmin(np.linalg.norm(coords - mean_coord, axis=1))
                centers[i] = new_center

    return labels


def precluster_and_reconstruct(num, max_iter=100, volume_shape=(182, 218, 182)):
    print(f"\n=== 处理编号: {num} ===")

    # 加载数据
    features = np.load(f'/data/wzh/ori/sum/171_{num}.npy') / 171
    coords = np.loadtxt(f'/home/wzh/MyCode/label_xyz/{num}.txt', dtype=int)

    voxel = features.shape[0]
    super_voxel = math.ceil(math.sqrt(voxel))  # 初始区域数
    S = (voxel // super_voxel)  # 近似等间隔邻域，作为搜索半径
    # S=1
    print(f"voxel={voxel}, super_voxel={super_voxel}, S={S}")

    # 计算距离矩阵
    d_c = 1 - cosine_similarity(features)
    d_g = cdist(coords, coords)
    D_c = np.max(d_c, axis=1, keepdims=True)
    D_g = np.max(d_g, axis=1, keepdims=True)
    D = np.sqrt((d_c / D_c) ** 2 + (d_g / D_g) ** 2)


    n_cluster=super_voxel*2
    # 聚类
    labels = slic_like_clustering(D, coords=coords, n_clusters=n_cluster, max_iter=max_iter, S=S)
    np.savetxt(f'precluster_{num}_{max_iter}_{S}_{n_cluster}.txt', labels, fmt='%d')

  
    volume = np.zeros(volume_shape, dtype=np.int16)
    for i in range(len(coords)):
        x, y, z = coords[i]
        volume[x, y, z] = labels[i] + 1  # 避免出现标签0表示背景

    nii = nib.Nifti1Image(volume, affine=np.eye(4))
    nib.save(nii, f'reconstrcut_{num}_{max_iter}_{S}_{n_cluster}.nii.gz')
    print(f"保存: reconstrcut_{num}_{max_iter}_{S}_{n_cluster}.nii.gz")



for num in range(9, 16):
    precluster_and_reconstruct(num)
