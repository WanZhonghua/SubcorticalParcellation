import numpy as np
from scipy.linalg import eigh as largest_eigh
import scipy.io as scio
from scipy import sparse
from sklearn import cluster
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph
import os
from tensorly.decomposition import tucker
import tensorly as tl
tl.set_backend('numpy')

def cal_knn_graph(distance, neighbor_num):
    # construct a knn graph
    neighbors_graph = kneighbors_graph(
        distance, neighbor_num, mode='connectivity', include_self=False)
    W = 0.5 * (neighbors_graph + neighbors_graph.T)
    return W

def prox_weight_tensor_nuclear_norm(Y, C):
    # calculate the weighted tensor nuclear norm
    # min_X ||X||_w* + 0.5||X - Y||_F^2
    n1, n2, n3 = np.shape(Y)
    X = np.zeros((n1, n2, n3), dtype=complex)
    # Y = np.fft.fft(Y, n3)
    Y = np.fft.fftn(Y)
    # Y = np.fft.fftn(Y, s=[n1, n2, n3])
    eps = 1e-6
    for i in range(n3):
        U, S, V = np.linalg.svd(Y[:, :, i], full_matrices=False)
        temp = np.power(S - eps, 2) - 4 * (C - eps * S)
        ind = np.where(temp > 0)
        ind = np.array(ind)
        r = np.max(ind.shape)
        if np.min(ind.shape) == 0:
            r = 0
        if r >= 1:
            temp2 = S[ind] - eps + np.sqrt(temp[ind])
            S = temp2.reshape(temp2.size, )
            X[:, :, i] = np.dot(np.dot(U[:, 0:r], np.diag(S)), V[:, 0:r].T)
    newX = np.fft.ifftn(X)
    # newX = np.fft.ifftn(X, s=[n1, n2, n3])
    # newX = np.fft.ifft(X, n3)

    return np.real(newX)


def consensus_graph_learning(A, cluster_num, lambda_1, rho, iteration_num):
    # optimize the consensus graph learning problem
    # min_H, Z 0.5||A - H'H||_F^2 + 0.5||Z - hatHhatH'||_F^2 + ||Z||_w*
    # s.t. H'H = I_k
    print(f'iteration_num:{iteration_num}')
    sample_num, sample_num, view_num = np.shape(A)
    print(f'sample_num:{sample_num}')
    print(f'view_num:{view_num}')
    # initial variables
    H = np.zeros((sample_num, cluster_num, view_num))
    HH = np.zeros((sample_num, sample_num, view_num))
    hatH = np.zeros((sample_num, cluster_num, view_num))
    hatHH = np.zeros((sample_num, sample_num, view_num))
    Q = np.zeros((sample_num, sample_num, view_num))
    Z = np.zeros((sample_num, sample_num, view_num))
    obj = np.zeros((iteration_num, 1))
    # loop
    for iter in range(iteration_num):
        # update H
        temp = np.zeros((sample_num, sample_num, view_num))
        G = np.zeros((sample_num, sample_num, view_num))
        for view in range(view_num):
            temp[:, :, view] = np.dot(np.dot(Q[:, :, view], 0.5 * (Z[:, :, view] + Z[:, :, view].T) - 0.5 * hatHH[:, :, view]), Q[:, :, view])
            G[:, :, view] = lambda_1 * A[:, :, view] + temp[:, :, view]
            _, H[:, :, view] = largest_eigh(G[:, :, view], subset_by_index=[sample_num - cluster_num, sample_num - 1])
            HH[:, :, view] = np.dot(H[:, :, view], H[:, :, view].T)
            # Q[:, :, view] = np.diag(1 / np.sqrt(np.diag(HH[:, :, view])))
            eps=1e-8
            Q[:, :, view] = np.diag(1 / np.sqrt(np.diag(HH[:, :, view]) + eps))
            hatH[:, :, view] = np.dot(Q[:, :, view], H[:, :, view])
            hatHH[:, :, view] = np.dot(hatH[:, :, view], hatH[:, :, view].T)
        # update Z
        hatHH2 = hatHH.transpose((0, 2, 1))
        Z2 = prox_weight_tensor_nuclear_norm(hatHH2, rho)
        Z = Z2.transpose((0, 2, 1))
        # update obj
        f = np.zeros((view_num, 1))
        for view in range(view_num):
            f[view] = 0.5 * lambda_1 * np.linalg.norm(A[:, :, view] - HH[:, :, view], ord='fro') + 0.5 * np.linalg.norm(Z[:, :, view] - hatHH[:, :, view], ord='fro')
        obj[iter] = np.sum(f)
        
        print(f"obj_{iter}:{obj[iter]}")

    save_dir_171=f'171_{lambda_1}_{rho}'
    os.makedirs(save_dir_171,exist_ok=True)
    
    for view in range(view_num):
        
        np.save(f'171_{lambda_1}_{rho}/{view+1}.npy',hatHH[:, :, view])
    
    
    # construct knn graph
    distance = np.zeros((sample_num, sample_num))
    for view in range(view_num):
        distance += hatHH[:, :, view]
    
    print(f"distance.shape:{distance.shape}")
    np.save(f'block_diag_lrt/distance_{lambda_1}_{rho}_{iteration_num}.npy', distance)
    
    W = cal_knn_graph(1 - distance, 15)
    
    # perform spectral clustering
    laplacian = sparse.csgraph.laplacian(W, normed=True)
    _, vec = sparse.linalg.eigsh(sparse.identity(laplacian.shape[0]) - laplacian, cluster_num, sigma=None, which='LA')
    
    embedding = normalize(vec)
    print(embedding.shape)
    np.save(f'block_diag_lrt/embedding_{lambda_1}_{rho}_{iteration_num}.npy', embedding)
    
    return W



save_dir='block_diag_lrt'
os.makedirs(save_dir,exist_ok=True)
view_num=171
sample_num=1230
# knn graph via sklearn.neighbors.kneighbors_graph
folder_path = '/home/wzh/MyCode/data/subcortical/precluster_block_diag_average'

# 获取所有 .npy 文件路径
file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')]

A = np.zeros((sample_num, sample_num, view_num))
for view in range(view_num):
    features = []
    features=np.load(file_paths[view])     
    knn_graph = cal_knn_graph(features, neighbor_num=15)
    print(f"knn_graph.shape:{knn_graph.shape}")
    S = sparse.identity(knn_graph.shape[0]) - sparse.csgraph.laplacian(knn_graph, normed=True).toarray()
    A[:, :, view] = S    
    
    
cluster_num =18 
iteration_num=100

parameter_lambda = [1,5, 10, 50, 100, 500, 1000, 5000]
parameter_rho = [1, 5, 10, 50, 100, 500, 1000, 5000]
for i in range(len(parameter_lambda)):
    for j in range(len(parameter_rho)):
        W = consensus_graph_learning(A, cluster_num, parameter_lambda[i], parameter_rho[j], iteration_num)
        np.save(f'block_diag_lrt/W_{parameter_lambda[i]}_{parameter_rho[j]}_{iteration_num}.npy', W)
        
            
            
