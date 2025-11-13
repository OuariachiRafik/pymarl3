import numpy as np

# 文件路径（和保存时保持一致）
save_path = "terran_causal_matrices.npy"

# 读取
causal_matrices = np.load(save_path, allow_pickle=True)

print("Causal Matrices Shape = ", causal_matrices.shape)
print("Total Matrices Saved = ", len(causal_matrices))

# 输出每一个 matrix
for i, mat in enumerate(causal_matrices):
    print(f"\n===== Matrix {i} =====")
    print(mat)
