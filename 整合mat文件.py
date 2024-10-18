import scipy.io
import numpy as np
import os

# 存储所有文件路径
file_paths = [
    'results1728390658.mat',
    'results1728390888.mat',
    'results1728567327.mat',
    'results1728618384.mat',
]

# 用于存储合并后的数据
merged_absorptions = []
merged_parameters = []

# 遍历所有文件并合并数据
for file_path in file_paths:
    # 加载每个 .mat 文件
    mat_data = scipy.io.loadmat(file_path)
    
    # 提取 absorptions 和 parameters 并合并
    if 'absorptions' in mat_data and 'parameters' in mat_data:
        merged_absorptions.append(mat_data['absorptions'])
        merged_parameters.append(mat_data['parameters'])
    else:
        print(f"文件 {file_path} 中没有找到所需的数据")

# 将合并后的数据转换为 numpy 数组
merged_absorptions = np.vstack(merged_absorptions)
merged_parameters = np.vstack(merged_parameters)

# 保存到新的 .mat 文件
scipy.io.savemat('merged_data.mat', {
    'absorptions': merged_absorptions,
    'parameters': merged_parameters
})

print("数据已合并并保存到 merged_data.mat")
