import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from cnn_1d import CNN1D, AbsorptionDataset  # 导入模型和数据集类

def load_model_and_predict(model_path, input_data):
    """
    加载预训练模型并进行预测
    
    参数:
        model_path: 保存的模型文件路径(.pth文件)
        input_data: 输入数据(光谱数据), shape应为(n_samples, n_features)
    
    返回:
        预测结果
    """
    # 检查是否有GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载保存的模型和参数
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取保存的参数
    best_params = checkpoint['best_params']
    scaler_X = checkpoint['scaler_X']
    scaler_y = checkpoint['scaler_y']
    
    # 初始化模型
    model = CNN1D(
        input_channels=1,
        output_size=5,  # 假设输出维度为5
        num_filters=best_params['num_filters'],
        kernel_size=best_params['kernel_size'],
        dropout_rate=best_params['dropout_rate'],
        n_features=input_data.shape[1]
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设置为评估模式
    
    # 数据预处理
    X_scaled = scaler_X.transform(input_data)
    
    # 创建数据加载器
    dataset = AbsorptionDataset(X_scaled, np.zeros((len(X_scaled), 5)))  # 创建虚拟标签
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 进行预测
    predictions = []
    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.append(outputs.cpu().numpy())
    
    # 合并预测结果
    predictions = np.vstack(predictions)
    
    # 将预测结果转换回原始尺度
    predictions = scaler_y.inverse_transform(predictions)
    
    return predictions

# 使用示例
if __name__ == "__main__":
    # 假设我们有新的光谱数据需要预测
    from scipy.io import loadmat
    
    # 加载新数据
    test_data = loadmat('your_test_data.mat')  # 替换为您的测试数据文件
    test_spectra = test_data['absorptions'].squeeze()  # 假设数据结构与训练数据相同
    
    # 加载模型并预测
    model_path = 'best_ensemble_model.pth'  # 您保存的模型文件路径
    predictions = load_model_and_predict(model_path, test_spectra)
    
    # 打印预测结果
    print("预测结果形状:", predictions.shape)
    print("\n前5个样本的预测结果:")
    for i in range(min(5, len(predictions))):
        print(f"样本 {i+1}: {predictions[i]}") 