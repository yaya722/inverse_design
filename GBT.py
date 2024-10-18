import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.io import loadmat
import matplotlib.pyplot as plt
import warnings
import xgboost as xgb

# 忽略一些警告信息
warnings.filterwarnings('ignore')

# 1. 数据加载和预处理
def load_and_preprocess_data(mat_file_path):
    data = loadmat(mat_file_path)
    print("Variables in the .mat file:", data.keys())

    # 请根据实际变量名替换 'absorptions' 和 'parameters'
    absorptions = data['absorptions']
    parameters = data['parameters']

    absorptions = np.array(absorptions).squeeze().astype(np.float32)
    parameters = np.array(parameters).squeeze().astype(np.float32)

    print(f'Absorptions shape: {absorptions.shape}')  # 应为 (样本数, 特征数)
    print(f'Parameters shape: {parameters.shape}')    # 应为 (样本数, 参数数)

    # 检查是否存在 NaN 或无穷大值
    assert not np.isnan(absorptions).any(), "Absorptions contain NaN"
    assert not np.isnan(parameters).any(), "Parameters contain NaN"
    assert not np.isinf(absorptions).any(), "Absorptions contain Inf"
    assert not np.isinf(parameters).any(), "Parameters contain Inf"

    return absorptions, parameters

# 2. 主程序
if __name__ == '__main__':
    # 设置随机种子
    np.random.seed(42)

    # 加载数据
    mat_file_path = 'merged_data.mat'  # 请确保文件路径正确
    X, y = load_and_preprocess_data(mat_file_path)

    # 划分训练集和测试集
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 标准化
    scaler_X = StandardScaler()
    X_train_full = scaler_X.fit_transform(X_train_full)
    X_test = scaler_X.transform(X_test)

    # PCA 降维
    n_pca_components = 100  # 您可以根据需要调整组件数量
    pca = PCA(n_components=n_pca_components)
    X_train_pca = pca.fit_transform(X_train_full)
    X_test_pca = pca.transform(X_test)

    # 转换为 DataFrame，方便后续处理
    X_train_pca = pd.DataFrame(X_train_pca)
    X_test_pca = pd.DataFrame(X_test_pca)

    # 针对每个目标变量分别训练模型
    y_train_full = pd.DataFrame(y_train_full)
    y_test = pd.DataFrame(y_test)

    # 初始化评估指标
    mse_list = []
    mae_list = []
    r2_list = []

    # 可视化预测结果
    plt.figure(figsize=(15, 10))

    for i in range(y.shape[1]):
        print(f"\nTraining model for Parameter {i+1}")

        # 当前目标变量
        y_train = y_train_full.iloc[:, i]
        y_test_single = y_test.iloc[:, i]

        # 创建 XGBoost 回归器
        xgb_reg = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )

        # 使用 GridSearchCV 进行超参数调优
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [4, 6, 8],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        grid_search = GridSearchCV(
            estimator=xgb_reg,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            verbose=1,
            n_jobs=-1
        )

        grid_search.fit(X_train_pca, y_train)
        print(f"Best parameters for Parameter {i+1}: {grid_search.best_params_}")

        # 使用最佳参数训练模型
        best_xgb_reg = grid_search.best_estimator_
        best_xgb_reg.fit(X_train_pca, y_train)

        # 在测试集上预测
        y_pred = best_xgb_reg.predict(X_test_pca)

        # 评估模型
        mse = mean_squared_error(y_test_single, y_pred)
        mae = mean_absolute_error(y_test_single, y_pred)
        r2 = r2_score(y_test_single, y_pred)

        mse_list.append(mse)
        mae_list.append(mae)
        r2_list.append(r2)

        print(f"Parameter {i+1}: MSE = {mse:.4f}, MAE = {mae:.4f}, R² = {r2:.4f}")

        # 绘制真实值与预测值的对比图
        plt.subplot(2, 3, i+1)
        plt.scatter(y_test_single, y_pred, alpha=0.3)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Parameter {i+1}: True vs Predicted')
        min_val = min(y_test_single.min(), y_pred.min())
        max_val = max(y_test_single.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('xgboost_prediction_results.png', dpi=300)
    plt.show()

    # 汇总评估结果
    for i in range(y.shape[1]):
        print(f"Parameter {i+1}: MSE = {mse_list[i]:.4f}, MAE = {mae_list[i]:.4f}, R² = {r2_list[i]:.4f}")
