```markdown:path/to/README.md
# 项目名称

## 目录
1. [简介](#简介)
2. [安装与运行](#安装与运行)
3. [主要功能](#主要功能)
    - [数据处理与加载](#1-数据处理与加载)
    - [模型架构](#2-模型架构)
    - [训练特性](#3-训练特性)
    - [超参数优化](#4-超参数优化)
    - [模型集成](#5-模型集成)
4. [评估与可视化](#评估与可视化)
5. [常见问题](#常见问题)
6. [贡献](#贡献)
7. [许可证](#许可证)

---

## 简介

本项目旨在通过1D卷积神经网络（CNN）和梯度提升树（GBT）模型对光谱数据进行分析与预测。项目涵盖数据加载与预处理、模型训练与评估、超参数优化，以及模型集成等多个环节。通过使用Optuna进行高效的超参数搜索，结合5折交叉验证和模型集成技术，提升模型的泛化能力和预测性能。

## 安装与运行

### 环境依赖

请确保已安装以下依赖库：

- Python 3.8+
- NumPy
- PyTorch
- scikit-learn
- SciPy
- Matplotlib
- Optuna
- tqdm
- XGBoost

可以使用以下命令安装所有依赖：

```bash
pip install -r requirements.txt
```

### 运行步骤

1. **数据合并**

    ```bash
    python 整合mat文件.py
    ```

    该脚本将多个 `.mat` 文件中的吸收光谱数据和参数数据合并，并保存为 `merged_data.mat`。

2. **训练1D CNN模型**

    ```bash
    python cnn_1d.py
    ```

    该脚本执行以下主要流程：
    - 数据加载与预处理
    - 超参数优化（使用Optuna）
    - 5折交叉验证训练模型
    - 模型集成
    - 评估与可视化

3. **训练梯度提升树（GBT）模型**

    ```bash
    python GBT.py
    ```

    使用XGBoost进行模型训练与评估。

4. **查看训练过程**

    使用TensorBoard查看训练日志和损失曲线：

    ```bash
    tensorboard --logdir=runs
    ```

    然后在浏览器中打开 [http://localhost:6006](http://localhost:6006) 查看实时训练情况。

## 主要功能

### 1. 数据处理与加载

- **`整合mat文件.py`**：合并多个 `.mat` 数据文件，生成统一的数据集。
- **`utils.py`**：包含数据加载、标准化、数据集划分和数据加载器创建等工具函数。

### 2. 模型架构

- **1D CNN模型** (`CNN1D` 类)：
  - 由多个残差块 (`ResidualBlock` 类) 组成，每个残差块包含SE注意力层 (`SELayer` 类)。
  - 通过堆叠卷积层和池化层提取特征，最后通过全连接层输出预测结果。
  
- **梯度提升树模型** (`GBT.py`):
  - 使用XGBoost进行训练，并通过网格搜索进行超参数调优。

### 3. 训练特性

- **交叉验证**：使用5折交叉验证提升模型的泛化能力。
- **早停机制**：防止过拟合，根据验证集损失自动停止训练。
- **学习率调度**：采用余弦退火学习率调度器，并结合预热策略。
- **混合精度训练**：利用半精度浮点数加速训练过程。
- **模型集成**：集成多个fold的最佳模型，进一步提高预测性能。

### 4. 超参数优化

使用Optuna进行高效的超参数搜索，包括：

- 过滤器数量 (`num_filters`)
- 卷积核大小 (`kernel_size`)
- Dropout比例 (`dropout_rate`)
- 学习率 (`learning_rate`)
- 权重衰减 (`weight_decay`)
- 训练轮数 (`n_epochs`)
- 批次大小 (`batch_size`)

**示例代码块：**

```python:cnn_1d.py
# 定义Optuna的目标函数
def objective(trial, X, y):
    """
    Optuna超参数优化的目标函数
    
    参数:
        trial: Optuna试验对象
        X: 输入特征
        y: 目标变量
    
    返回:
        float: 验证集上的平均损失
    """
    try:
        # 定义超参数搜索空间
        num_filters = trial.suggest_categorical('num_filters', [32, 64, 128])
        kernel_size = trial.suggest_int('kernel_size', 3, 7, step=2)
        dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.3)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-4)
        n_epochs = trial.suggest_int('n_epochs', 100, 300)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])

        # 使5折交叉验证评估当前超参数组合
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        val_losses = []

        # 对每个fold进行训练和评估
        for train_index, val_index in kf.split(X):
            # 数据划分和预处理
            X_train_cv, X_val_cv = X[train_index], X[val_index]
            y_train_cv, y_val_cv = y[train_index], y[val_index]

            # 标准化特征和目标变量
            scaler_X = StandardScaler()
            X_train_cv = scaler_X.fit_transform(X_train_cv)
            X_val_cv = scaler_X.transform(X_val_cv)

            scaler_y = StandardScaler()
            y_train_cv = scaler_y.fit_transform(y_train_cv)
            y_val_cv = scaler_y.transform(y_val_cv)

            # 创建数据加载器
            train_loader = DataLoader(
                AbsorptionDataset(X_train_cv, y_train_cv), 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=4, 
                pin_memory=True,
                worker_init_fn=seed_worker
            )
            val_loader = DataLoader(
                AbsorptionDataset(X_val_cv, y_val_cv), 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=4, 
                pin_memory=True,
                worker_init_fn=seed_worker
            )

            # 初始化模型、优化器和学习率调度器
            model = CNN1D(
                input_channels=1, 
                output_size=y.shape[1], 
                num_filters=num_filters,
                kernel_size=kernel_size, 
                dropout_rate=dropout_rate, 
                n_features=X_train_cv.shape[1]
            ).to(device)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

            # 训练循环
            best_val_loss_cv = np.inf
            epochs_no_improve = 0
            patience = 30  # 早停耐心值
            
            scaler_cv = GradScaler()  # 用于混合精度训练

            for epoch in range(n_epochs):
                # 训练和评估一个epoch
                train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device, scaler_cv)
                val_loss, val_mae, _, _ = eval_epoch(model, val_loader, criterion, device, epoch)

                scheduler.step()

                # 早停检查
                if val_loss < best_val_loss_cv:
                    best_val_loss_cv = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        break

            val_losses.append(best_val_loss_cv)

        # 返回所有fold的平均验证损失
        return np.mean(val_losses)
```

### 5. 模型集成

通过加载每个fold的最佳模型，构建集成模型，并对测试集进行最终评估。

**示例代码块：**

```python:cnn_1d.py
# 构建和评估集成模型
models = []
for i in range(5):
    # 为每个fold创建一个新的模型实例
    model_i = CNN1D(
        input_channels=1, 
        output_size=y.shape[1],
        num_filters=best_params['num_filters'],
        kernel_size=best_params['kernel_size'],
        dropout_rate=best_params['dropout_rate'],
        n_features=X_train_full.shape[1]
    ).to(device)
    
    # 加载每个fold的最佳模型权重
    model_path = f'models/best_model_fold_{i}.pth'
    if os.path.exists(model_path):
        model_i.load_state_dict(torch.load(model_path))
        model_i.eval()  # 设置为评估模式
        models.append(model_i)
    else:
        print(f"模型文件 {model_path} 不存在。")

# 创建集成模型
ensemble_model = EnsembleModel(models).to(device)

# 保存集成模型和相关参数
torch.save({
    'model_state_dict': ensemble_model.state_dict(),
    'best_params': best_params,
    'scaler_X': scaler_X,  # 保存特征缩放器
    'scaler_y': scaler_y   # 保存标签缩放器
}, 'best_ensemble_model.pth')
```

## 评估与可视化

在测试集上评估集成模型的性能，并生成可视化图表以展示预测结果与真实值的对比，以及训练过程中的损失曲线。

**评估指标包括：**
- 均方误差（MSE）
- 平均绝对误差（MAE）
- R² 分数

**可视化部分：**

- **预测值与真实值对比图**
- **训练与验证损失曲线**

生成的图表将保存为 `training_results.png`。

## 常见问题

### 模型训练不收敛

请确保：
- 数据预处理正确，且没有缺失值或异常值。
- 超参数选择合理，可以尝试调整学习率或增加训练轮数。
- 检查硬件资源是否充足，尤其是GPU显存。

### 数据加载错误

确保 `.mat` 文件路径正确，并且文件中包含 `absorptions` 和 `parameters` 变量。

## 贡献

欢迎提交 Issues 或 Pull Requests 来贡献代码和改进建议！

## 许可证

本项目采用 [MIT 许可证](LICENSE)。
```