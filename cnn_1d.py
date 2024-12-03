import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.io import loadmat
import matplotlib.pyplot as plt
import optuna
import warnings
from tqdm import tqdm  # 用于进度条显示

# 忽略一些警告信息
warnings.filterwarnings('ignore')

# 设置随机种子和确定性选项
def set_seed(seed=620):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# 1. 数据加载和预处理
def load_and_preprocess_data(mat_file_path):
    try:
        data = loadmat(mat_file_path)
        # 添加数据范围检查
        absorptions = np.array(data['absorptions']).squeeze().astype(np.float32)
        parameters = np.array(data['parameters']).squeeze().astype(np.float32)
        
        # 添加数据有效性检查
        if absorptions.size == 0 or parameters.size == 0:
            raise ValueError("Empty data arrays")
            
        # 添加数据一致性检查
        if len(absorptions) != len(parameters):
            raise ValueError("Mismatch in number of samples between absorptions and parameters")
            
        return absorptions, parameters
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

# 2. 定义数据集
class AbsorptionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # shape: (N, 1, n_features)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 3. 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.dropout2(self.bn2(self.conv2(out)))
        out += residual
        return self.act2(out)

# 4. 定义1D CNN模型
class CNN1D(nn.Module):
    def __init__(self, input_channels=1, output_size=5, num_filters=64, kernel_size=5, dropout_rate=0.1, n_features=601):
        super(CNN1D, self).__init__()
        
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if kernel_size >= n_features:
            raise ValueError("kernel_size must be smaller than n_features")
            
        # 计算最终特征长度
        self.final_length = n_features
        for _ in range(4):  # 减少到4次池化
            self.final_length = (self.final_length + 1) // 2
            
        if self.final_length <= 0:
            raise ValueError("Input size too small for the current architecture")
            
        # 简化网络结构，4组卷积层
        self.layers = nn.Sequential(
            # 第一层卷积组 (64)
            ResidualBlock(input_channels, num_filters, kernel_size, dropout_rate),
            nn.MaxPool1d(kernel_size=2),
            
            # 第二层卷积组 (128)
            ResidualBlock(num_filters, num_filters * 2, kernel_size, dropout_rate),
            nn.MaxPool1d(kernel_size=2),
            
            # 第三层卷积组 (256)
            ResidualBlock(num_filters * 2, num_filters * 4, kernel_size, dropout_rate),
            nn.MaxPool1d(kernel_size=2),
            
            # 第四层卷积组 (512)
            ResidualBlock(num_filters * 4, num_filters * 8, kernel_size, dropout_rate),
            nn.MaxPool1d(kernel_size=2)
        )

        # 调整全连接层结构
        final_length = n_features // 16  # 4次池化，每次池化kernel_size=2
        self.fc = nn.Sequential(
            nn.Linear(num_filters * 8 * final_length, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 5. 定义训练和评估函数
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    dataset_size = len(dataloader.dataset)

    for X_batch, y_batch in tqdm(dataloader, desc="Training", leave=False):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        # 使用item()减少GPU内存占用
        running_loss += loss.item() * X_batch.size(0)
        with torch.no_grad():
            mae = torch.mean(torch.abs(outputs - y_batch)).item()
        running_mae += mae * X_batch.size(0)
        
        # 清理不需要的张量
        del outputs, loss
        torch.cuda.empty_cache()

    return running_loss / dataset_size, running_mae / dataset_size

def eval_epoch(model, dataloader, criterion, device, split='Validation'):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(dataloader, desc=f"Evaluating {split}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            running_loss += loss.item() * X_batch.size(0)
            running_mae += torch.mean(torch.abs(outputs - y_batch)).item() * X_batch.size(0)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_mae = running_mae / len(dataloader.dataset)

    return epoch_loss, epoch_mae, np.concatenate(all_preds), np.concatenate(all_targets)

# 设置随机种子
def seed_worker(worker_id):
    np.random.seed(42 + worker_id)
    torch.manual_seed(42 + worker_id)

# 6. 定义Optuna的目标函数
def objective(trial):
    try:
        num_filters = trial.suggest_categorical('num_filters', [16, 32, 64, 128])
        kernel_size = trial.suggest_int('kernel_size', 3, 9, step=2)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        n_epochs = trial.suggest_int('n_epochs', 50, 250)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        val_losses = []

        for train_index, val_index in kf.split(X):
            X_train_cv, X_val_cv = X[train_index], X[val_index]
            y_train_cv, y_val_cv = y[train_index], y[val_index]

            scaler_X = StandardScaler()
            X_train_cv = scaler_X.fit_transform(X_train_cv)
            X_val_cv = scaler_X.transform(X_val_cv)

            scaler_y = StandardScaler()
            y_train_cv = scaler_y.fit_transform(y_train_cv)
            y_val_cv = scaler_y.transform(y_val_cv)

            train_loader = DataLoader(
                AbsorptionDataset(X_train_cv, y_train_cv),
                batch_size=batch_size,
                shuffle=True,
                worker_init_fn=seed_worker,
                num_workers=2  # 可以根据你的系统调整
            )
            val_loader = DataLoader(
                AbsorptionDataset(X_val_cv, y_val_cv),
                batch_size=batch_size,
                shuffle=False,
                worker_init_fn=seed_worker,
                num_workers=2
            )

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

            best_val_loss_cv = np.inf
            epochs_no_improve = 0
            patience = 30  # 增加早停耐心

            for epoch in range(n_epochs):
                train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_mae, _, _ = eval_epoch(model, val_loader, criterion, device)

                scheduler.step()

                if val_loss < best_val_loss_cv:
                    best_val_loss_cv = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        break

            val_losses.append(best_val_loss_cv)

        return np.mean(val_losses)
    except Exception as e:
        print(f"Trial failed due to {e}")
        return float('inf')

# 7. 主程序
if __name__ == '__main__':
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    mat_file_path = 'merged_data.mat'  # 请确保文件路径正确
    X, y = load_and_preprocess_data(mat_file_path)

    if X is None or y is None:
        raise RuntimeError("Failed to load data.")

    study = optuna.create_study(direction='minimize', study_name='CNN1D_Hyperparameter_Optimization')
    study.optimize(objective, n_trials=200, timeout=14400)  # 增加n_trials和timeout

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print(f"  Validation Loss: {trial.value}")
    print("  Best hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    best_params = {
        param: trial.params[param] 
        for param in ['num_filters', 'kernel_size', 'dropout_rate', 'learning_rate', 'weight_decay', 'n_epochs', 'batch_size']
    }

    # 分割数据为训练集和测试集
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = StandardScaler()
    X_train_full = scaler_X.fit_transform(X_train_full)
    X_test = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train_full)
    y_test = scaler_y.transform(y_test)

    train_loader = DataLoader(
        AbsorptionDataset(X_train_full, y_train),
        batch_size=best_params['batch_size'],
        shuffle=True,
        worker_init_fn=seed_worker,
        num_workers=2
    )
    test_loader = DataLoader(
        AbsorptionDataset(X_test, y_test),
        batch_size=best_params['batch_size'],
        shuffle=False,
        worker_init_fn=seed_worker,
        num_workers=2
    )

    # 分割训练集为训练部分和验证部分
    X_train_part, X_val, y_train_part, y_val = train_test_split(X_train_full, y_train, test_size=0.1, random_state=42)
    train_part_loader = DataLoader(
        AbsorptionDataset(X_train_part, y_train_part),
        batch_size=best_params['batch_size'],
        shuffle=True,
        worker_init_fn=seed_worker,
        num_workers=2
    )
    val_final_loader = DataLoader(
        AbsorptionDataset(X_val, y_val),
        batch_size=best_params['batch_size'],
        shuffle=False,
        worker_init_fn=seed_worker,
        num_workers=2
    )

    # 初始化模型
    model = CNN1D(
        input_channels=1,
        output_size=y.shape[1],
        num_filters=best_params['num_filters'],
        kernel_size=best_params['kernel_size'],
        dropout_rate=best_params['dropout_rate'],
        n_features=X_train_full.shape[1]
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True,
        min_lr=1e-6  # 添加最小学习率限制
    )

    # 在主训练循环之前初始化列表
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = 30
    min_delta = 1e-6
    train_losses = []  # 添加这行
    val_losses_history = []  # 添加这行

    for epoch in range(best_params['n_epochs']):
        train_loss, train_mae = train_epoch(model, train_part_loader, criterion, optimizer, device)
        val_loss, val_mae, _, _ = eval_epoch(model, val_final_loader, criterion, device)
        
        # 记录损失值
        train_losses.append(train_loss)  # 添加训练损失
        val_losses_history.append(val_loss)  # 添加验证损失
        
        print(f"\nEpoch {epoch+1}/{best_params['n_epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)
        
        # 改进的早停逻辑
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_losses': train_losses,  # 保存训练历史
                'val_losses': val_losses_history,  # 保存验证历史
            }, 'best_cnn_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f'Early stopping triggered at epoch {epoch}')
                break

    # 加载最佳模型
    model.load_state_dict(torch.load('best_cnn_model.pth')['model_state_dict'])
    model.eval()

    # 在测试集上评估
    test_loss, test_mae, test_preds, test_targets = eval_epoch(model, test_loader, criterion, device, split='Test')
    print(f"\nTest Loss: {test_loss:.4f} | Test MAE: {test_mae:.4f}")

    # 反标准化预测值和真实值
    y_pred = scaler_y.inverse_transform(test_preds)
    y_true = scaler_y.inverse_transform(test_targets)

    # 确保 y_true 和 y_pred 的形状相同
    assert y_true.shape == y_pred.shape, "y_true 和 y_pred 的形状不一致"

    # 计算并输出每个输出变量的 R² 分数
    print("\n各输出变量的 R² 分数：")
    for i in range(y_true.shape[1]):
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        print(f"输出 {i+1} 的 R² 分数: {r2:.4f}")

    # 计算总体 R² 分数
    overall_r2 = r2_score(y_true, y_pred)
    print(f"\n总体 R² 分数: {overall_r2:.4f}")

    # 绘制真实值与预测值
    num_outputs = y_true.shape[1]
    plt.figure(figsize=(18, 6 * num_outputs))  # 动态调整高度

    for i in range(num_outputs):
        plt.subplot(num_outputs, 1, i + 1)
        plt.plot(y_true[:, i], label=f'True Values - Output {i+1}', alpha=0.7)
        plt.plot(y_pred[:, i], label=f'Predicted Values - Output {i+1}', alpha=0.7)
        plt.xlabel('Samples')
        plt.ylabel('Value')
        plt.title(f'True vs Predicted Values for Output {i+1}')
        plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300)
    plt.show()

    # 绘制训练和验证损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curve.png', dpi=300)
    plt.show()
    plt.close()
