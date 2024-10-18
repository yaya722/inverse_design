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
from torch.utils.tensorboard import SummaryWriter
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
        print("Variables in the .mat file:", data.keys())

        absorptions = np.array(data['absorptions']).squeeze().astype(np.float32)
        parameters = np.array(data['parameters']).squeeze().astype(np.float32)

        print(f'Absorptions shape: {absorptions.shape}')  # 应为 (20000, 601)
        print(f'Parameters shape: {parameters.shape}')    # 应为 (20000, 5)

        # 检查是否存在NaN或无穷大值
        if np.any(np.isnan(absorptions)) or np.any(np.isnan(parameters)):
            raise ValueError("Data contains NaN values")
        if np.any(np.isinf(absorptions)) or np.any(np.isinf(parameters)):
            raise ValueError("Data contains infinite values")

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
        self.layers = nn.Sequential(
            ResidualBlock(input_channels, num_filters, kernel_size, dropout_rate),
            nn.MaxPool1d(kernel_size=2),
            ResidualBlock(num_filters, num_filters * 2, kernel_size, dropout_rate),
            nn.MaxPool1d(kernel_size=2),
            ResidualBlock(num_filters * 2, num_filters * 4, kernel_size, dropout_rate),
            nn.MaxPool1d(kernel_size=2),
            ResidualBlock(num_filters * 4, num_filters * 8, kernel_size, dropout_rate),
            nn.MaxPool1d(kernel_size=2)
        )

        final_length = n_features // 16  # 4次池化，每次池化kernel_size=2
        self.fc = nn.Sequential(
            nn.Linear(num_filters * 8 * final_length, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 5. 定义训练和评估函数
def train_epoch(model, dataloader, criterion, optimizer, device, writer, epoch):
    model.train()
    running_loss = 0.0
    running_mae = 0.0

    for X_batch, y_batch in tqdm(dataloader, desc="Training", leave=False):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
        running_mae += torch.mean(torch.abs(outputs - y_batch)).item() * X_batch.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_mae = running_mae / len(dataloader.dataset)

    if writer and epoch is not None:
        writer.add_scalar('Loss/Train', epoch_loss, epoch)
        writer.add_scalar('MAE/Train', epoch_mae, epoch)

    return epoch_loss, epoch_mae

def eval_epoch(model, dataloader, criterion, device, writer, epoch, split='Validation'):
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

    if writer and epoch is not None:
        writer.add_scalar(f'Loss/{split}', epoch_loss, epoch)
        writer.add_scalar(f'MAE/{split}', epoch_mae, epoch)

    return epoch_loss, epoch_mae, np.concatenate(all_preds), np.concatenate(all_targets)

# 设置随机种子
def seed_worker(worker_id):
    np.random.seed(42 + worker_id)
    torch.manual_seed(42 + worker_id)

# 6. 定义Optuna的目标函数
def objective(trial):
    try:
        num_filters = trial.suggest_categorical('num_filters', [32, 64, 128])
        kernel_size = trial.suggest_int('kernel_size', 3, 7, step=2)
        dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.3)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-4)
        n_epochs = trial.suggest_int('n_epochs', 100, 300)
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

            train_loader = DataLoader(AbsorptionDataset(X_train_cv, y_train_cv), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker)
            val_loader = DataLoader(AbsorptionDataset(X_val_cv, y_val_cv), batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker)

            model = CNN1D(input_channels=1, output_size=y.shape[1], num_filters=num_filters,
                           kernel_size=kernel_size, dropout_rate=dropout_rate, n_features=X_train_cv.shape[1]).to(device)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

            best_val_loss_cv = np.inf
            epochs_no_improve = 0
            patience = 30  # 增加早停耐心

            for epoch in range(n_epochs):
                train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device, None, None)
                val_loss, val_mae, _, _ = eval_epoch(model, val_loader, criterion, device, None, None)

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

    best_params = {param: trial.params[param] for param in ['num_filters', 'kernel_size', 'dropout_rate', 'learning_rate', 'weight_decay', 'n_epochs', 'batch_size']}

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = StandardScaler()
    X_train_full = scaler_X.fit_transform(X_train_full)
    X_test = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train_full)
    y_test = scaler_y.transform(y_test)

    train_loader = DataLoader(AbsorptionDataset(X_train_full, y_train), batch_size=best_params['batch_size'], shuffle=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(AbsorptionDataset(X_test, y_test), batch_size=best_params['batch_size'], shuffle=False, worker_init_fn=seed_worker)

    X_train_part, X_val, y_train_part, y_val = train_test_split(X_train_full, y_train, test_size=0.1, random_state=42)
    train_part_loader = DataLoader(AbsorptionDataset(X_train_part, y_train_part), batch_size=best_params['batch_size'], shuffle=True, worker_init_fn=seed_worker)
    val_final_loader = DataLoader(AbsorptionDataset(X_val, y_val), batch_size=best_params['batch_size'], shuffle=False, worker_init_fn=seed_worker)

    writer = SummaryWriter('runs/CNN1D_experiment')
    model = CNN1D(input_channels=1, output_size=y.shape[1], num_filters=best_params['num_filters'],
                   kernel_size=best_params['kernel_size'], dropout_rate=best_params['dropout_rate'],
                   n_features=X_train_full.shape[1]).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    best_val_loss = np.inf
    epochs_no_improve = 0

    for epoch in range(best_params['n_epochs']):
        print(f"\nEpoch {epoch+1}/{best_params['n_epochs']}")
        train_loss, train_mae = train_epoch(model, train_part_loader, criterion, optimizer, device, writer, epoch)
        val_loss, val_mae, _, _ = eval_epoch(model, val_final_loader, criterion, device, writer, epoch)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_cnn_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 30:  # 提前停止
                print('Early stopping triggered!')
                break

    model.load_state_dict(torch.load('best_cnn_model.pth'))
    model.eval()

    test_loss, test_mae, test_preds, test_targets = eval_epoch(model, test_loader, criterion, device, writer, epoch, split='Test')
    print(f"\nTest Loss: {test_loss:.4f} | Test MAE: {test_mae:.4f}")

    y_pred = scaler_y.inverse_transform(test_preds)
    y_true = scaler_y.inverse_transform(test_targets)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(val_final_loader.dataset.y.cpu().numpy(), label='True')
    plt.plot(test_preds, label='Predicted')
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.title('True vs Predicted Values')
    plt.legend()
    plt.savefig('training_metrics.png', dpi=300)
    plt.show()

    plt.close(writer)  # 关闭TensorBoard
