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
import warnings
from tqdm import tqdm  # 用于进度条显示
import math
import torch.nn.functional as F

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

# 添加Mish激活函数
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# 3. 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            Mish(),
            nn.Dropout(dropout_rate)
        )
        
        # 第二个卷积层（与第一层保持一致）
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            Mish(),
            nn.Dropout(dropout_rate)
        )
        
        # 残差连接
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity(),
            nn.BatchNorm1d(out_channels) if in_channels != out_channels else nn.Identity()
        )
        
        # 最终激活函数
        self.final_act = Mish()

    def forward(self, x):
        # 主路径
        out = self.conv1(x)
        out = self.conv2(out)
        
        # 残差连接
        residual = self.residual(x)
        
        # 合并主路径和残差
        out = out + residual
        return self.final_act(out)

# 添加RoPE位置编码
class RoPE(nn.Module):
    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in
                            torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)

# 添加LinearAttention
class LinearAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(shape=(input_resolution, dim))

    def forward(self, x):
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k = qk[0], qk[1]

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, n, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, n, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = x.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, n, c).permute(0, 2, 1)
        x = x + self.lepe(v).permute(0, 2, 1).reshape(b, n, c)

        return x

# 自定义DropPath实现
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

# 4. 定义1D CNN模型
class CNN1D(nn.Module):
    def __init__(self, input_channels=1, output_size=5, num_filters=32, kernel_size=5, dropout_rate=0.2, n_features=601):
        super(CNN1D, self).__init__()
        
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if kernel_size >= n_features:
            raise ValueError("kernel_size must be smaller than n_features")
            
        self.final_length = n_features
        for _ in range(4):
            self.final_length = (self.final_length + 1) // 2
            
        if self.final_length <= 0:
            raise ValueError("Input size too small for the current architecture")
            
        # 初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_channels, num_filters, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters),
            Mish()
        )
        
        # 添加LinearAttention层
        self.attention1 = LinearAttention(num_filters, n_features, num_heads=4)
        
        # 主干网络
        self.stage1 = nn.Sequential(
            ResidualBlock(num_filters, num_filters, kernel_size, dropout_rate),
            nn.MaxPool1d(kernel_size=2),
            LinearAttention(num_filters, n_features // 2, num_heads=4)
        )
        
        self.stage2 = nn.Sequential(
            ResidualBlock(num_filters, num_filters * 2, kernel_size, dropout_rate),
            nn.MaxPool1d(kernel_size=2),
            LinearAttention(num_filters * 2, n_features // 4, num_heads=4)
        )
        
        self.stage3 = nn.Sequential(
            ResidualBlock(num_filters * 2, num_filters * 4, kernel_size, dropout_rate),
            nn.MaxPool1d(kernel_size=2),
            LinearAttention(num_filters * 4, n_features // 8, num_heads=4)
        )

        self.stage4 = nn.Sequential(
            ResidualBlock(num_filters * 4, num_filters * 8, kernel_size, dropout_rate),
            nn.MaxPool1d(kernel_size=2),
            LinearAttention(num_filters * 8, n_features // 16, num_heads=4)
        )

        # 全连接层
        final_length = n_features // 16
        self.fc = nn.Sequential(
            nn.Linear(num_filters * 8 * final_length, 512),
            nn.LayerNorm(512),
            Mish(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            Mish(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            Mish(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        # 初始卷积
        x = self.initial_conv(x)
        
        # 转换维度以适应LinearAttention
        x = x.permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
        x = self.attention1(x)
        x = x.permute(0, 2, 1)  # [B, L, C] -> [B, C, L]
        
        # 主干网络处理
        for stage in [self.stage1, self.stage2, self.stage3, self.stage4]:
            # ResidualBlock和MaxPool1d期望[B, C, L]格式
            x = stage[0:2](x)
            # LinearAttention期望[B, L, C]格式
            x = x.permute(0, 2, 1)
            x = stage[2](x)
            x = x.permute(0, 2, 1)
        
        # 展平并通过全连接层
        x = x.contiguous()  # 确保内存连续
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# 5. 定义训练和评估函数
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    all_preds = []
    all_targets = []
    dataset_size = len(dataloader.dataset)

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
        with torch.no_grad():
            mae = torch.mean(torch.abs(outputs - y_batch)).item()
            running_mae += mae * X_batch.size(0)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    epoch_loss = running_loss / dataset_size
    epoch_mae = running_mae / dataset_size
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    r2 = r2_score(all_targets, all_preds)

    return epoch_loss, epoch_mae, r2

def eval_epoch(model, dataloader, criterion, device, split='Validation'):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            running_loss += loss.item() * X_batch.size(0)
            running_mae += torch.mean(torch.abs(outputs - y_batch)).item() * X_batch.size(0)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_mae = running_mae / len(dataloader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    r2 = r2_score(all_targets, all_preds)

    return epoch_loss, epoch_mae, r2, all_preds, all_targets

# 设置随机种子
def seed_worker(worker_id):
    np.random.seed(42 + worker_id)
    torch.manual_seed(42 + worker_id)

# 定义带预热的学习率调度器
class WarmupReduceLROnPlateau:
    def __init__(self, optimizer, warmup_epochs=5, warmup_start_lr=1e-6, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = optimizer.param_groups[0]['lr']  # 预热后的目标学习率
        self.current_epoch = 0
        
        # 创建ReduceLROnPlateau调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            verbose=verbose,
            min_lr=min_lr
        )
    
    def step(self, metrics=None):
        if self.current_epoch < self.warmup_epochs:
            # 在预热期间线性增加学习率
            progress = self.current_epoch / self.warmup_epochs
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * progress
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            if self.scheduler.verbose:
                print(f'Warmup learning rate: {lr:.6f}')
        else:
            # 预热结束后使用ReduceLROnPlateau
            self.scheduler.step(metrics)
        
        self.current_epoch += 1
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

# 添加加权均方误差损失函数
class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.register_buffer('weights', weights)
    
    def forward(self, pred, target):
        return torch.mean(self.weights * (pred - target) ** 2)
    
    # 添加方法以更新权重
    def update_weights(self, new_weights):
        self.weights = new_weights

# 7. 主程序
if __name__ == '__main__':
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    mat_file_path = 'merged_data.mat'  # 请确保文件路径正确

    # 加载和预处理数据
    absorptions, parameters = load_and_preprocess_data(mat_file_path)
    if absorptions is None or parameters is None:
        print("Failed to load data")
        exit()

    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(absorptions)
    y_scaled = scaler_y.fit_transform(parameters)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    # 创建数据加载器
    train_dataset = AbsorptionDataset(X_train, y_train)
    test_dataset = AbsorptionDataset(X_test, y_test)

    # 固定超参数
    num_filters = 64
    kernel_size = 5
    dropout_rate = 0.07499385126240433
    learning_rate = 0.0009249609392561427
    weight_decay = 3.372482718194792e-04
    n_epochs = 209
    batch_size = 256

    g = torch.Generator()
    g.manual_seed(42)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        worker_init_fn=seed_worker,
        generator=g
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g
    )

    # 初始化模型、损失函数和优化器
    model = CNN1D(
        input_channels=1,
        output_size=5,
        num_filters=num_filters,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        n_features=absorptions.shape[1]
    ).to(device)


    weights = torch.tensor([0.5, 1.7, 1.7, 1.7, 0.5], dtype=torch.float32).to(device)
    criterion = WeightedMSELoss(weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = WarmupReduceLROnPlateau(
        optimizer,
        warmup_epochs=7,          # 预热期为5个epoch
        warmup_start_lr=1e-6,     # 预热起始学习率
        mode='min',               # 监控验证损失
        factor=0.5,              # 学习率降低为原来的一半
        patience=7,              # 等待5个epoch验证损失没有改善才降低学习率
        verbose=True,            # 打印学习率变化信息
        min_lr=1e-6             # 最小学习率
    )

    # 训练模型
    best_val_loss = float('inf')
    best_val_r2 = float('-inf')  # 添加最佳R²分数跟踪
    early_stopping_counter = 0
    early_stopping_patience = 30
    train_losses = []
    val_losses = []
    train_r2s = []
    val_r2s = []

    # 使用tqdm显示训练进度
    for epoch in range(n_epochs):
        # 在每个epoch开始前重新创建训练集的DataLoader
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(epoch)  # 使用epoch作为种子以确保每个epoch的打乱方式不同
        )
        
        # 训练阶段
        model.train()
        train_loss, train_mae, train_r2 = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_r2s.append(train_r2)
        
        # 验证阶段
        model.eval()
        val_loss, val_mae, val_r2, _, _ = eval_epoch(model, test_loader, criterion, device, split='Validation')
        val_losses.append(val_loss)
        val_r2s.append(val_r2)
        
        # 更新学习率
        scheduler.step(val_loss)
        current_lr = scheduler.get_lr()
        
        # 保存最佳模型（基于验证损失）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.scheduler.state_dict(),
                'val_loss': val_loss,
                'val_r2': val_r2,
            }, 'best_loss_model.pth')
        else:
            early_stopping_counter += 1
        
        # 保存R²最高的模型
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.scheduler.state_dict(),
                'val_loss': val_loss,
                'val_r2': val_r2,
            }, 'best_r2_model.pth')
        
        # 打印训练信息
        print(f'Epoch [{epoch+1}/{n_epochs}] | Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, R2: {train_r2:.4f} | Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, R2: {val_r2:.4f} | LR: {current_lr:.6f}')
        
        # 早停
        if early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break

    # 训练结束后，加载验证损失最小的模型进行测试
    print("\n加载并评估模型...")
    print("1. 基于验证损失的最佳模型：")
    checkpoint = torch.load('best_loss_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_mae, test_r2, test_preds, test_targets = eval_epoch(model, test_loader, criterion, device, split='Test')
    print(f"Test Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, R2: {test_r2:.4f}")

    # 加载R²最高的模型进行测试
    print("\n2. 基于R²分数的最佳模型：")
    checkpoint = torch.load('best_r2_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_mae, test_r2, test_preds, test_targets = eval_epoch(model, test_loader, criterion, device, split='Test')
    print(f"Test Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, R2: {test_r2:.4f}")

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
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curve.png', dpi=300)
    plt.show()
    plt.close()
