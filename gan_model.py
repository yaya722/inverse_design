import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 数据集类
class AbsorptionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = torch.softmax(attention, dim=-1)
        out = torch.matmul(attention, v)
        return out

# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.2),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x):
        return x + self.block(x)

# 生成器网络
class Generator(nn.Module):
    def __init__(self, input_size=601, hidden_size=512, output_size=5):
        super(Generator, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2)
        )
        
        self.attention = SelfAttention(hidden_size)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(4)
        ])
        
        self.feature_extraction = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size, hidden_size//2),
            nn.LayerNorm(hidden_size//2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.LayerNorm(hidden_size//4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size//4, output_size),
            nn.Tanh()
        )
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.input_layer(x)
        x = self.attention(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
            
        x = self.feature_extraction(x)
        x = self.output_layer(x)
        return x

# 判别器网络
class Discriminator(nn.Module):
    def __init__(self, input_size=601, param_size=5, hidden_size=512):
        super(Discriminator, self).__init__()
        
        self.absorption_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )
        
        self.attention = SelfAttention(hidden_size)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(2)
        ])
        
        self.param_net = nn.Sequential(
            nn.Linear(param_size, hidden_size//2),
            nn.LayerNorm(hidden_size//2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )
        
        self.main = nn.Sequential(
            nn.Linear(hidden_size + hidden_size//2, hidden_size//2),
            nn.LayerNorm(hidden_size//2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, absorption, params):
        x = self.absorption_net(absorption)
        x = self.attention(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
            
        param_features = self.param_net(params)
        combined = torch.cat((x, param_features), dim=1)
        return self.main(combined)

# 学习率预热调度器
def get_warmup_lr(initial_lr, epoch, warmup_epochs):
    if epoch >= warmup_epochs:
        return initial_lr
    return initial_lr * (epoch + 1) / warmup_epochs

# 评估函数
def evaluate(generator, data_loader, device, criterion_params):
    generator.eval()
    total_loss = 0
    all_real = []
    all_pred = []
    
    with torch.no_grad():
        for absorption, real_params in data_loader:
            absorption, real_params = absorption.to(device), real_params.to(device)
            fake_params = generator(absorption)
            loss = criterion_params(fake_params, real_params)
            total_loss += loss.item()
            
            # 收集预测和真实值用于计算MAE和R2
            all_real.extend(real_params.cpu().numpy())
            all_pred.extend(fake_params.cpu().numpy())
    
    all_real = np.array(all_real)
    all_pred = np.array(all_pred)
    mae = mean_absolute_error(all_real, all_pred)
    r2 = r2_score(all_real, all_pred)
    
    return total_loss / len(data_loader), mae, r2

# 训练函数
def train_gan(generator, discriminator, train_loader, val_loader, num_epochs, device,
              g_optimizer, d_optimizer, criterion_gan, criterion_params,
              g_scheduler, d_scheduler):
    
    best_val_loss = float('inf')
    initial_lr = g_optimizer.param_groups[0]['lr']
    warmup_epochs = 5
    
    for epoch in range(num_epochs):
        # 更新学习率
        if epoch < warmup_epochs:
            lr = get_warmup_lr(initial_lr, epoch, warmup_epochs)
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = lr
        
        # 训练阶段
        generator.train()
        discriminator.train()
        
        total_g_loss = 0
        total_d_loss = 0
        total_param_loss = 0
        all_real_train = []
        all_pred_train = []
        
        for absorption, real_params in train_loader:
            batch_size = absorption.size(0)
            absorption, real_params = absorption.to(device), real_params.to(device)
            
            # 训练判别器
            discriminator.zero_grad()
            
            # 添加标签平滑
            real_labels = torch.ones(batch_size, 1).to(device) * 0.9  # 标签平滑
            fake_labels = torch.zeros(batch_size, 1).to(device) + 0.1  # 标签平滑
            
            d_real_output = discriminator(absorption, real_params)
            d_real_loss = criterion_gan(d_real_output, real_labels)
            
            fake_params = generator(absorption)
            d_fake_output = discriminator(absorption, fake_params.detach())
            d_fake_loss = criterion_gan(d_fake_output, fake_labels)
            
            # 添加梯度惩罚
            alpha = torch.rand(batch_size, 1).to(device)
            interpolated_params = (alpha * real_params + (1 - alpha) * fake_params.detach()).requires_grad_(True)
            d_interpolated = discriminator(absorption, interpolated_params)
            
            gradients = torch.autograd.grad(
                outputs=d_interpolated,
                inputs=interpolated_params,
                grad_outputs=torch.ones_like(d_interpolated),
                create_graph=True,
                retain_graph=True
            )[0]
            
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            
            d_loss = (d_real_loss + d_fake_loss) / 2 + 10.0 * gradient_penalty
            d_loss.backward()
            d_optimizer.step()
            
            # 训练生成器
            generator.zero_grad()
            
            g_fake_output = discriminator(absorption, fake_params)
            g_gan_loss = criterion_gan(g_fake_output, real_labels)
            
            # 组合MSE和Huber损失
            mse_loss = criterion_params(fake_params, real_params)
            huber_loss = nn.SmoothL1Loss()(fake_params, real_params)
            param_loss = 0.5 * mse_loss + 0.5 * huber_loss
            
            g_loss = g_gan_loss + 10 * param_loss
            g_loss.backward()
            g_optimizer.step()
            
            # 收集训练指标
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            total_param_loss += param_loss.item()
            all_real_train.extend(real_params.cpu().numpy())
            all_pred_train.extend(fake_params.detach().cpu().numpy())
        
        # 计算训练集的MAE和R2
        train_mae = mean_absolute_error(np.array(all_real_train), np.array(all_pred_train))
        train_r2 = r2_score(np.array(all_real_train), np.array(all_pred_train))
        train_loss = total_param_loss/len(train_loader)
        
        # 验证阶段
        val_loss, val_mae, val_r2 = evaluate(generator, val_loader, device, criterion_params)
        
        # 获取当前学习率
        current_lr = g_optimizer.param_groups[0]['lr']
        
        # 打印训练信息
        print(f'Epoch [{epoch+1}/{num_epochs}] | '
              f'Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, R2: {train_r2:.4f} | '
              f'Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, R2: {val_r2:.4f} | '
              f'LR: {current_lr:.6f}')
        
        # 在预热期后使用学习率调度器
        if epoch >= warmup_epochs:
            g_scheduler.step(val_loss)
            d_scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, 'best_gan_model.pth')

if __name__ == '__main__':
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载数据
    data = loadmat('merged_data.mat')
    absorptions = np.array(data['absorptions']).squeeze().astype(np.float32)
    parameters = np.array(data['parameters']).squeeze().astype(np.float32)
    
    # 标准化数据
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    absorptions_scaled = scaler_X.fit_transform(absorptions)
    parameters_scaled = scaler_y.fit_transform(parameters)
    
    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        absorptions_scaled, parameters_scaled, test_size=0.2, random_state=42
    )
    
    # 创建数据加载器
    train_dataset = AbsorptionDataset(X_train, y_train)
    val_dataset = AbsorptionDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    # 初始化模型
    generator = Generator(hidden_size=512).to(device)
    discriminator = Discriminator(hidden_size=512).to(device)
    
    # 初始学习率和预热参数
    initial_lr = 0.001
    warmup_epochs = 5
    
    # 优化器
    g_optimizer = optim.AdamW(generator.parameters(), lr=initial_lr, weight_decay=0.01)
    d_optimizer = optim.AdamW(discriminator.parameters(), lr=initial_lr, weight_decay=0.01)
    
    # 学习率调度器
    g_scheduler = optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    d_scheduler = optim.lr_scheduler.ReduceLROnPlateau(d_optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # 损失函数
    criterion_gan = nn.BCELoss()
    criterion_params = nn.MSELoss()
    criterion_huber = nn.SmoothL1Loss()
    
    # 训练模型
    num_epochs = 200
    train_gan(generator, discriminator, train_loader, val_loader, num_epochs, device,
             g_optimizer, d_optimizer, criterion_gan, criterion_params,
             g_scheduler, d_scheduler)

    # 保存最终模型和数据处理器
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }, 'final_gan_model.pth')
