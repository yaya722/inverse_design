import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.io import loadmat
import matplotlib.pyplot as plt

data = loadmat('merged_data.mat')

print("Variables in the .mat file:", data.keys())

absorptions = data['absorptions']      # 替换为实际的变量名
parameters = data['parameters']        # 替换为实际的变量名

# 3. Ensure they are NumPy arrays and have the correct shape
absorptions = np.array(absorptions)
parameters = np.array(parameters)

# 如果有多余的维度，可以使用 squeeze() 去除
absorptions = absorptions.squeeze()
parameters = parameters.squeeze()

print(f'Absorptions shape: {absorptions.shape}')  # 应为 (30000, 601)
print(f'Parameters shape: {parameters.shape}')    # 应为 (30000, 3)

# 确保数据类型为 float32
absorptions = absorptions.astype(np.float32)
parameters = parameters.astype(np.float32)

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    absorptions, parameters, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.125, random_state=42  # 0.125 x 0.8 = 0.1
)

# 5. Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 6. Create Dataset and DataLoader
class AbsorptionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = AbsorptionDataset(X_train, y_train)
val_dataset = AbsorptionDataset(X_val, y_val)
test_dataset = AbsorptionDataset(X_test, y_test)

batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 7. Define the Model
class FCNN(nn.Module):
    def __init__(self, input_size=601, output_size=3):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)
        
        
        
        self.fc4 = nn.Linear(128, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        
        x = torch.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x

model = FCNN(input_size=601, output_size=3)

# 8. Define Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 9. Train the Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
model.to(device)

num_epochs = 200
patience = 20
best_val_loss = np.inf
epochs_no_improve = 0

train_losses = []
val_losses = []
train_maes = []
val_maes = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X_batch.size(0)
        running_mae += torch.mean(torch.abs(outputs - y_batch)).item() * X_batch.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_mae = running_mae / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    train_maes.append(epoch_mae)
    
    model.eval()
    val_running_loss = 0.0
    val_running_mae = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            val_running_loss += loss.item() * X_batch.size(0)
            val_running_mae += torch.mean(torch.abs(outputs - y_batch)).item() * X_batch.size(0)
    
    val_epoch_loss = val_running_loss / len(val_loader.dataset)
    val_epoch_mae = val_running_mae / len(val_loader.dataset)
    val_losses.append(val_epoch_loss)
    val_maes.append(val_epoch_mae)
    
    print(f'Epoch {epoch+1}/{num_epochs} | '
          f'Train Loss: {epoch_loss:.4f} | Train MAE: {epoch_mae:.4f} | '
          f'Val Loss: {val_epoch_loss:.4f} | Val MAE: {val_epoch_mae:.4f}')
    
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print('Early stopping triggered!')
            break

# 10. Evaluate the Model
model.load_state_dict(torch.load('best_model.pth'))
model.to(device)
model.eval()

test_running_loss = 0.0
test_running_mae = 0.0
all_preds = []
all_targets = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        test_running_loss += loss.item() * X_batch.size(0)
        test_running_mae += torch.mean(torch.abs(outputs - y_batch)).item() * X_batch.size(0)
        
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())

test_loss = test_running_loss / len(test_loader.dataset)
test_mae = test_running_mae / len(test_loader.dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

y_pred = np.concatenate(all_preds, axis=0)
y_true = np.concatenate(all_targets, axis=0)

# 11. Visualize Training Process
plt.figure(figsize=(12, 5))

# Loss Curve
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# MAE Curve
plt.subplot(1, 2, 2)
plt.plot(train_maes, label='Train MAE')
plt.plot(val_maes, label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Training and Validation MAE')
plt.legend()

plt.tight_layout()
plt.show()

# 12. Prediction and Analysis
mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
print(f"Mean Squared Error for each parameter: {mse}")

# Visualize True vs Predicted for the first parameter
plt.figure(figsize=(6, 6))
plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.3)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Parameter 1: True vs Predicted')
plt.plot([y_true[:, 0].min(), y_true[:, 0].max()],
         [y_true[:, 0].min(), y_true[:, 0].max()], 'r--')
plt.grid(True)
plt.show()
