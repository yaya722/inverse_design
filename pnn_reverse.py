import numpy as np
import scipy.io as io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import time
from torch.utils.tensorboard import SummaryWriter

# Optimized EarlyStopping class
class EarlyStopping:
    """Early Stopping implementation to prevent overfitting by monitoring validation loss."""

    def __init__(self, patience=10, verbose=False, delta=0, path='best_model.pth', trace_func=print):
        """
        Args:
            patience (int): How many epochs to wait after the last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path to save the best model.
            trace_func (callable): Function to call for logging (default: print).
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.best_model_state = model.state_dict()

    def load_best_model(self, model):
        """Loads the best model saved during training."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
        else:
            model.load_state_dict(torch.load(self.path))

# Data loading and preprocessing
def load_data_from_mat(mat_path):
    """Loads data from a .mat file."""
    try:
        data = io.loadmat(mat_path)
        print("Successfully loaded .mat file.")
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        raise e

    # Adjust variable names based on your .mat file structure
    # Assuming 'parameters' and 'absorptions' are the variable names
    if 'parameters' in data:
        parameters = data['parameters']
        print("'parameters' variable found.")
    else:
        raise KeyError("Cannot find 'parameters' variable in the .mat file.")

    if 'absorptions' in data:
        absorptions = data['absorptions']
        print("'absorptions' variable found.")
    else:
        raise KeyError("Cannot find 'absorptions' variable in the .mat file.")

    # Ensure data shapes are correct
    if parameters.shape[0] != absorptions.shape[0]:
        raise ValueError("Number of parameters does not match number of absorption spectra.")

    print(f"Parameters shape: {parameters.shape}, Absorptions shape: {absorptions.shape}")
    return absorptions, parameters  # Inputs are absorptions, outputs are parameters

# Define the MLP model
class MLPRegressor(nn.Module):
    def __init__(self, input_dim=601, output_dim=3):
        super(MLPRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# Define the loss function
criterion = nn.MSELoss()

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, device, num_epochs=1000):
    writer = SummaryWriter(log_dir='runs/absorption_to_parameter')
    train_losses = []
    val_losses = []
    start_time = time.time()

    print("Starting training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            try:
                inputs = inputs.to(device)
                targets = targets.to(device)
            except Exception as e:
                print(f"Error moving data to device: {e}")
                continue

            optimizer.zero_grad()
            try:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            except Exception as e:
                print(f"Error during forward pass or loss computation: {e}")
                continue
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                try:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                except Exception as e:
                    print(f"Error moving validation data to device: {e}")
                    continue

                try:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                except Exception as e:
                    print(f"Error during forward pass or loss computation: {e}")
                    continue

                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

        # Scheduler step
        scheduler.step(avg_val_loss)

        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Early Stopping check
        early_stopping(avg_val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    writer.close()

    # Load the best model
    try:
        early_stopping.load_best_model(model)
        print("Loaded best model weights.")
    except Exception as e:
        print(f"Error loading best model weights: {e}")

    return train_losses, val_losses

# Main function
def main(mat_path):
    # Load data from .mat file
    try:
        absorptions, parameters = load_data_from_mat(mat_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Data preprocessing
    print("Starting data preprocessing...")
    # Standardize absorptions (inputs)
    try:
        scaler_abs = StandardScaler()
        absorptions_scaled = scaler_abs.fit_transform(absorptions)
        print("Absorptions data standardized.")
    except Exception as e:
        print(f"Error during absorptions data standardization: {e}")
        return

    # Standardize parameters (outputs)
    try:
        scaler_params = StandardScaler()
        parameters_scaled = scaler_params.fit_transform(parameters)
        print("Parameters data standardized.")
    except Exception as e:
        print(f"Error during parameters data standardization: {e}")
        return

    # Split the dataset
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            absorptions_scaled, parameters_scaled, test_size=0.10, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42
        )
        print(f"Train size: {X_train.shape}, Validation size: {X_val.shape}, Test size: {X_test.shape}")
    except Exception as e:
        print(f"Error during dataset splitting: {e}")
        return

    # Convert to PyTorch tensors
    try:
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        print("Data converted to tensors.")
    except Exception as e:
        print(f"Error converting data to tensors: {e}")
        return

    # Create datasets and dataloaders
    try:
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        batch_size = 22500  # Reduce batch_size to lower memory usage
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Use batch_size=1 for testing
        print("Dataloaders created.")
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        return

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate the model
    try:
        model = MLPRegressor(input_dim=601, output_dim=3).to(device)
        print("Model instantiated.")
    except Exception as e:
        print(f"Error instantiating model: {e}")
        return

    # Check for multiple GPUs and use DataParallel if available
    if torch.cuda.device_count() > 1:
        try:
            print(f"Using {torch.cuda.device_count()} GPUs for parallel training.")
            model = nn.DataParallel(model)
        except Exception as e:
            print(f"Error using DataParallel: {e}")

    # Define optimizer
    try:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        print("Optimizer defined.")
    except Exception as e:
        print(f"Error defining optimizer: {e}")
        return

    # Define learning rate scheduler
    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        print("Learning rate scheduler defined.")
    except Exception as e:
        print(f"Error defining learning rate scheduler: {e}")
        return

    # Define EarlyStopping
    checkpoint_path = 'best_model.pth'
    early_stopping = EarlyStopping(patience=20, verbose=True, delta=1e-4, path=checkpoint_path)
    print("EarlyStopping instantiated.")

    # Train the model
    try:
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, device, num_epochs=1000
        )
    except Exception as e:
        print(f"Error during training: {e}")
        return

    # Save the final model
    try:
        # Define save directory
        save_dir = r'C:\Users\Neymar\Desktop\experiment\设计结构来保留光\神经网络加遗传算法优化\实验数据\predict'
        os.makedirs(save_dir, exist_ok=True)

        # Save the final model
        final_model_path = os.path.join(save_dir, 'final_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"Model saved to {final_model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

    # Plot training and validation loss
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(train_losses)), train_losses, color='r', linestyle='-', linewidth=2, label='Training Loss')
        plt.plot(range(len(val_losses)), val_losses, color='b', linestyle='-', linewidth=2, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Error plotting loss curves: {e}")

    # Test set predictions and result plotting
    def inverse_transform_predictions(preds, scaler):
        try:
            return scaler.inverse_transform(preds)
        except Exception as e:
            print(f"Error during inverse transformation: {e}")
            return preds  # Return original predictions

    try:
        # Randomly select 30 test samples
        num_predictions = 5
        test_size = X_test.shape[0]
        if test_size < num_predictions:
            num_predictions = test_size
            print(f"Not enough test samples, adjusting number of prediction samples to {num_predictions}")
        random_indices = np.random.choice(test_size, num_predictions, replace=False)

        model.eval()
        with torch.no_grad():
            for i, idx in enumerate(random_indices):
                # Prepare data
                input_abs = X_test[idx].unsqueeze(0).to(device)  # (1, 601)
                y_real = y_test[idx].cpu().numpy().reshape(1, -1)  # (1, 3)

                # Forward pass
                y_pred = model(input_abs)

                # Convert to CPU and numpy
                y_pred_np = y_pred.squeeze().cpu().numpy().reshape(1, -1)  # (1, 3)

                # Inverse transform
                y_real_orig = inverse_transform_predictions(y_real, scaler_params)
                y_pred_orig = inverse_transform_predictions(y_pred_np, scaler_params)

                # Plot
                plt.figure(figsize=(8, 6))
                parameters_names = ['Parameter 1', 'Parameter 2', 'Parameter 3']  # Adjust based on actual parameter names
                x = np.arange(len(parameters_names))
                width = 0.35

                plt.bar(x - width / 2, y_real_orig.flatten(), width, label='Real Value', color='red')
                plt.bar(x + width / 2, y_pred_orig.flatten(), width, label='Predicted Value', color='royalblue')

                plt.ylabel('Value')
                plt.title(f'Sample {i + 1} Real Value vs. Predicted Value')
                plt.xticks(x, parameters_names)
                plt.legend()
                plt.tight_layout()

                # Save the plot
                save_path = os.path.join(save_dir, f'prediction_sample_{i + 1}.png')
                try:
                    plt.savefig(save_path, dpi=300)
                    print(f"Saved prediction result to {save_path}")
                except Exception as e:
                    print(f"Error saving plot: {e}")
                plt.close()
    except Exception as e:
        print(f"Error during test set predictions or plotting: {e}")

    # Evaluate model performance on the test set
    def evaluate_model(model, test_loader, scaler_params, device):
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                try:
                    inputs = inputs.to(device)
                    outputs = model(inputs).cpu().numpy()
                    all_preds.append(outputs)
                    all_targets.append(targets.cpu().numpy())
                except Exception as e:
                    print(f"Error during prediction on test set: {e}")
                    continue

        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)

        # Inverse transform
        all_preds_orig = inverse_transform_predictions(all_preds, scaler_params)
        all_targets_orig = inverse_transform_predictions(all_targets, scaler_params)

        # Calculate evaluation metrics
        try:
            mse = mean_squared_error(all_targets_orig, all_preds_orig)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(all_targets_orig, all_preds_orig)
            r2 = r2_score(all_targets_orig, all_preds_orig)
            print(f"Test Set Evaluation Metrics:\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}")
        except Exception as e:
            print(f"Error calculating evaluation metrics: {e}")

        # Plot error distribution
        try:
            errors = all_preds_orig - all_targets_orig
            plt.figure(figsize=(12, 4))
            for i in range(errors.shape[1]):
                plt.subplot(1, errors.shape[1], i + 1)
                plt.hist(errors[:, i], bins=30, color='skyblue', edgecolor='black')
                plt.title(f'Parameter {i + 1} Error Distribution')
                plt.xlabel('Error')
                plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting error distribution: {e}")

        # Plot real vs predicted values
        try:
            plt.figure(figsize=(12, 4))
            num_params = all_targets_orig.shape[1]
            for i in range(num_params):
                plt.subplot(1, num_params, i + 1)
                plt.scatter(all_targets_orig[:, i], all_preds_orig[:, i], alpha=0.5, color='b', edgecolors='k')
                plt.plot([all_targets_orig[:, i].min(), all_targets_orig[:, i].max()],
                         [all_targets_orig[:, i].min(), all_targets_orig[:, i].max()],
                         'r--')
                plt.xlabel('Real Value')
                plt.ylabel('Predicted Value')
                plt.title(f'Parameter {i + 1}')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting real vs predicted scatter plots: {e}")

    try:
        evaluate_model(model, test_loader, scaler_params, device)
    except Exception as e:
        print(f"Error during model evaluation: {e}")

if __name__ == "__main__":
    # Specify the path to the .mat file
    mat_file_path = r'C:\Users\Neymar\Desktop\experiment\设计结构来保留光\神经网络加遗传算法优化\实验数据\merged_data.mat'  # Replace with actual path

    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(f"The specified .mat file does not exist: {mat_file_path}")

    main(mat_file_path)
