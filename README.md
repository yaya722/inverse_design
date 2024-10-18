# 1D Convolutional Neural Network for Predicting Material Properties

## Overview
This project implements a 1D Convolutional Neural Network (CNN) to predict material properties from spectroscopic data. The model is designed to learn the relationship between absorption spectra and corresponding parameters, using data from a `.mat` file that contains spectroscopic information.

The code leverages Optuna for hyperparameter optimization, PyTorch for model training, and scikit-learn for data preprocessing. The entire workflow includes data loading, model definition, hyperparameter tuning, training, validation, and testing.

## Requirements
- Python 3.7+
- PyTorch
- scikit-learn
- Optuna
- Matplotlib
- TensorBoard
- tqdm
- scipy

You can install the necessary packages using the following command:

```sh
pip install numpy torch scikit-learn optuna matplotlib tensorboard tqdm scipy
```

## Usage

### 1. Data Preparation
Ensure that your dataset is in the form of a `.mat` file with at least two key variables: `absorptions` and `parameters`. These should be arrays containing the absorption spectra and their respective parameters for training the model. The script reads this `.mat` file and extracts the variables.

### 2. Hyperparameter Optimization
The script uses Optuna to find the best hyperparameters for the CNN model. The parameters include the number of filters, kernel size, dropout rate, learning rate, weight decay, and others. You can modify the number of trials or the timeout duration to control how long the optimization runs.

### 3. Training the Model
After finding the best hyperparameters, the script trains the final CNN model on the full training dataset, while validating it on a small subset of the data. It uses early stopping to prevent overfitting.

The model architecture includes several residual blocks to improve training efficiency and extract more useful features from the spectroscopic data.

### 4. Evaluation and Results
The script evaluates the trained model on a held-out test set, and outputs performance metrics like loss and mean absolute error (MAE). The results are plotted and saved as a PNG image (`training_metrics.png`). You can also visualize training metrics using TensorBoard.

### 5. Running the Code
To run the training and optimization pipeline, execute the script:

```sh
python cnn_1d.py
```

Make sure to replace the `mat_file_path` variable in the script with the path to your `.mat` file.

## Model Architecture
- **Residual Blocks:** The CNN model is built with residual blocks that improve training by enabling better gradient flow. Each residual block consists of convolutional layers followed by batch normalization, ReLU activation, and dropout.
- **Fully Connected Layer:** The final features extracted by the convolutional layers are passed through fully connected layers to predict the material parameters.

## Hyperparameter Optimization
The script uses Optuna for hyperparameter tuning, optimizing parameters such as:
- Number of filters
- Kernel size
- Dropout rate
- Learning rate
- Batch size
- Number of epochs

## Evaluation Metrics
- **Mean Squared Error (MSE)**: Used as the loss function for training.
- **Mean Absolute Error (MAE)**: Used as an evaluation metric during validation and testing.

## Output
- **Best Hyperparameters**: The script prints the optimal hyperparameters found by Optuna.
- **Trained Model**: The model is saved as `best_cnn_model.pth`.
- **Performance Plot**: A plot showing true vs. predicted values is saved as `training_metrics.png`.

## Notes
- Ensure that the `.mat` file is properly formatted and contains no NaN or infinite values, as these will cause issues during data loading.
- The code utilizes GPU if available, significantly speeding up the training process.

## Acknowledgments
This code was developed using PyTorch, Optuna, and other open-source libraries. Special thanks to the contributors of these projects for providing tools that make deep learning projects accessible and efficient.

## License
This project is licensed under the MIT License.

