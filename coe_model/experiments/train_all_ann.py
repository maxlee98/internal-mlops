import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from datetime import datetime

# ML Model Creation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow


# Set random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

# Initialize MLFlow tracking
mlflow.set_tracking_uri("http://127.0.0.1:8080/")  # Replace with your MLFlow server URI
mlflow.set_experiment("COE_Prediction-ANN")


# Load data
current_dir = os.path.dirname(os.path.realpath(__file__))
data_fldr = os.path.join(current_dir, "..", "data")

coe_df = pd.read_excel(os.path.join(data_fldr, "COE_Export.xlsx"), sheet_name="Yearly")
cpi_df = pd.read_excel(os.path.join(data_fldr, "ConsumerPriceIndex.xlsx"), sheet_name="Consolidate")
ni_df = pd.read_excel(os.path.join(data_fldr, "NationalIncome.xlsx"), sheet_name="Consolidate")
hh_df = pd.read_excel(os.path.join(data_fldr, "Household.xlsx"), sheet_name="Consolidate")
ms_df = pd.read_excel(os.path.join(data_fldr, "MaritalStatus.xlsx"), sheet_name="Consolidate")
pp_df = pd.read_excel(os.path.join(data_fldr, "Population.xlsx"), sheet_name="Consolidate")

# Merge dataframes
coe_cat_df = coe_df.loc[coe_df['Category'] == "A", :]
dfs = [coe_cat_df, cpi_df, ni_df, hh_df, ms_df, pp_df]
merged_df = reduce(lambda left, right: pd.merge(left, right, on='Year', how='left'), dfs)
df = merged_df.drop(columns=['Category']).drop([24]).set_index('Year')

# Split data into features and target
X = df.drop('Value', axis=1)
y = df['Value']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Split data into train and test sets
test_size = int(len(X) * 0.20)
X_train = X_scaled_df.iloc[:-test_size, :]
X_test = X_scaled_df.iloc[-test_size:, :]
y_train = y.iloc[:-test_size]
y_test = y.iloc[-test_size:]

# Convert data to PyTorch tensors
X_np = X_train.values.astype(np.float32)
y_np = y_train.values.astype(np.float32)
X_tensor = torch.tensor(X_np)
y_tensor = torch.tensor(y_np).reshape(-1, 1)

# Define your neural network model
class ANN(nn.Module):
    def __init__(self, input_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, loss function, and optimizer
input_size = X_tensor.shape[1]
model = ANN(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Log parameters
with mlflow.start_run() as run:
    run_id = run.info.run_id
    mlflow.set_tag("run_id", run_id) 

    num_epochs = 2000
    batch_size = 32
    losses = []

    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", 0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(X_tensor.size(0))
        for i in range(0, X_tensor.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_X, batch_y = X_tensor[indices], y_tensor[indices]

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Log training loss for each epoch
        mlflow.log_metric('train_loss', loss.item(), step=epoch)

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save model checkpoint
    mlflow.pytorch.log_model(model, "model")

    # Evaluation mode
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test.values.astype(np.float32))
        output_tensor = model(X_test_tensor)
        pred = pd.Series(output_tensor.numpy().flatten(), name='Predicted Value')

    plt.figure(figsize=(10, 6), facecolor="white")
    plt.plot(y_test.index, y_test, label='Actual')
    plt.plot(y_test.index, pred, label='Predicted', color="Orange")
    plt.xlabel('Year')
    plt.xticks(ticks=y_test.index)
    plt.ylabel('Value')
    plt.title('COE Prices')
    plt.legend()
    plt.grid(True)
    plt.savefig("COE_Prediction_ANN.png")
    mlflow.log_artifact(f"COE_Prediction_ANN.png")

