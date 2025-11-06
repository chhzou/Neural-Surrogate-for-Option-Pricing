import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models import SurrogateMLP
from src.utils import plot_parity, plot_residuals, mse, mae
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="black_scholes")
args = parser.parse_args()

dataset_name = args.dataset.lower()
if dataset_name == "bates":
    data_path = "data_bates.npz"
    model_path = "model_bates.pt"
elif dataset_name == "heston":
    data_path = "data_heston.npz"
    model_path = "model_heston.pt"
else:
    data_path = "data_bs.npz"
    model_path = "model_bs.pt"

data = np.load(data_path)
X = np.column_stack([data[k] for k in data.keys() if k != 'price'])
y_true = data['price']
X = (X - X.mean(0)) / X.std(0)
X_torch = torch.tensor(X, dtype=torch.float32)

model = SurrogateMLP(X.shape[1])
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

with torch.no_grad():
    y_pred = model(X_torch).numpy()

print(f"MSE: {mse(y_true, y_pred):.4e}")
print(f"MAE: {mae(y_true, y_pred):.4e}")

plot_parity(y_true, y_pred)
plot_residuals(y_true, y_pred)