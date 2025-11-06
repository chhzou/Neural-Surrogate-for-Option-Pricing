import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from src.models import SurrogateMLP
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="black_scholes")
args = parser.parse_args()

dataset_name = args.dataset.lower()
if dataset_name == "bates":
    data_path = "data_bates.npz"
elif dataset_name == "heston":
    data_path = "data_heston.npz"
else:
    data_path = "data_black_scholes.npz"


def train_model(data_path=data_path, lr=1e-3, epochs=200, batch_size=4096):
    with np.load(data_path) as data:
        X = np.column_stack([data[k] for k in data.keys() if k != 'price'])
        y = data['price']
    X = (X - X.mean(0)) / X.std(0)
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(X_train, y_train)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SurrogateMLP(X_train.shape[1]).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running_loss += loss.item() * len(xb)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dl.dataset):.6f}")
    if data_path == "data_black_scholes.npz":
        torch.save(model.state_dict(), 'model_bs.pt')
        print("Model saved to model_bs.pt")
    elif data_path == "data_heston.npz":
        torch.save(model.state_dict(), 'model_heston.pt')
        print("Model saved to model_heston.pt")  
    elif data_path == "data_bates.npz":
        torch.save(model.state_dict(), 'model_bates.pt')
        print("Model saved to model_bates.pt") 

if __name__ == '__main__':
    train_model()
