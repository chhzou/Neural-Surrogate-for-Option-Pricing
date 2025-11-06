import matplotlib.pyplot as plt
import numpy as np

def plot_parity(y_true, y_pred):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=2, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Parity Plot")
    plt.grid(True)
    plt.show()
    
def plot_residuals(y_true, y_pred):
    plt.figure(figsize=(6,4))
    plt.hist(y_pred - y_true, bins=50, alpha=0.7)
    plt.title("Residuals")
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.show()

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
