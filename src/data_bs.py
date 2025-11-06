import numpy as np
from scipy.stats import norm

def bs_price(S, K, T, r, sigma, option='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def generate_bs_dataset(n_samples=200000, save_path="data_bs.npz"):
    S = np.exp(np.random.uniform(np.log(10), np.log(200), n_samples))
    K = np.exp(np.random.uniform(np.log(10), np.log(200), n_samples))
    T = np.random.uniform(1/365, 2.0, n_samples)
    r = np.random.uniform(0.0, 0.05, n_samples)
    sigma = np.random.uniform(0.05, 1.0, n_samples)
    price = bs_price(S, K, T, r, sigma)
    np.savez(save_path, S=S, K=K, T=T, r=r, sigma=sigma, price=price)
    print(f"Saved Black–Scholes dataset to {save_path}")

if __name__ == "__main__":
    generate_bs_dataset()
