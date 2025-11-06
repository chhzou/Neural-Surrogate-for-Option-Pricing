import numpy as np
import yaml


def heston_mc_price(S0, K, T, r, v0, kappa, theta, sigma_v, rho, n_paths=10000, n_steps=100, option='call'):
    dt = T / n_steps
    S = np.full((n_paths,), S0)
    v = np.full((n_paths,), v0)
    for _ in range(n_steps):
        z1 = np.random.normal(size=n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.normal(size=n_paths)
        v = np.abs(v + kappa * (theta - v) * dt + sigma_v * np.sqrt(v * dt) * z1)
        S = S * np.exp((r - 0.5 * v) * dt + np.sqrt(v * dt) * z2)
    payoff = np.maximum(S - K, 0) if option == 'call' else np.maximum(K - S, 0)
    return np.exp(-r * T) * payoff.mean()

def generate_heston_dataset(cfg_path="configs/heston_config.yaml", save_path="data_heston.npz"):
    # Load config
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Extract parameters
    S0_min, S0_max = cfg.get("S0", [50, 150])
    K_min,  K_max  = cfg.get("K", [50, 150])
    T_min,  T_max  = cfg.get("T", [0.1, 2.0])
    r_min,  r_max  = cfg.get("r", [0.0, 0.05])
    v0_min, v0_max = cfg.get("v0", [0.02, 0.3])
    kappa_min, kappa_max = cfg.get("kappa", [1.0, 3.0])
    theta_min, theta_max = cfg.get("theta", [0.02, 0.3])
    sigma_v_min, sigma_v_max = cfg.get("sigma_v", [0.1, 0.6])
    rho_min,   rho_max   = cfg.get("rho", [-0.8, 0.0])

    n_samples = cfg.get("n_samples", 5000)

    # Generate samples
    S = np.random.uniform(S0_min, S0_max, n_samples)
    K  = np.random.uniform(K_min,  K_max,  n_samples)
    T  = np.random.uniform(T_min,  T_max,  n_samples)
    r  = np.random.uniform(r_min,  r_max,  n_samples)
    v0 = np.random.uniform(v0_min, v0_max, n_samples)
    kappa = np.random.uniform(kappa_min, kappa_max, n_samples)
    theta = np.random.uniform(theta_min, theta_max, n_samples)
    sigma_v = np.random.uniform(sigma_v_min, sigma_v_max, n_samples)
    rho   = np.random.uniform(rho_min,   rho_max,   n_samples)

    prices = [heston_mc_price(S[i], K[i], T[i], r[i], v0[i], kappa[i], theta[i], sigma_v[i], rho[i]) for i in range(n_samples)]
    np.savez(save_path, S=S, K=K, T=T, r=r, v0=v0, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho, price=prices)
    print(f"Saved Heston dataset to {save_path}")

if __name__ == "__main__":
    generate_heston_dataset()
