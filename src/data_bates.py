import numpy as np
import yaml
from tqdm import tqdm


def bates_mc_price(S0, K, T, r, v0, kappa, theta, sigma_v, rho,\
 lam, mu_j, sigma_j, n_paths=2000, n_steps=100, option='call'):
    """
    Bates model: dS_t = S_t * ( (r-lam*E[J-1]) * dt + sqrt{v_t} * dW_t^S + (J-1) * dN_t)
    Vectorized Euler-like simulation of Bates model for European call price.
    - S0: spot
    - K: strike
    - T: maturity
    - v0: initial variance
    - kappa, theta, sigma_v: Heston params
    - rho: correlation
    - lam: jump intensity (lambda)
    - mu_j, sigma_j: parameters for log-normal jump multiplier log(J)~N(mu_j, sigma_j^2)
    Returns: discounted expected payoff (scalar)
    """
    # precompute compensator
    EJ_minus_1 = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0

    dt = T / n_steps
    # initialize states
    X = np.full((n_paths,), np.log(S0))
    v = np.full((n_paths,), v0)
    for _ in range(n_steps):
        # correlated normals for Heston diffusion
        z1 = np.random.normal(size=n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.normal(size=n_paths)

        # variance update
        v = np.maximum(v + kappa * (theta - v) * dt + sigma_v * np.sqrt(v * dt) * z1, 0.0)

        # drift term
        drift = (r - lam * EJ_minus_1 - 0.5 * v) * dt

        # diffusion term
        diffusion = np.sqrt(v * dt) * z2

        X += drift + diffusion

        # jump term
        n_jumps = np.random.poisson(lam * dt, size=n_paths)
        # Only process nonzero jumps to be efficient
        nz = n_jumps > 0
        if np.any(nz):
            # For each path with n_jumps > 0, sample sum of n iid N(mu_j, sigma_j^2)
            n = n_jumps[nz]
            total_mu = n * mu_j
            total_var = n * (sigma_j ** 2)
            Y_sum = np.random.normal(loc=total_mu, scale=np.sqrt(total_var))
            X[nz] += Y_sum
    S = np.exp(X)
    payoff = np.maximum(S - K, 0.0) if option == 'call' else np.maximum(K - S, 0.0)
    return np.exp(-r * T) * payoff.mean()

def generate_bates_dataset(cfg_path="configs/bates_config.yaml", save_path="data_bates.npz"):
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
    lam_min, lam_max = cfg.get("lambda_range", [0.0, 2.0])
    mu_min, mu_max = cfg.get("mu_j_range", [-0.2, 0.2])
    sj_min, sj_max = cfg.get("sigma_j_range", [0.05, 0.5])

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
    lam = np.random.uniform(lam_min, lam_max, n_samples)
    mu_j = np.random.uniform(mu_min, mu_max, n_samples)
    sigma_j = np.random.uniform(sj_min, sj_max, n_samples)

    # prices = [bates_mc_price(S[i], K[i], T[i], r[i], v0[i], kappa[i],\
    #  theta[i], sigma_v[i], rho[i], lam[i], mu_j[i], sigma_j[i]) \
    #  for i in tqdm(range(n_samples))]

    from joblib import Parallel, delayed
    prices = Parallel(n_jobs=-1, prefer="threads")(
        delayed(bates_mc_price)(
            S[i], K[i], T[i], r[i], v0[i], kappa[i],
            theta[i], sigma_v[i], rho[i],
            lam[i], mu_j[i], sigma_j[i]
        ) for i in tqdm(range(n_samples))
    )


    np.savez(save_path, S=S, K=K, T=T, r=r, v0=v0, kappa=kappa,\
     theta=theta, sigma_v=sigma_v, rho=rho, lam=lam, mu_j=mu_j,\
     sigma_j=sigma_j, price=prices)
    print(f"Saved Bates dataset to {save_path}")

if __name__ == "__main__":
    generate_bates_dataset()
