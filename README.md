# Neural Surrogate for Option Pricing

This repository provides neural-network surrogate models for fast and accurate European option pricing under multiple stochastic models: Black–Scholes, Heston, and Bates.

## Features
- Synthetic data generation for:
  - Black–Scholes (analytic solutions)
  - Heston (Monte Carlo simulation)
  - Bates (stochastic volatility with jump-diffusion, Monte Carlo)
- PyTorch-based MLP surrogate models for pricing options.
- Lightweight training and evaluation pipelines.
- Jupyter notebook (Google Colab–ready).
- Speed and accuracy benchmarking against Monte Carlo methods

## Structure
```
option-surrogate/
├── configs/
│   ├── bates_config.yaml
│   ├── bs_config.yaml
│   └── heston_config.yaml
├── notebooks/
│   └── Neural_Surrogate_for_Option_Pricing_(Bates).ipynb
├── requirements.txt
├── README.md
├── src/
      ├── data_bates.py
      ├── data_bs.py
      ├── data_heston.py
      ├── eval.py
      ├── models.py
      ├── train.py
      └── utils.py
```


## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate synthetic data:
   ```bash
   python src/data_bates.py  # Bates
   python src/data_heston.py # Heston
   python src/data_bs.py     # Black–Scholes
   ```

3. Train surrogate:
   ```bash
   python src/train.py --dataset bates  
   python src/train.py --dataset heston 
   python src/train.py --dataset bs     
   ```

4. Evaluate results:
   ```bash
   python src/eval.py --dataset bates
   python src/eval.py --dataset heston   
   python src/eval.py --dataset bs
   ```
5. Interactive demos:

   Open the Bates model demo notebook for an interactive, step-by-step workflow:
   `notebooks/Neural_Surrogate_for_Option_Pricing_(Bates).ipynb`

   This notebook is Google Colab–ready and covers:
   - Synthetic data generation for Bates model
   - Training and evaluation of neural surrogate
   - Speed benchmarking against Monte Carlo pricing

## Requirements
See `requirements.txt`.

## License
MIT