import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import adfuller

# === THE PFN SAMPLER
class PFNNewtonSampler:
    def __init__(self, num_timesteps: int = 100, t_min: float = 0.0, t_max: float = 120.0):
        self.num_timesteps = num_timesteps
        self.t_min = t_min
        self.t_max = t_max

    def sample_priors(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Samples physical parameters from prior distributions."""
        T_env = torch.empty(batch_size).uniform_(290.0, 310.0)   
        T_0 = torch.empty(batch_size).uniform_(5.0, 50.0)        
        k = torch.empty(batch_size).uniform_(0.005, 0.05)        
        t_0 = torch.empty(batch_size).uniform_(self.t_min, 30.0) 
        error_variance = torch.empty(batch_size).uniform_(0.1, 2.5) 

        return {
            "T_env": T_env, "T_0": T_0, "k": k, "t_0": t_0, "error_variance": error_variance
        }

    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates a batch of time series data based on the regression model with added Gaussian noise."""
        params = self.sample_priors(batch_size)
        t_grid = torch.linspace(self.t_min, self.t_max, self.num_timesteps).unsqueeze(0).repeat(batch_size, 1)
        
        T_env_exp = params["T_env"].unsqueeze(1)
        T_0_exp = params["T_0"].unsqueeze(1)
        k_exp = params["k"].unsqueeze(1)
        t_0_exp = params["t_0"].unsqueeze(1)
        var_exp = params["error_variance"].unsqueeze(1)
        
        # The actual physical model with a Heaviside step at the time of discharge
        mask = (t_grid >= t_0_exp).float()
        clean_temperature = T_env_exp + mask * T_0_exp * torch.exp(-k_exp * (t_grid - t_0_exp))
        
        # Adds homoscedastic Gaussian noise
        std_dev = torch.sqrt(var_exp)
        gaussian_noise = torch.randn_like(clean_temperature) * std_dev
        observed_T = clean_temperature + gaussian_noise
        
        targets = torch.stack([
            params["T_env"], params["T_0"], params["k"], params["t_0"], params["error_variance"]
        ], dim=1)
        
        return t_grid, observed_T, targets