"""
Data augmentation methods for TFP analysis using advanced generative models and statistical techniques.
This module provides tools for generating synthetic productivity datasets while preserving real-world patterns.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.covariance import EmpiricalCovariance
from scipy import stats
from scipy.stats import ks_2samp, norm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings

class ProductivityDataset(Dataset):
    """Custom Dataset class for productivity data."""
    def __init__(self, data):
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class Generator(nn.Module):
    """Enhanced Generator network for productivity data synthesis."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 4, output_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.network(x)

class Discriminator(nn.Module):
    """Enhanced Discriminator network for productivity data validation."""
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

class ProductivityGAN:
    """GAN-based productivity data augmentation with enhanced architecture."""
    def __init__(self, input_dim, hidden_dim=256, noise_dim=100):
        self.generator = Generator(noise_dim, hidden_dim, input_dim)
        self.discriminator = Discriminator(input_dim, hidden_dim)
        self.noise_dim = noise_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
    def train(self, data, epochs=1000, batch_size=64, lr=0.0001):
        """Train the GAN model with enhanced training strategy."""
        dataset = ProductivityDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        criterion = nn.BCELoss()
        
        for epoch in range(epochs):
            for batch in dataloader:
                batch_size = len(batch)
                real_data = batch.to(self.device)
                
                # Train Discriminator
                d_optimizer.zero_grad()
                label_real = torch.ones(batch_size, 1).to(self.device)
                label_fake = torch.zeros(batch_size, 1).to(self.device)
                
                output_real = self.discriminator(real_data)
                d_loss_real = criterion(output_real, label_real)
                
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                fake_data = self.generator(noise)
                output_fake = self.discriminator(fake_data.detach())
                d_loss_fake = criterion(output_fake, label_fake)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()
                
                # Train Generator
                g_optimizer.zero_grad()
                output_fake = self.discriminator(fake_data)
                g_loss = criterion(output_fake, label_real)
                g_loss.backward()
                g_optimizer.step()
                
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')
    
    def generate_samples(self, n_samples):
        """Generate synthetic productivity data samples."""
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(n_samples, self.noise_dim).to(self.device)
            synthetic_data = self.generator(noise).cpu().numpy()
        return synthetic_data

class CopulaBasedAugmentation:
    """Enhanced copula-based data augmentation for preserving dependency structures."""
    def __init__(self):
        self.scalers = {}
        self.copula_params = {}
    
    def fit(self, data):
        """Fit the copula model to the data."""
        self.n_features = data.shape[1]
        
        # Transform marginals to uniform
        for i in range(self.n_features):
            scaler = MinMaxScaler()
            self.scalers[i] = scaler.fit(data[:, i:i+1])
        
        # Estimate correlation matrix
        transformed_data = np.column_stack([
            self.scalers[i].transform(data[:, i:i+1]).ravel()
            for i in range(self.n_features)
        ])
        self.copula_params['correlation'] = np.corrcoef(transformed_data.T)
    
    def generate_samples(self, n_samples):
        """Generate synthetic samples using the fitted copula."""
        # Generate correlated uniform variables
        mvn_samples = np.random.multivariate_normal(
            mean=np.zeros(self.n_features),
            cov=self.copula_params['correlation'],
            size=n_samples
        )
        uniform_samples = stats.norm.cdf(mvn_samples)
        
        # Transform back to original scale
        synthetic_data = np.column_stack([
            self.scalers[i].inverse_transform(uniform_samples[:, i:i+1])
            for i in range(self.n_features)
        ])
        return synthetic_data

class BootstrapAugmentation:
    """Advanced bootstrap-based data augmentation with adaptive smoothing."""
    def __init__(self, smooth_factor=0.05):
        self.smooth_factor = smooth_factor
    
    def generate_samples(self, data, n_samples):
        """Generate synthetic samples using smoothed bootstrap with adaptive noise."""
        n_orig, n_features = data.shape
        synthetic_data = np.zeros((n_samples, n_features))
        
        for i in range(n_samples):
            # Sample with replacement
            idx = np.random.randint(0, n_orig, size=1)
            base_sample = data[idx]
            
            # Add noise for smoothing
            noise = np.random.normal(0, self.smooth_factor, size=(1, n_features))
            synthetic_data[i] = base_sample + noise
            
        return synthetic_data

def augment_productivity_data(data, method='gan', n_samples=1000, **kwargs):
    """
    Main function for augmenting productivity data using various methods.
    
    Parameters:
    -----------
    data : array-like
        Original productivity data
    method : str
        Augmentation method ('gan', 'copula', or 'bootstrap')
    n_samples : int
        Number of synthetic samples to generate
    **kwargs : dict
        Additional parameters for specific methods
    
    Returns:
    --------
    array-like
        Synthetic productivity data samples
    """
    data = np.array(data)
    
    if method == 'gan':
        input_dim = data.shape[1]
        gan = ProductivityGAN(input_dim=input_dim, **kwargs)
        gan.train(data)
        synthetic_data = gan.generate_samples(n_samples)
    
    elif method == 'copula':
        copula = CopulaBasedAugmentation()
        copula.fit(data)
        synthetic_data = copula.generate_samples(n_samples)
    
    elif method == 'bootstrap':
        bootstrap = BootstrapAugmentation(**kwargs)
        synthetic_data = bootstrap.generate_samples(data, n_samples)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return synthetic_data

def validate_synthetic_data(original_data, synthetic_data):
    """
    Validate synthetic data by comparing statistical properties with original data.
    
    Parameters:
    -----------
    original_data : array-like
        Original productivity data
    synthetic_data : array-like
        Generated synthetic data
    
    Returns:
    --------
    dict
        Dictionary containing validation metrics
    """
    metrics = {}
    
    # Basic statistics
    metrics['mean_diff'] = np.mean(np.abs(
        np.mean(original_data, axis=0) - np.mean(synthetic_data, axis=0)
    ))
    metrics['std_diff'] = np.mean(np.abs(
        np.std(original_data, axis=0) - np.std(synthetic_data, axis=0)
    ))
    
    # Correlation structure
    orig_corr = np.corrcoef(original_data.T)
    synth_corr = np.corrcoef(synthetic_data.T)
    metrics['corr_diff'] = np.mean(np.abs(orig_corr - synth_corr))
    
    # Distribution similarity (KS test)
    ks_stats = []
    for i in range(original_data.shape[1]):
        stat, _ = ks_2samp(original_data[:, i], synthetic_data[:, i])
        ks_stats.append(stat)
    metrics['ks_stat'] = np.mean(ks_stats)
    
    return metrics
