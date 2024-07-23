# VAE/vae.py
import vae

import numpy as np
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.weights1 = nn.Parameter(torch.randn(input_dim, hidden_dim).float())
        self.bias1 = nn.Parameter(torch.randn(hidden_dim).float())
        self.weights2 = nn.Parameter(torch.randn(hidden_dim, latent_dim).float())
        self.bias2 = nn.Parameter(torch.randn(latent_dim).float())
        self.weights3 = nn.Parameter(torch.randn(latent_dim, hidden_dim).float())
        self.bias3 = nn.Parameter(torch.randn(hidden_dim).float())
        self.weights4 = nn.Parameter(torch.randn(hidden_dim, input_dim).float())
        self.bias4 = nn.Parameter(torch.randn(input_dim).float())

    def encode(self, x):
        mu = torch.zeros((x.shape[0], self.latent_dim), dtype=torch.float32).numpy()
        logvar = torch.zeros((x.shape[0], self.latent_dim), dtype=torch.float32).numpy()
        vae.encode(x.numpy(), self.weights1.detach().numpy(), self.bias1.detach().numpy(), self.weights2.detach().numpy(), self.bias2.detach().numpy(), mu, logvar, self.input_dim, self.hidden_dim, self.latent_dim, x.shape[0])
        return torch.tensor(mu), torch.tensor(logvar)

    def reparameterize(self, mu, logvar):
        z = torch.zeros_like(mu).numpy()
        vae.reparameterize(mu.numpy(), logvar.numpy(), z, self.latent_dim, np.random.randint(1e6), mu.shape[0])
        return torch.tensor(z)

    def decode(self, z):
        output = torch.zeros((z.shape[0], self.input_dim), dtype=torch.float32).numpy()
        vae.decode(z.numpy(), self.weights3.detach().numpy(), self.bias3.detach().numpy(), self.weights4.detach().numpy(), self.bias4.detach().numpy(), output, self.latent_dim, self.hidden_dim, self.input_dim, z.shape[0])
        return torch.tensor(output)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
