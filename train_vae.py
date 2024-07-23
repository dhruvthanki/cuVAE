import os
import sys

# Ensure the VAE module is in the PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'VAE'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))
from VAE import VAE

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Hyperparameters
input_dim = 784
hidden_dim = 400
latent_dim = 20
batch_size = 128
epochs = 10
learning_rate = 1e-3

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model and optimizer
model = VAE(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, input_dim).float()

        # Forward pass
        recon_batch, mu, logvar = model(data)

        # Compute loss
        loss = loss_function(recon_batch, data, mu, logvar)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f'Epoch: {epoch+1}, Loss: {train_loss / len(train_loader.dataset)}')

# Save the trained weights and biases
torch.save(model.state_dict(), "vae_weights_biases.pth")
