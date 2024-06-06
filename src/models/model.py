import os
import numpy as bo
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SemisupervisedVAE(nn.Module):
    def __init__(self, z_dim, y_dim, input_shape, device='cpu'):
        super(SemisupervisedVAE, self).__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.input_shape = input_shape
        self.devise = device
        hidden_units = 512
        
        # Encoder for q(y|x)
        self.encoder_y = nn.Sequential(
            nn.Linear(self.input_shape, hidden_units),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units),
            nn.Linear(hidden_units, self.y_dim)
        )
        
        # Encoder for q(z|x,y)
        self.encoder_z = nn.Sequential(
            nn.Linear(self.input_shape + self.y_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
        )
        self.encode_z_mean = nn.Linear(hidden_units, self.z_dim)
        self.encode_z_logvar = nn.Linear(hidden_units, self.z_dim)

        # Decoder for p(z|y)
        self.decoder_z = nn.Sequential(
            nn.Linear(self.y_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
        )
        self.decode_z_mean = nn.Linear(hidden_units, self.z_dim)
        self.decode_z_logvar = nn.Linear(hidden_units, self.z_dim)

        # Decoder for p(x|z)
        self.decoder_x = nn.Sequential(
            nn.Linear(self.z_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, self.input_shape),
            nn.Sigmoid()
        )
    

    # q(y|x)
    def qy_encoder(self, x):
        y_logits = self.encoder_y(x)
        y_prb = F.softmax(y_logits, dim=1)
        return y_logits, y_prb
    
    # q(z|x,y)
    def qz_encoder(self, x, y):
        xy = torch.cat([x,y], dim=1)
        h_z = self.encoder_z(xy)
        h = torch.relu(h_z)
        z_mean = self.encode_z_mean(h)
        z_logvar = self.encode_z_logvar(h)
        return z_mean, z_logvar
    
    # p(z|y)
    def pz_decoder(self, y):
        h_z = self.decoder_z(y)
        h = torch.relu(h_z)
        z_mean_prior = self.decode_z_mean(h)
        z_logvar_prior = self.decode_z_logvar(h)
        return z_mean_prior, z_logvar_prior
    
    # p(x|z)
    def px_decoder(self, z):
        h_x = self.decoder_x(z)
        return h_x
    
    # reparemetrize
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    
    def forward(self, lx, ly):
        '''
        lx: labeled data
        ly: labeled data labels
        return:
        y_hat: class prefiction data
        z : latent variable
        z_mu: mean of z
        z_logvar: logvar of z
        z_mean_prior: mean of z prior
        z_logvar_prior: logvar of z prior
        '''
        # ly = F.one_hot(ly, num_classes = 6)
        # ly = ly.float()
        # q(y|x)
        y_logits, y_prb = self.qy_encoder(lx)
        # q(z|x,y)
        z_mu, z_logvar = self.qz_encoder(lx, ly)
        z = self.reparametrize(z_mu, z_logvar)
        # p(z|y)
        z_mean_prior, z_logvar_prior = self.pz_decoder(ly)
        # p(x|z)
        x_hat = self.px_decoder(z)
        
        outinfo = {
            "z":z,
            "z_mu":z_mu,
            "z_logvar":z_logvar,
            "z_mean_prior":z_mean_prior,
            "z_logvar_prior":z_logvar_prior,
            "y_prb":y_prb,
            "y_logits":y_logits,
            "x_hat":x_hat
        }

        return outinfo



    
    





