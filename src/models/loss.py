import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

class LossFunctions:
    eps = 1e-8

    def reconstruction_loss(self, real, predictions):
      """Mean Squared Error between the true and predicted outputs
         loss = Σ(real - predicted)^2

      Args:
          real: (array) corresponding array containing the true labels
          predictions: (array) corresponding array containing the predicted labels
 
      Returns:
          output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      loss = torch.sum((real - predictions)**2,axis=1)
      return -torch.log(torch.mean(loss))


    def log_normal(self, x, mu, log_var):
      """Logarithm of normal distribution with mean=mu and variance=var
         log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

      Args:
         x: (array) corresponding array containing the input
         mu: (array) corresponding array containing the mean 
         var: (array) corresponding array containing the variance

      Returns:
         output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      d = x.size(-1)
      return -d/2*np.log(2*np.pi)-1/2*torch.sum(log_var,1)-1/2*torch.sum((x-mu)**2/torch.exp(log_var),1)
    #   if self.eps > 0.0:
    #     var = var + self.eps
    #   return -0.5 * torch.sum(
    #     np.log(2.0 * np.pi) + 0.5*var + torch.pow(x - mu, 2) / torch.exp(var), dim=-1)


    def gaussian_loss(self, z, z_mu, z_log_var, z_mu_prior, z_log_var_prior):
      """Variational loss when using labeled data without considering reconstruction loss
         loss = log q(z|x,y) - log p(z) - log p(y)

      Args:
         z: (array) array containing the gaussian latent variable
         z_mu: (array) array containing the mean of the inference model
         z_log_var: (array) array containing the variance of the inference model
         z_mu_prior: (array) array containing the prior mean of the generative model
         z_log_var_prior: (array) array containing the prior variance of the generative mode
         
      Returns:
         output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      loss = self.log_normal(z, z_mu, z_log_var) - self.log_normal(z, z_mu_prior, z_log_var_prior)
    #   print(f'gaussian_loss : {loss}')
      return torch.mean(loss)


    def entropy(self, logits, targets):
      """Entropy loss
          loss = (1/n) * -Σ targets*log(predicted)

      Args:
          logits: (array) corresponding array containing the logits of the categorical variable
          real: (array) corresponding array containing the true labels
 
      Returns:
          output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      loss_cross = nn.CrossEntropyLoss()
      logits = F.log_softmax(logits, dim=-1)
      loss = torch.mean(loss_cross(logits, targets))
      return loss
    #   targets = F.one_hot(targets, num_classes=6)
    #   targets = targets.float()
    #   log_q = F.log_softmax(logits, dim=-1)

      
    #   return torch.mean(torch.sum(targets * log_q, dim=-1))

