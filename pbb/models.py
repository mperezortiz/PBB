import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ipdb
import torch.distributions as td
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange


class Gaussian(nn.Module):
    def __init__(self, mu, rho, device='cuda', fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed)
        self.rho = nn.Parameter(rho, requires_grad=not fixed)
        self.device = device

    @property
    def sigma(self):
        # We use rho instead of sigma so that sigma is always positive during
        # the optimisation. We use sigma = log(exp(rho)+1)
        m = nn.Softplus()
        return m(self.rho)

    def sample(self):
        # Return a sample from the Gaussian distribution
        epsilon = torch.randn(self.sigma.size()).to(self.device)
        return self.mu + self.sigma * epsilon

    def compute_kl(self, other):
        # Compute KL divergence between two Gaussians
        # (refer to the paper)
        # b is the variance of priors
        b1 = torch.pow(self.sigma, 2)
        b0 = torch.pow(other.sigma, 2)

        term1 = torch.log(torch.div(b0, b1))
        term2 = torch.div(
            torch.pow(self.mu - other.mu, 2), b0)
        term3 = torch.div(b1, b0)
        kl_div = (torch.mul(term1 + term2 + term3 - 1, 0.5)).sum()
        return kl_div


class Laplace(nn.Module):
    def __init__(self, mu, rho, device='cuda', fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed)
        self.rho = nn.Parameter(rho, requires_grad=not fixed)
        self.device = device

    @property
    def scale(self):
        # We use rho instead of sigma so that sigma is always positive during
        # the optimisation. We use sigma = log(exp(rho)+1)
        m = nn.Softplus()
        return m(self.rho)

    def sample(self):
        # Return a sample from the Laplace distribution
        # we need to do scaling due to numerical issues
        epsilon = (0.999*torch.rand(self.scale.size())-0.49999).to(self.device)
        result = self.mu - torch.mul(torch.mul(self.scale, torch.sign(epsilon)),
                                     torch.log(1-2*torch.abs(epsilon)))
        return result

    def compute_kl(self, other):
        # Compute KL divergence between two Laplaces
        # (refer to the paper)
        # b is the variance of priors
        b1 = self.scale
        b0 = other.scale
        term1 = torch.log(torch.div(b0, b1))
        aux = torch.abs(self.mu - other.mu)
        term2 = torch.div(aux, b0)
        term3 = torch.div(b1, b0) * torch.exp(torch.div(-aux, b1))

        kl_div = (term1 + term2 + term3 - 1).sum()
        return kl_div


class ProbLinear(nn.Module):
    # Our network will be made of probabilistic linear layers
    def __init__(self, in_features, out_features, rho_prior, prior_dist='gaussian', device='cuda', init_layer=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Set sigma for the truncated gaussian of weights
        sigma_weights = 1/np.sqrt(in_features)

        if init_layer:
            weights_mu_init = init_layer.weight
            bias_mu_init = init_layer.bias
        else:
            # Initialise Q bias means with truncated normal
            weights_mu_init = nn.init.trunc_normal_(torch.Tensor(
                out_features, in_features), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            bias_mu_init = torch.zeros(out_features)

        weights_rho_init = torch.ones(out_features, in_features) * rho_prior

        bias_rho_init = torch.ones(out_features) * rho_prior

        if prior_dist == 'gaussian':
            self.bias = Gaussian(bias_mu_init.clone(),
                                 bias_rho_init.clone(), device=device, fixed=False)
            self.weight = Gaussian(weights_mu_init.clone(),
                                   weights_rho_init.clone(), device=device, fixed=False)
            self.weight_prior = Gaussian(
                weights_mu_init.clone(), weights_rho_init.clone(), device=device, fixed=True)
            self.bias_prior = Gaussian(
                bias_mu_init.clone(), bias_rho_init.clone(), device=device, fixed=True)
        elif prior_dist == 'laplace':
            self.bias = Laplace(bias_mu_init.clone(),
                                bias_rho_init.clone(), device=device, fixed=False)
            self.weight = Laplace(weights_mu_init.clone(),
                                  weights_rho_init.clone(), device=device, fixed=False)
            self.weight_prior = Laplace(
                weights_mu_init.clone(), weights_rho_init.clone(), device=device, fixed=True)
            self.bias_prior = Laplace(
                bias_mu_init.clone(), bias_rho_init.clone(), device=device, fixed=True)
        else:
            assert False

        self.kl_div = 0

    def forward(self, input, sample=False):
        if self.training or sample:
            # during training we sample from the model distribution
            # sample = True can also be set during testing
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            # otherwise we use the posterior
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training:
            # sum of the KL computed for weights and biases
            self.kl_div = self.weight.compute_kl(self.weight_prior) + \
                self.bias.compute_kl(self.bias_prior)

        return F.linear(input, weight, bias)


class ProbNN(nn.Module):
    def __init__(self, rho_prior, prior_dist='gaussian', device='cuda', init_net=None):
        # initialise our network
        super().__init__()
        if init_net:
            self.l1 = ProbLinear(28*28, 600, rho_prior, prior_dist=prior_dist,
                                 device=device, init_layer=init_net.l1)
            self.l2 = ProbLinear(600, 600, rho_prior, prior_dist=prior_dist,
                                 device=device, init_layer=init_net.l2)
            self.l3 = ProbLinear(600, 600, rho_prior, prior_dist=prior_dist,
                                 device=device, init_layer=init_net.l3)
            self.l4 = ProbLinear(600, 10, rho_prior, prior_dist=prior_dist,
                                 device=device, init_layer=init_net.l4)
        else:
            self.l1 = ProbLinear(28*28, 600, rho_prior, prior_dist=prior_dist,
                                 device=device)
            self.l2 = ProbLinear(600, 600, rho_prior, prior_dist=prior_dist,
                                 device=device)
            self.l3 = ProbLinear(600, 600, rho_prior, prior_dist=prior_dist,
                                 device=device)
            self.l4 = ProbLinear(600, 10, rho_prior, prior_dist=prior_dist,
                                 device=device)

    def forward(self, x, sample=False, clamping=True, pmin=1e-4):
        # forward pass for the network
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = F.relu(self.l3(x, sample))
        x = output_transform(self.l4(x, sample), clamping, pmin)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.l1.kl_div + self.l2.kl_div + self.l3.kl_div + self.l4.kl_div


class ProbConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, rho_prior, prior_dist='gaussian', device='cuda', stride=1, padding=0, dilation=1, bias=True, name='PCNN', init_layer=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1

        # Compute and set sigma for the truncated gaussian of weights
        in_features = self.in_channels
        for k in self.kernel_size:
            in_features *= k
        sigma_weights = 1/np.sqrt(in_features)

        if init_layer:
            weights_mu_init = init_layer.weight
            bias_mu_init = init_layer.bias
        else:
            weights_mu_init = nn.init.trunc_normal_(torch.Tensor(
                out_channels, in_channels, *self.kernel_size), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            bias_mu_init = torch.zeros(out_channels)

        weights_rho_init = torch.ones(
            out_channels, in_channels, *self.kernel_size) * rho_prior

        # Initialise Q bias means with truncated normal,
        # initialise Q bias rhos from RHO_PRIOR
        bias_rho_init = torch.ones(out_channels) * rho_prior

        if prior_dist == 'gaussian':
            self.weight = Gaussian(weights_mu_init.clone(),
                                   weights_rho_init.clone(), device=device, fixed=False)
            self.bias = Gaussian(bias_mu_init.clone(),
                                 bias_rho_init.clone(), device=device, fixed=False)
            # Set prior Q_0 using random initialisation and RHO_PRIOR
            self.weight_prior = Gaussian(
                weights_mu_init.clone(), weights_rho_init.clone(), device=device, fixed=True)
            self.bias_prior = Gaussian(
                bias_mu_init.clone(), bias_rho_init.clone(), device=device, fixed=True)
        elif prior_dist == 'laplace':
            self.weight = Laplace(weights_mu_init.clone(),
                                  weights_rho_init.clone(), device=device, fixed=False)
            self.bias = Laplace(bias_mu_init.clone(),
                                bias_rho_init.clone(), device=device, fixed=False)
            # Set prior Q_0 using random initialisation and RHO_PRIOR
            self.weight_prior = Laplace(
                weights_mu_init.clone(), weights_rho_init.clone(), device=device, fixed=True)
            self.bias_prior = Laplace(
                bias_mu_init.clone(), bias_rho_init.clone(), device=device, fixed=True)
        else:
            assert False

        self.kl_div = 0

    def forward(self, input, sample=False):
        if self.training or sample:
            # during training we sample from the model distribution
            # sample = True can also be set during testing
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            # otherwise we use the posterior
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training:
            # sum of the KL computed for weights and biases
            self.kl_div = self.weight.compute_kl(
                self.weight_prior) + self.bias.compute_kl(self.bias_prior)

        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)


class ProbCNN(nn.Module):
    def __init__(self, rho_prior, prior_dist='gaussian', device='cuda', init_net=None):
        # initialise our network
        super().__init__()
        if init_net:
            self.conv1 = ProbConv2d(
                1, 32, 3, rho_prior, prior_dist=prior_dist, device=device, init_layer=init_net.conv1)
            self.conv2 = ProbConv2d(
                32, 64, 3, rho_prior, prior_dist=prior_dist, device=device, init_layer=init_net.conv2)
            self.fc1 = ProbLinear(9216, 128, rho_prior, prior_dist=prior_dist,
                                  device=device, init_layer=init_net.fc1)
            self.fc2 = ProbLinear(128, 10, rho_prior, prior_dist=prior_dist,
                                  device=device, init_layer=init_net.fc2)
        else:
            self.conv1 = ProbConv2d(
                1, 32, 3, rho_prior, prior_dist=prior_dist, device=device)
            self.conv2 = ProbConv2d(
                32, 64, 3, rho_prior, prior_dist=prior_dist,  device=device)
            self.fc1 = ProbLinear(9216, 128, rho_prior,
                                  prior_dist=prior_dist, device=device)
            self.fc2 = ProbLinear(128, 10, rho_prior,
                                  prior_dist=prior_dist, device=device)

    def forward(self, x, sample=False, clamping=True, pmin=1e-4):
        # forward pass for the network
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x, sample))
        x = output_transform(self.fc2(x, sample), clamping, pmin)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.conv1.kl_div + self.conv2.kl_div + self.fc1.kl_div + self.fc2.kl_div


class CNNet0(nn.Module):
    def __init__(self, dropout_prob):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.d1 = nn.Dropout2d(dropout_prob)
        self.d2 = nn.Dropout2d(dropout_prob)
        self.d3 = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.conv1(x)
        x = self.d1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.d2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.d3(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class Linear(nn.Module):
    # Our network will be made of probabilistic linear layers
    def __init__(self, in_features, out_features, device='cuda'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Set sigma for the truncated gaussian of weights
        sigma_weights = 1/np.sqrt(in_features)

        self.weight = nn.Parameter(nn.init.trunc_normal_(torch.Tensor(
            out_features, in_features), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights), requires_grad=True)  # .to(device)
        self.bias = nn.Parameter(torch.zeros(
            out_features), requires_grad=True)  # .to(device)

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        return F.linear(input, weight, bias)


class NNet0(nn.Module):
    def __init__(self, dropout_prob=0.0, device='cuda'):
        # initialise our network
        super().__init__()
        self.l1 = Linear(28*28, 600, device)
        self.d1 = nn.Dropout(dropout_prob)
        self.l2 = Linear(600, 600, device)
        self.d2 = nn.Dropout(dropout_prob)
        self.l3 = Linear(600, 600, device)
        self.d3 = nn.Dropout(dropout_prob)
        self.l4 = Linear(600, 10, device)

    def forward(self, x):
        # forward pass for the network
        x = x.view(-1, 28*28)
        x = self.l1(x)
        x = self.d1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = self.d2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = self.d3(x)
        x = F.relu(x)
        x = output_transform(self.l4(x), clamping=False)
        return x


class NNet1(nn.Module):
    def __init__(self, dropout_prob=0.0, device='cuda'):
        # initialise our network
        super().__init__()
        self.l1 = nn.Linear(28*28, 600)
        self.d1 = nn.Dropout(dropout_prob)
        self.l2 = nn.Linear(600, 600)
        self.d2 = nn.Dropout(dropout_prob)
        self.l3 = nn.Linear(600, 600)
        self.d3 = nn.Dropout(dropout_prob)
        self.l4 = nn.Linear(600, 10)

    def forward(self, x):
        # forward pass for the network
        x = x.view(-1, 28*28)
        x = self.l1(x)
        x = self.d1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = self.d2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = self.d3(x)
        x = F.relu(x)
        x = output_transform(self.l4(x), clamping=False)
        return x


class ProbCNNLarge(nn.Module):
    def __init__(self, rho_prior, prior_dist, device='cuda', init_net=None):
        """ProbCNN Builder."""
        super().__init__()
        if init_net:
            self.conv1 = ProbConv2d(
                in_channels=3, out_channels=32, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv1)
            self.conv2 = ProbConv2d(
                in_channels=32, out_channels=64, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv2)
            self.conv3 = ProbConv2d(
                in_channels=64, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv3)
            self.conv4 = ProbConv2d(
                in_channels=128, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv4)
            self.conv5 = ProbConv2d(
                in_channels=128, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv5)
            self.conv6 = ProbConv2d(
                in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv6)
            self.fcl1 = ProbLinear(4096, 1024, rho_prior=rho_prior,
                                   prior_dist=prior_dist, device=device, init_layer=init_net.fcl1)
            self.fcl2 = ProbLinear(1024, 512, rho_prior=rho_prior,
                                   prior_dist=prior_dist, device=device, init_layer=init_net.fcl2)
            self.fcl3 = ProbLinear(
                512, 10, rho_prior=rho_prior, prior_dist=prior_dist, device=device, init_layer=init_net.fcl3)
        else:
            self.conv1 = ProbConv2d(
                in_channels=3, out_channels=32, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1)
            self.conv2 = ProbConv2d(
                in_channels=32, out_channels=64, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1)
            self.conv3 = ProbConv2d(
                in_channels=64, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1)
            self.conv4 = ProbConv2d(
                in_channels=128, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1)
            self.conv5 = ProbConv2d(
                in_channels=128, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1)
            self.conv6 = ProbConv2d(
                in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1)
            self.fcl1 = ProbLinear(
                4096, 1024, rho_prior=rho_prior, prior_dist=prior_dist, device=device)
            self.fcl2 = ProbLinear(
                1024, 512, rho_prior=rho_prior, prior_dist=prior_dist, device=device)
            self.fcl3 = ProbLinear(
                512, 10, rho_prior=rho_prior, prior_dist=prior_dist, device=device)

    def forward(self, x, sample=False, clamping=True, pmin=1e-4):
        """Perform forward."""
        # conv layers
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x, sample))
        x = F.relu(self.conv4(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv5(x, sample))
        x = F.relu(self.conv6(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = F.relu(self.fcl1(x, sample))
        x = F.relu(self.fcl2(x, sample))
        x = self.fcl3(x, sample)
        x = output_transform(x, clamping, pmin)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.conv1.kl_div + self.conv2.kl_div + self.conv3.kl_div + self.conv4.kl_div + self.conv5.kl_div + self.conv6.kl_div + self.fcl1.kl_div + self.fcl2.kl_div + self.fcl3.kl_div


class CNNet0Large(nn.Module):
    def __init__(self, dropout_prob):
        """CNN Builder."""
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.fcl1 = nn.Linear(4096, 1024)
        self.fcl2 = nn.Linear(1024, 512)
        self.fcl3 = nn.Linear(512, 10)
        self.d1 = nn.Dropout2d(0.0)
        self.d2 = nn.Dropout2d(dropout_prob)
        self.d3 = nn.Dropout2d(dropout_prob)
        self.d4 = nn.Dropout2d(dropout_prob)
        self.d5 = nn.Dropout2d(dropout_prob)
        self.d6 = nn.Dropout2d(dropout_prob)
        self.d7 = nn.Dropout(dropout_prob)
        self.d8 = nn.Dropout(dropout_prob)

    def forward(self, x):
        """Perform forward."""
        # conv layers
        x = self.d1(F.relu(self.conv1(x)))
        x = self.d2(F.relu(self.conv2(x)))
        x = (F.max_pool2d(x, kernel_size=2, stride=2))
        x = self.d3(F.relu(self.conv3(x)))
        x = self.d4(F.relu(self.conv4(x)))
        x = (F.max_pool2d(x, kernel_size=2, stride=2))
        x = self.d5(F.relu(self.conv5(x)))
        x = self.d6(F.relu(self.conv6(x)))
        x = (F.max_pool2d(x, kernel_size=2, stride=2))
        # flatten
        # ipdb.set_trace()
        x = x.view(x.size(0), -1)
        # ipdb.set_trace()
        # fc layer
        x = F.relu(self.d7(self.fcl1(x)))
        x = F.relu(self.d8(self.fcl2(x)))
        x = self.fcl3(x)
        x = F.log_softmax(x, dim=1)
        return x


class CNNet0EvenLarger(nn.Module):
    def __init__(self, dropout_prob):
        """CNN Builder."""
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64,  kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64,  kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128,  kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128,  kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256,  kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=256,  kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(
            in_channels=256, out_channels=256,  kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(
            in_channels=256, out_channels=256,  kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(
            in_channels=256, out_channels=512,  kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(
            in_channels=512, out_channels=512,  kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(
            in_channels=512, out_channels=512,  kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(
            in_channels=512, out_channels=512,  kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(
            in_channels=512, out_channels=512,  kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(
            in_channels=512, out_channels=512,  kernel_size=3, padding=1)
        self.conv15 = nn.Conv2d(
            in_channels=512, out_channels=512,  kernel_size=3, padding=1)
        self.conv16 = nn.Conv2d(
            in_channels=512, out_channels=512,  kernel_size=3, padding=1)
        #self.fcl1 = nn.Linear(512, 10)
        self.fcl1 = nn.Linear(512, 512)
        self.fcl2 = nn.Linear(512, 512)
        self.fcl3 = nn.Linear(512, 10)
        self.d1 = nn.Dropout(dropout_prob)
        self.d2 = nn.Dropout(dropout_prob)
        self.d3 = nn.Dropout(dropout_prob)
        self.d4 = nn.Dropout(dropout_prob)
        self.d5 = nn.Dropout(dropout_prob)
        self.d6 = nn.Dropout(dropout_prob)
        self.d7 = nn.Dropout(dropout_prob)
        self.d8 = nn.Dropout(dropout_prob)
        self.d9 = nn.Dropout(dropout_prob)
        self.d10 = nn.Dropout(dropout_prob)
        self.d11 = nn.Dropout(dropout_prob)
        self.d12 = nn.Dropout(dropout_prob)
        self.d13 = nn.Dropout(dropout_prob)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.d14 = nn.Dropout(dropout_prob)
        self.d15 = nn.Dropout(dropout_prob)
        self.d16 = nn.Dropout(dropout_prob)
        self.d17 = nn.Dropout(dropout_prob)
        self.d18 = nn.Dropout(dropout_prob)

    def forward(self, x):
        """Perform forward."""
        # conv layers
        x = F.relu(self.d1(self.conv1(x)))
        x = F.relu(self.d2(self.conv2(x)))
        x = self.pooling(x)
        x = F.relu(self.d3(self.conv3(x)))
        x = F.relu(self.d4(self.conv4(x)))
        x = self.pooling(x)
        x = F.relu(self.d5(self.conv5(x)))
        x = F.relu(self.d6(self.conv6(x)))
        x = F.relu(self.d7(self.conv7(x)))
        x = F.relu(self.d8(self.conv8(x)))
        x = self.pooling(x)
        x = F.relu(self.d9(self.conv9(x)))
        x = F.relu(self.d10(self.conv10(x)))
        x = F.relu(self.d11(self.conv11(x)))
        x = F.relu(self.d12(self.conv12(x)))
        x = self.pooling(x)
        # ipdb.set_trace()
        x = F.relu(self.d13(self.conv13(x)))
        x = F.relu(self.d14(self.conv14(x)))
        x = F.relu(self.d15(self.conv15(x)))
        x = F.relu(self.d16(self.conv16(x)))
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        # fc layer
        x = F.relu(self.d17(self.fcl1(x)))
        x = F.relu(self.d18(self.fcl2(x)))
        x = self.fcl3(x)
        x = F.log_softmax(x, dim=1)
        return x


class ProbCNNEvenLarger(nn.Module):
    def __init__(self, rho_prior, prior_dist, device='cuda', init_net=None):
        """ProbCNN Builder."""
        super().__init__()
        if init_net:
            self.conv1 = ProbConv2d(
                in_channels=3, out_channels=64, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv1)
            self.conv2 = ProbConv2d(
                in_channels=64, out_channels=64, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv2)
            self.conv3 = ProbConv2d(
                in_channels=64, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv3)
            self.conv4 = ProbConv2d(
                in_channels=128, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv4)
            self.conv5 = ProbConv2d(
                in_channels=128, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv5)
            self.conv6 = ProbConv2d(
                in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv6)
            self.conv7 = ProbConv2d(
                in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv7)
            self.conv7 = ProbConv2d(
                in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv7)
            self.conv8 = ProbConv2d(
                in_channels=256, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv8)
            self.conv9 = ProbConv2d(
                in_channels=512, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv9)
            self.conv10 = ProbConv2d(
                in_channels=512, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv10)
            self.conv11 = ProbConv2d(
                in_channels=512, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv11)
            self.conv12 = ProbConv2d(
                in_channels=512, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv12)
            self.conv13 = ProbConv2d(
                in_channels=512, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv13)
            self.fcl1 = ProbLinear(512, 512, rho_prior=rho_prior,
                                   prior_dist=prior_dist, device=device, init_layer=init_net.fcl1)
            self.fcl2 = ProbLinear(512, 512, rho_prior=rho_prior,
                                   prior_dist=prior_dist, device=device, init_layer=init_net.fcl2)
            self.fcl3 = ProbLinear(
                512, 10, rho_prior=rho_prior, prior_dist=prior_dist, device=device, init_layer=init_net.fcl3)
        else:
            assert False
            self.conv1 = ProbConv2d(
                in_channels=3, out_channels=64, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1)
            self.conv2 = ProbConv2d(
                in_channels=32, out_channels=64, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1)
            self.conv3 = ProbConv2d(
                in_channels=64, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1)
            self.conv4 = ProbConv2d(
                in_channels=128, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1)
            self.conv5 = ProbConv2d(
                in_channels=128, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1)
            self.conv6 = ProbConv2d(
                in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1)
            self.fcl1 = ProbLinear(
                4096, 1024, rho_prior=rho_prior, prior_dist=prior_dist, device=device)
            self.fcl2 = ProbLinear(
                1024, 512, rho_prior=rho_prior, prior_dist=prior_dist, device=device)
            self.fcl3 = ProbLinear(
                512, 10, rho_prior=rho_prior, prior_dist=prior_dist, device=device)

    def forward(self, x, sample=False, clamping=True, pmin=1e-4):
        """Perform forward."""
        # conv layers
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x, sample))
        x = F.relu(self.conv4(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv5(x, sample))
        x = F.relu(self.conv6(x, sample))
        x = F.relu(self.conv7(x, sample))
        x = F.relu(self.conv8(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv9(x, sample))
        x = F.relu(self.conv10(x, sample))
        x = F.relu(self.conv11(x, sample))
        x = F.relu(self.conv12(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv13(x, sample))
        x = F.relu(self.conv14(x, sample))
        x = F.relu(self.conv15(x, sample))
        x = F.relu(self.conv16(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = F.relu(self.fcl1(x, sample))
        x = F.relu(self.fcl2(x, sample))
        x = self.fcl3(x, sample)
        x = output_transform(x, clamping, pmin)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.conv1.kl_div + self.conv2.kl_div + self.conv3.kl_div + self.conv4.kl_div + self.conv5.kl_div + self.conv6.kl_div + self.conv7.kl_div + self.conv8.kl_div + self.conv9.kl_div + self.conv10.kl_div + self.conv11.kl_div + self.conv12.kl_div + self.conv13.kl_div + self.fcl1.kl_div + self.fcl2.kl_div + self.fcl3.kl_div


class CNNet0Larger(nn.Module):
    def __init__(self, dropout_prob):
        """CNN Builder."""
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.fcl1 = nn.Linear(2048, 1024)
        # self.fcl2 = nn.Linear(4096, 1024)
        self.fcl2 = nn.Linear(1024, 512)
        self.fcl3 = nn.Linear(512, 10)
        self.d1 = nn.Dropout(dropout_prob)
        self.d2 = nn.Dropout(dropout_prob)
        self.d3 = nn.Dropout(dropout_prob)
        self.d4 = nn.Dropout(dropout_prob)
        self.d5 = nn.Dropout(dropout_prob)
        self.d6 = nn.Dropout(dropout_prob)
        self.d7 = nn.Dropout(dropout_prob)
        self.d8 = nn.Dropout(dropout_prob)
        self.d9 = nn.Dropout(dropout_prob)
        self.d10 = nn.Dropout(dropout_prob)
        self.d11 = nn.Dropout(dropout_prob)
        self.d12 = nn.Dropout(dropout_prob)

    def forward(self, x):
        """Perform forward."""
        # conv layers
        x = F.relu(self.d1(self.conv1(x)))
        x = F.relu(self.d2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.d3(self.conv3(x)))
        x = F.relu(self.d4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.d5(self.conv5(x)))
        x = F.relu(self.d6(self.conv6(x)))
        x = F.relu(self.d7(self.conv7(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.d8(self.conv8(x)))
        x = F.relu(self.d9(self.conv9(x)))
        x = F.relu(self.d10(self.conv10(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = F.relu(self.d11(self.fcl1(x)))
        x = F.relu(self.d12(self.fcl2(x)))
        x = self.fcl3(x)
        x = F.log_softmax(x, dim=1)
        return x


class CNNet0Larger2(nn.Module):
    def __init__(self, dropout_prob):
        """CNN Builder."""
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.fcl1 = nn.Linear(2048, 1024)
        self.fcl2 = nn.Linear(1024, 512)
        self.fcl3 = nn.Linear(512, 10)
        self.d1 = nn.Dropout(dropout_prob)
        self.d2 = nn.Dropout(dropout_prob)
        self.d3 = nn.Dropout(dropout_prob)
        self.d4 = nn.Dropout(dropout_prob)
        self.d5 = nn.Dropout(dropout_prob)
        self.d6 = nn.Dropout(dropout_prob)
        self.d7 = nn.Dropout(dropout_prob)
        self.d8 = nn.Dropout(dropout_prob)
        self.d9 = nn.Dropout(dropout_prob)
        self.d10 = nn.Dropout(dropout_prob)
        self.d11 = nn.Dropout(dropout_prob)
        self.d12 = nn.Dropout(dropout_prob)
        self.d13 = nn.Dropout(dropout_prob)
        self.d14 = nn.Dropout(dropout_prob)

    def forward(self, x):
        """Perform forward."""
        # conv layers
        x = F.relu(self.d1(self.conv1(x)))
        x = F.relu(self.d2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.d3(self.conv3(x)))
        x = F.relu(self.d4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.d5(self.conv5(x)))
        x = F.relu(self.d6(self.conv6(x)))
        x = F.relu(self.d7(self.conv7(x)))
        x = F.relu(self.d8(self.conv8(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.d9(self.conv9(x)))
        x = F.relu(self.d10(self.conv10(x)))
        x = F.relu(self.d11(self.conv11(x)))
        x = F.relu(self.d12(self.conv12(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = F.relu(self.d13(self.fcl1(x)))
        x = F.relu(self.d14(self.fcl2(x)))
        x = self.fcl3(x)
        x = F.log_softmax(x, dim=1)
        return x


class ProbCNNLarger2(nn.Module):
    def __init__(self, rho_prior, prior_dist, device='cuda', init_net=None):
        """ProbCNN Builder."""
        super().__init__()
        if init_net:
            self.conv1 = ProbConv2d(
                in_channels=3, out_channels=32, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv1)
            self.conv2 = ProbConv2d(
                in_channels=32, out_channels=64, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv2)
            self.conv3 = ProbConv2d(
                in_channels=64, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv3)
            self.conv4 = ProbConv2d(
                in_channels=128, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv4)
            self.conv5 = ProbConv2d(
                in_channels=128, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv5)
            self.conv6 = ProbConv2d(
                in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv6)
            self.conv7 = ProbConv2d(
                in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv7)
            self.conv8 = ProbConv2d(
                in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv8)
            self.conv9 = ProbConv2d(
                in_channels=256, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv9)
            self.conv10 = ProbConv2d(
                in_channels=512, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv10)
            self.conv11 = ProbConv2d(
                in_channels=512, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv11)
            self.conv12 = ProbConv2d(
                in_channels=512, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv12)
            self.fcl1 = ProbLinear(2048, 1024, rho_prior=rho_prior,
                                   prior_dist=prior_dist, device=device, init_layer=init_net.fcl1)
            self.fcl2 = ProbLinear(1024, 512, rho_prior=rho_prior,
                                   prior_dist=prior_dist, device=device, init_layer=init_net.fcl2)
            self.fcl3 = ProbLinear(
                512, 10, rho_prior=rho_prior, prior_dist=prior_dist, device=device, init_layer=init_net.fcl3)
        else:
            assert False
            self.conv1 = ProbConv2d(
                in_channels=3, out_channels=64, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1)
            self.conv2 = ProbConv2d(
                in_channels=32, out_channels=64, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1)
            self.conv3 = ProbConv2d(
                in_channels=64, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1)
            self.conv4 = ProbConv2d(
                in_channels=128, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1)
            self.conv5 = ProbConv2d(
                in_channels=128, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1)
            self.conv6 = ProbConv2d(
                in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1)
            self.fcl1 = ProbLinear(
                4096, 1024, rho_prior=rho_prior, prior_dist=prior_dist, device=device)
            self.fcl2 = ProbLinear(
                1024, 512, rho_prior=rho_prior, prior_dist=prior_dist, device=device)
            self.fcl3 = ProbLinear(
                512, 10, rho_prior=rho_prior, prior_dist=prior_dist, device=device)

    def forward(self, x, sample=False, clamping=True, pmin=1e-4):
        """Perform forward."""
        # conv layers
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x, sample))
        x = F.relu(self.conv4(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv5(x, sample))
        x = F.relu(self.conv6(x, sample))
        x = F.relu(self.conv7(x, sample))
        x = F.relu(self.conv8(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv9(x, sample))
        x = F.relu(self.conv10(x, sample))
        x = F.relu(self.conv11(x, sample))
        x = F.relu(self.conv12(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = F.relu(self.fcl1(x, sample))
        x = F.relu(self.fcl2(x, sample))
        x = self.fcl3(x, sample)
        x = output_transform(x, clamping, pmin)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.conv1.kl_div + self.conv2.kl_div + self.conv3.kl_div + self.conv4.kl_div + self.conv5.kl_div + self.conv6.kl_div + self.conv7.kl_div + self.conv8.kl_div + self.conv9.kl_div + self.conv10.kl_div + self.conv11.kl_div + self.conv12.kl_div + self.fcl1.kl_div + self.fcl2.kl_div + self.fcl3.kl_div


class ProbCNNLarger(nn.Module):
    def __init__(self, rho_prior, prior_dist, device='cuda', init_net=None):
        """ProbCNN Builder."""
        super().__init__()
        if init_net:
            self.conv1 = ProbConv2d(
                in_channels=3, out_channels=32, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv1)
            self.conv2 = ProbConv2d(
                in_channels=32, out_channels=64, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv2)
            self.conv3 = ProbConv2d(
                in_channels=64, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv3)
            self.conv4 = ProbConv2d(
                in_channels=128, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv4)
            self.conv5 = ProbConv2d(
                in_channels=128, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv5)
            self.conv6 = ProbConv2d(
                in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv6)
            self.conv7 = ProbConv2d(
                in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv7)
            self.conv8 = ProbConv2d(
                in_channels=256, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv8)
            self.conv9 = ProbConv2d(
                in_channels=512, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv9)
            self.conv10 = ProbConv2d(
                in_channels=512, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1, init_layer=init_net.conv10)
            self.fcl1 = ProbLinear(2048, 1024, rho_prior=rho_prior,
                                   prior_dist=prior_dist, device=device, init_layer=init_net.fcl1)
            self.fcl2 = ProbLinear(1024, 512, rho_prior=rho_prior,
                                   prior_dist=prior_dist, device=device, init_layer=init_net.fcl2)
            self.fcl3 = ProbLinear(
                512, 10, rho_prior=rho_prior, prior_dist=prior_dist, device=device, init_layer=init_net.fcl3)
        else:
            assert False
            self.conv1 = ProbConv2d(
                in_channels=3, out_channels=64, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1)
            self.conv2 = ProbConv2d(
                in_channels=32, out_channels=64, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1)
            self.conv3 = ProbConv2d(
                in_channels=64, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1)
            self.conv4 = ProbConv2d(
                in_channels=128, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1)
            self.conv5 = ProbConv2d(
                in_channels=128, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1)
            self.conv6 = ProbConv2d(
                in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device, kernel_size=3, padding=1)
            self.fcl1 = ProbLinear(
                4096, 1024, rho_prior=rho_prior, prior_dist=prior_dist, device=device)
            self.fcl2 = ProbLinear(
                1024, 512, rho_prior=rho_prior, prior_dist=prior_dist, device=device)
            self.fcl3 = ProbLinear(
                512, 10, rho_prior=rho_prior, prior_dist=prior_dist, device=device)

    def forward(self, x, sample=False, clamping=True, pmin=1e-4):
        """Perform forward."""
        # conv layers
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x, sample))
        x = F.relu(self.conv4(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv5(x, sample))
        x = F.relu(self.conv6(x, sample))
        x = F.relu(self.conv7(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv8(x, sample))
        x = F.relu(self.conv9(x, sample))
        x = F.relu(self.conv10(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = F.relu(self.fcl1(x, sample))
        x = F.relu(self.fcl2(x, sample))
        x = self.fcl3(x, sample)
        x = output_transform(x, clamping, pmin)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.conv1.kl_div + self.conv2.kl_div + self.conv3.kl_div + self.conv4.kl_div + self.conv5.kl_div + self.conv6.kl_div + self.conv7.kl_div + self.conv8.kl_div + self.conv9.kl_div + self.conv10.kl_div + self.fcl1.kl_div + self.fcl2.kl_div + self.fcl3.kl_div


def output_transform(x, clamping=True, pmin=1e-4):
    # lower bound output prob
    output = F.log_softmax(x, dim=1)
    if clamping:
        output = torch.clamp(output, np.log(pmin))
    return output


class Lambda_var(nn.Module):
    def __init__(self, lamb, n):
        super().__init__()
        self.lamb = nn.Parameter(torch.tensor([lamb]), requires_grad=True)
        self.min = 1/np.sqrt(n)

    @property
    def lamb_scaled(self):
        # We restrict lamb_scaled to be between 1/sqrt(n) and 1.
        m = nn.Sigmoid()
        return (m(self.lamb) * (1-self.min) + self.min)


def trainNet0(net, optimizer, epoch, valid_loader, device='cuda', verbose=False):
    # train and report training metrics
    net.train()
    total, correct, avgloss = 0.0, 0.0, 0.0
    for batch_id, (data, target) in enumerate(tqdm(valid_loader)):
        data, target = data.to(device), target.to(device)
        net.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        avgloss = avgloss + loss.detach()
    # show the average loss and KL during the epoch
    if verbose:
        print(
            f"-Epoch {epoch :.5f}, Train loss: {avgloss/batch_id :.5f}, Train err:  {1-(correct/total):.5f}")


def testNet0(net, test_loader, device='cuda', verbose=True):
    # compute mean test zero-one error
    net.eval()
    correct, total = 0, 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = net(data)
            loss = F.nll_loss(outputs, target)
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    print(
        f"-Test loss: {loss :.5f}, Test err:  {1-(correct/total):.5f}")

    return 1-(correct/total)


def trainPNN(net, optimizer, pbbound, epoch, train_loader, lambda_var=None, optimizer_lambda=None, verbose=False, rub01=None):
    # train and report training metrics
    net.train()
    # variables that keep information about the results of optimising the bound
    avgerr, avgbound, avgkl, avgloss = 0.0, 0.0, 0.0, 0.0

    if pbbound.objective == 'flamb':
        lambda_var.train()
        # variables that keep information about the results of optimising lambda (only for flamb)
        avgerr_l, avgbound_l, avgkl_l, avgloss_l = 0.0, 0.0, 0.0, 0.0

    if pbbound.objective == 'bbb':
        clamping = False
    else:
        clamping = True

    for batch_id, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(pbbound.device), target.to(pbbound.device)
        net.zero_grad()
        bound, kl, _, loss, err = pbbound.train_obj(
            net, data, target, lambda_var=lambda_var, clamping=clamping, rub01=rub01)
        bound.backward()
        optimizer.step()
        avgbound += bound.item()
        # ipdb.set_trace()
        avgkl += kl
        avgloss += loss.item()
        avgerr += err

        if pbbound.objective == 'flamb':
            # for flamb we also need to optimise the lambda variable
            lambda_var.zero_grad()
            # import ipdb
            # ipdb.set_trace()
            bound_l, kl_l, _, loss_l, err_l = pbbound.train_obj(
                net, data, target, lambda_var=lambda_var, clamping=clamping)
            bound_l.backward()
            optimizer_lambda.step()
            avgbound_l += bound_l.item()
            avgkl_l += kl_l
            avgloss_l += loss_l.item()
            avgerr_l += err_l

    if verbose:
        # show the average of the metrics during the epoch
        print(
            f"-Batch average epoch {epoch :.0f} results, Train obj: {avgbound/batch_id :.5f}, KL/n: {avgkl/batch_id :.5f}, NLL loss: {avgloss/batch_id :.5f}, Train 0-1 Error:  {avgerr/batch_id :.5f}")
        if pbbound.objective == 'flamb':
            print(
                f"-After optimising lambda: Train obj: {avgbound_l/batch_id :.5f}, KL/n: {avgkl_l/batch_id :.5f}, NLL loss: {avgloss_l/batch_id :.5f}, Train 0-1 Error:  {avgerr_l/batch_id :.5f}, last lambda value: {lambda_var.lamb_scaled.item() :.5f}")


def testStochastic(net, test_loader, pbbound, device='cuda'):
    # compute mean test accuracy
    net.eval()
    correct, cross_entropy, total = 0, 0.0, 0.0
    outputs = torch.zeros(test_loader.batch_size, pbbound.classes).to(device)
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(tqdm(test_loader)):
            data, target = data.to(device), target.to(device)
            for i in range(len(data)):
                outputs[i, :] = net(data[i:i+1], sample=True,
                                    clamping=True, pmin=pbbound.pmin)
            cross_entropy += pbbound.compute_empirical_risk(
                outputs, target, bounded=True)
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    return cross_entropy/batch_id, 1-(correct/total)


def testPosterior(net, test_loader, pbbound, device='cuda'):
    # compute mean test accuracy
    net.eval()
    correct, total = 0, 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = net(data, sample=False, clamping=True, pmin=pbbound.pmin)
            cross_entropy = pbbound.compute_empirical_risk(
                outputs, target, bounded=True)
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    return cross_entropy, 1-(correct/total)


def computeFinalMetrics(net, toolarge, pbbound, device='cuda', lambda_var=None, train_loader=None, whole_train=None):
    net.eval()
    with torch.no_grad():
        if toolarge:
            train_obj_rub_train, kl, loss_ce_train, err_01_train, pbklbce_train, pbkl01_train = pbbound.compute_final_bounds(
                net, lambda_var=lambda_var, clamping=True, data_loader=train_loader)
        else:
            # a bit hacky, we load the whole dataset to compute the bound
            for data, target in whole_train:
                data, target = data.to(device), target.to(device)
                train_obj_rub_train, kl, loss_ce_train, err_01_train, pbklbce_train, pbkl01_train = pbbound.compute_final_bounds(
                    net, lambda_var=lambda_var, clamping=True, input=data, target=target)

    return train_obj_rub_train, pbklbce_train, pbkl01_train, kl, loss_ce_train, err_01_train


def testEnsemble(net, test_loader, pbbound, device='cuda', samples=100):
    net.eval()
    correct, cross_entropy, total = 0, 0.0, 0.0
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(tqdm(test_loader)):
            data, target = data.to(device), target.to(device)
            outputs = torch.zeros(samples, test_loader.batch_size,
                                  pbbound.classes).to(device)
            for i in range(samples):
                outputs[i] = net(data, sample=True,
                                 clamping=True, pmin=pbbound.pmin)
            avgoutput = outputs.mean(0)
            cross_entropy = pbbound.compute_empirical_risk(
                avgoutput, target, bounded=True)
            pred = avgoutput.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    return cross_entropy/batch_id, 1-(correct/total)
