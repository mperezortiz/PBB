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
from pbb.models import ProbNN, ProbCNN, ProbCNNLarge, CNNet0EvenLarger, ProbCNNLarger2, CNNet0Larger2, ProbCNNEvenLarger, ProbCNNLarger, CNNet0, CNNet0Large, CNNet0Larger, NNet0, NNet1, trainNet0, testNet0, Lambda_var, trainPNN, computeFinalMetrics, testPosterior, testStochastic, testEnsemble
from pbb.bounds import PBBound
from pbb import data
import matplotlib.pyplot as plt


def runexp(name_data, objective, prior_type, model, sigma_prior, pmin, learning_rate, momentum, learning_rate_prior, momentum_prior, delta=0.025, larger=False, delta_test=0.01, mc_samples=1000, samples_ensemble=100, bbb_penalty=0.1, initial_lamb=6.0, train_epochs=500, prior_dist='gaussian', verbose=False, device='cuda', prior_epochs=20, dropout_prob=0.2, perc_train=1.0, verbose_test=False, perc_prior=0.2, noise_targets=0.0, batch_size=250):
    toolarge = False
    # this makes the initialised prior the same for all bounds
    torch.manual_seed(7)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    loader_kargs = {'num_workers': 1,
                    'pin_memory': True} if torch.cuda.is_available() else {}
    train, test = data.loaddataset(name_data, noise=noise_targets)

    if prior_type == 'rand':
        dropout_prob = 0.0

    # initialise model
    if model == 'cnn':
        if name_data == 'cifar10':
            if larger == True:
                net0 = CNNet0Larger(dropout_prob=dropout_prob).to(device)
            else:
                net0 = CNNet0Large(dropout_prob=dropout_prob).to(device)
        else:
            net0 = CNNet0(dropout_prob=dropout_prob).to(device)
    else:
        net0 = NNet0(dropout_prob=dropout_prob, device=device).to(device)

    # ipdb.set_trace()
    if prior_type == 'rand':
        train_loader, test_loader, _, val_bound_one_batch, _, val_bound = data.loadbatches(
            train, test, loader_kargs, batch_size, prior=False, perc_train=perc_train, perc_prior=perc_prior)
        errornet0 = testNet0(net0, test_loader, device=device)
    elif prior_type == 'learnt':
        train_loader, test_loader, valid_loader, val_bound_one_batch, _, val_bound = data.loadbatches(
            train, test, loader_kargs, batch_size, prior=True, perc_train=perc_train, perc_prior=perc_prior)
        optimizer = optim.SGD(
            net0.parameters(), lr=learning_rate_prior, momentum=momentum_prior)
        for epoch in trange(prior_epochs):
            trainNet0(net0, optimizer, epoch, valid_loader,
                      device=device, verbose=verbose)
        errornet0 = testNet0(net0, test_loader, device=device)

    train_size = len(train_loader.dataset)
    classes = len(train_loader.dataset.classes)

    rho_prior = math.log(math.exp(sigma_prior)-1.0)
    if model == 'cnn':
        toolarge = True
        if name_data == 'cifar10':
            if larger == True:
                net = ProbCNNLarger(rho_prior, prior_dist=prior_dist,
                                    device=device, init_net=net0).to(device)
            else:
                net = ProbCNNLarge(rho_prior, prior_dist=prior_dist,
                                   device=device, init_net=net0).to(device)
        else:
            net = ProbCNN(rho_prior, prior_dist=prior_dist,
                          device=device, init_net=net0).to(device)
    elif model == 'fcn':
        net = ProbNN(rho_prior, prior_dist=prior_dist,
                     device=device, init_net=net0).to(device)

    bound = PBBound(objective, pmin, classes, train_size, delta,
                    delta_test, mc_samples, bbb_penalty, device)

    if objective == 'flamb':
        lambda_var = Lambda_var(initial_lamb, train_size).to(device)
        optimizer_lambda = optim.SGD(
            lambda_var.parameters(), lr=learning_rate, momentum=momentum)
    else:
        optimizer_lambda = None
        lambda_var = None
    optimizer = optim.SGD(
        net.parameters(), lr=learning_rate, momentum=momentum)

    pbkl01_train = 1.0

    for epoch in trange(train_epochs):
        trainPNN(net, optimizer, bound, epoch,
                 train_loader, lambda_var, optimizer_lambda, verbose, pbkl01_train)
        if verbose_test:
            if (epoch % 10 == 0):
                train_obj_rub_train, pbklbce_train, pbkl01_train, kl, loss_ce_train, err_01_train = computeFinalMetrics(net, toolarge, bound, device=device, lambda_var=lambda_var,
                                                                                                                        train_loader=val_bound, whole_train=val_bound_one_batch)
                stch_loss, stch_err = testStochastic(
                    net, test_loader, bound, device=device)
                post_loss, post_err = testPosterior(
                    net, test_loader, bound, device=device)
                ens_loss, ens_err = testEnsemble(net, test_loader, bound,
                                                 device=device, samples=samples_ensemble)
                print(f"Sigma, pmin, LR, momentum, LR_prior, momentum_prior, bbb_penalty, dropout, RUB_train, PBKLB_CE, PBKLB_01, KL, Train NLL loss, Train error, Stch loss, Stch error, Post loss, post error, ens loss, ens error, error net0, noise_targets, perc_train, perc_prior")
                print(f"{sigma_prior :.5f}, {pmin :.5f}, {learning_rate :.5f}, {momentum :.5f}, {learning_rate_prior :.5f}, {momentum_prior :.5f}, {bbb_penalty : .5f}, {dropout_prob :.5f}, {train_obj_rub_train :.5f}, {pbklbce_train :.5f}, {pbkl01_train :.5f}, {kl :.5f}, {loss_ce_train :.5f}, {err_01_train :.5f}, {stch_loss :.5f}, {stch_err :.5f}, {post_loss :.5f}, {post_err :.5f}, {ens_loss :.5f}, {ens_err :.5f}, {errornet0 :.5f}, {noise_targets :.5f}, {perc_train :.5f}, {perc_prior :.5f}")

    train_obj_rub_train, pbklbce_train, pbkl01_train, kl, loss_ce_train, err_01_train = computeFinalMetrics(net, toolarge, bound, device=device, lambda_var=lambda_var,
                                                                                                            train_loader=val_bound, whole_train=val_bound_one_batch)
    stch_loss, stch_err = testStochastic(
        net, test_loader, bound, device=device)
    post_loss, post_err = testPosterior(
        net, test_loader, bound, device=device)
    ens_loss, ens_err = testEnsemble(net, test_loader, bound,
                                     device=device, samples=samples_ensemble)
    print(f"Sigma, pmin, LR, momentum, LR_prior, momentum_prior, bbb_penalty, dropout, RUB_train, PBKLB_CE, PBKLB_01, KL, Train NLL loss, Train error, Stch loss, Stch error, Post loss, post error, ens loss, ens error, error net0, noise_targets, perc_train, perc_prior")
    print(
        f"{sigma_prior :.5f}, {pmin :.5f}, {learning_rate :.5f}, {momentum :.5f}, {learning_rate_prior :.5f}, {momentum_prior :.5f}, {bbb_penalty : .5f}, {dropout_prob :.5f}, {train_obj_rub_train :.5f}, {pbklbce_train :.5f}, {pbkl01_train :.5f}, {kl :.5f}, {loss_ce_train :.5f}, {err_01_train :.5f}, {stch_loss :.5f}, {stch_err :.5f}, {post_loss :.5f}, {post_err :.5f}, {ens_loss :.5f}, {ens_err :.5f}, {errornet0 :.5f}, {noise_targets :.5f}, {perc_train :.5f}, {perc_prior :.5f}")


def count_parameters(model): return sum(p.numel()
                                        for p in model.parameters() if p.requires_grad)
