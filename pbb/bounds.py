import math
import numpy as np
import torch
import torch.distributions as td
from tqdm import tqdm, trange
import torch.nn.functional as F


class PBBound():
    # PAC-Bayes bound class
    def __init__(self, objective='fquad', pmin=1e-4, classes=10, train_size=50000, delta=0.025, delta_test=0.01, mc_samples=1000, bbb_penalty=0.1, device='cuda'):
        super().__init__()
        self.objective = objective
        self.pmin = pmin
        self.classes = classes
        self.device = device
        self.train_size = train_size
        self.delta = delta
        self.delta_test = delta_test
        self.mc_samples = mc_samples
        self.bbb_penalty = bbb_penalty

    def compute_empirical_risk(self, outputs, targets, bounded=True):
        empirical_risk = F.nll_loss(outputs, targets)
        if bounded == True:
            empirical_risk = (1./(np.log(1./self.pmin))) * empirical_risk
        return empirical_risk

    def compute_losses(self, net, data, target, clamping=True):
        outputs = net(data, sample=True,
                      clamping=clamping, pmin=self.pmin)
        loss_ce = self.compute_empirical_risk(
            outputs, target, clamping)
        pred = outputs.max(1, keepdim=True)[1]
        correct = pred.eq(
            target.view_as(pred)).sum().item()
        total = target.size(0)
        loss_01 = 1-(correct/total)
        return loss_ce, loss_01, outputs

    def bound(self, empirical_risk, kl, lambda_var=None, rub01=1.0):
        if self.objective == 'fquad':
            kl = kl * self.bbb_penalty
            repeated_kl_ratio = torch.div(
                kl + np.log((2*np.sqrt(self.train_size))/self.delta), 2*self.train_size)
            first_term = torch.sqrt(
                empirical_risk + repeated_kl_ratio)
            second_term = torch.sqrt(repeated_kl_ratio)
            train_obj = torch.pow(first_term + second_term, 2)
        elif self.objective == 'flamb':
            kl = kl * self.bbb_penalty
            lamb = lambda_var.lamb_scaled
            kl_term = torch.div(
                kl + np.log((2*np.sqrt(self.train_size)) / self.delta), self.train_size*lamb*(1 - lamb/2))
            first_term = torch.div(empirical_risk, 1 - lamb/2)
            train_obj = first_term + kl_term
        elif self.objective == 'fclassic':
            kl = kl * self.bbb_penalty
            kl_ratio = torch.div(
                kl + np.log((2*np.sqrt(self.train_size))/self.delta), 2*self.train_size)
            train_obj = empirical_risk + torch.sqrt(kl_ratio)
        elif self.objective == 'bbb':
            # ipdb.set_trace()
            train_obj = empirical_risk + \
                self.bbb_penalty * (kl/self.train_size)
        else:
            assert False
        return train_obj

    def mcsampling(self, net, input, target, batches=True, clamping=True, data_loader=None):
        error = 0.0
        cross_entropy = 0.0
        if batches:
            for batch_id, (data_batch, target_batch) in enumerate(tqdm(data_loader)):
                data_batch, target_batch = data_batch.to(
                    self.device), target_batch.to(self.device)
                cross_entropy_mc = 0.0
                error_mc = 0.0
                for i in range(self.mc_samples):
                    loss_ce, loss_01, _ = self.compute_losses(net,
                                                              data_batch, target_batch, clamping)
                    cross_entropy_mc += loss_ce
                    error_mc += loss_01
                # we average cross-entropy and 0-1 error over all MC samples
                cross_entropy += cross_entropy_mc/self.mc_samples
                error += error_mc/self.mc_samples
            # we average cross-entropy and 0-1 error over all batches
            cross_entropy /= batch_id
            error /= batch_id
        else:
            cross_entropy_mc = 0.0
            error_mc = 0.0
            for i in range(self.mc_samples):
                loss_ce, loss_01, _ = self.compute_losses(net,
                                                          input, target, clamping)
                cross_entropy_mc += loss_ce
                error_mc += loss_01
                # we average cross-entropy and 0-1 error over all MC samples
            cross_entropy += cross_entropy_mc/self.mc_samples
            error += error_mc/self.mc_samples
        return cross_entropy, error

    def train_obj(self, net, input, target, clamping=True, lambda_var=None, rub01=1.0):
        outputs = torch.zeros(target.size(0), self.classes).to(self.device)
        kl = net.compute_kl()
        loss_ce, loss_01, outputs = self.compute_losses(net,
                                                        input, target, clamping)

        train_obj = self.bound(loss_ce, kl, lambda_var, rub01=rub01)
        return train_obj, kl/self.train_size, outputs, loss_ce, loss_01

    def computePBkl01(self, net, input=None, target=None, data_loader=None, clamping=True, lambda_var=None):
        kl = net.compute_kl()
        net.eval()
        with torch.no_grad():
            if data_loader:
                _, error_01 = self.mcsampling(net, input, target, batches=True,
                                              clamping=True, data_loader=data_loader)
            else:
                _, error_01 = self.mcsampling(net, input, target, batches=False,
                                              clamping=True)
            empirical_risk_01 = inv_kl(
                error_01, np.log(2/self.delta_test)/self.mc_samples)

            rubpbkl_01 = inv_kl(empirical_risk_01, (kl + np.log((2 *
                                                                 np.sqrt(self.train_size))/self.delta_test))/self.train_size)
        return rubpbkl_01

    def compute_final_bounds(self, net, input=None, target=None, data_loader=None, clamping=True, lambda_var=None):
        kl = net.compute_kl()
        if data_loader:
            error_ce, error_01 = self.mcsampling(net, input, target, batches=True,
                                                 clamping=True, data_loader=data_loader)
        else:
            error_ce, error_01 = self.mcsampling(net, input, target, batches=False,
                                                 clamping=True)

        empirical_risk_ce = inv_kl(
            error_ce.item(), np.log(2/self.delta_test)/self.mc_samples)
        empirical_risk_01 = inv_kl(
            error_01, np.log(2/self.delta_test)/self.mc_samples)

        train_obj = self.bound(empirical_risk_ce, kl, lambda_var)

        rubpbkl_ce = inv_kl(empirical_risk_ce, (kl + np.log((2 *
                                                             np.sqrt(self.train_size))/self.delta_test))/self.train_size)
        rubpbkl_01 = inv_kl(empirical_risk_01, (kl + np.log((2 *
                                                             np.sqrt(self.train_size))/self.delta_test))/self.train_size)
        return train_obj.item(), kl.item()/self.train_size, empirical_risk_ce, empirical_risk_01, rubpbkl_ce, rubpbkl_01


def inv_kl(qs, kl):
    # computation of the inversion of the binary KL
    qd = 0
    ikl = 0
    izq = qs
    dch = 1-1e-10
    while((dch-izq)/dch >= 1e-5):
        p = (izq+dch)*.5
        if qs == 0:
            ikl = kl-(0+(1-qs)*math.log((1-qs)/(1-p)))
        elif qs == 1:
            ikl = kl-(qs*math.log(qs/p)+0)
        else:
            ikl = kl-(qs*math.log(qs/p)+(1-qs) * math.log((1-qs)/(1-p)))
        if ikl < 0:
            dch = p
        else:
            izq = p
        qd = p
    return qd
