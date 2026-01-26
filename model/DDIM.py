
import torch
import torch.nn as nn
import torchvision
from torch.nn import init
from torch.nn import functional as F
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage.interpolation import rotate
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial
import random
import os


class DiffusionProcessDDIM():
    def __init__(self, beta_1, beta_T, T, c, w, diffusion_fn, device, shape, eta, tau=1, scheduling='uniform'):
        '''
        beta_1        : beta_1 of diffusion process
        beta_T        : beta_T of diffusion process
        T             : step of diffusion process
        c             : The sample condition
        w             : The weight of sample condition
        diffusion_fn  : trained diffusion network
        shape         : data shape
        eta           : coefficient of sigma
        tau           : accelerating of diffusion process
        scheduling    : scheduling mode of diffusion process
        '''
        self.betas = torch.linspace(start=beta_1, end=beta_T, steps=T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start=beta_1, end=beta_T, steps=T), dim=0).to(device=device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])

        self.shape = shape
        self.sigmas = torch.sqrt((1 - self.alpha_prev_bars) / (1 - self.alpha_bars)) * torch.sqrt(
            1 - (self.alpha_bars / self.alpha_prev_bars))
        self.diffusion_fn = diffusion_fn
        self.device = device
        self.T = T
        self.c = c
        self.w = w
        self.eta = eta
        self.tau = tau
        self.scheduling = scheduling

    def _get_process_scheduling(self, reverse=True):
        if self.scheduling == 'uniform':
            diffusion_process = list(range(0, len(self.alpha_bars), self.tau)) + [len(self.alpha_bars) - 1]
        elif self.scheduling == 'exp':
            diffusion_process = (np.linspace(0, np.sqrt(len(self.alpha_bars) * 0.8), self.tau) ** 2)
            diffusion_process = [int(s) for s in list(diffusion_process)] + [len(self.alpha_bars) - 1]
        else:
            assert 'Not Implementation'

        diffusion_process = zip(reversed(diffusion_process[:-1]), reversed(diffusion_process[1:])) if reverse else zip(
            diffusion_process[1:], diffusion_process[:-1])
        return diffusion_process

    def _one_reverse_diffusion_step(self, x, n_sample):
        '''
        x   : perturbated data
        '''
        diffusion_process = self._get_process_scheduling(reverse=True)
        c_i = self.c.repeat(n_sample, 1)
        context_mask = torch.zeros(n_sample, 1).to(self.device)

        # context_mask = torch.ones(n_sample, 1).to(self.device)

        # double the batch

        for prev_idx, idx in diffusion_process:
            t_is = torch.tensor([idx / self.T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            self.diffusion_fn.eval()
            noise = torch.zeros(n_sample, *self.shape).to(self.device) if idx == 0 else torch.randn(n_sample, *self.shape).to(self.device)

            eps = self.diffusion_fn.nn_model(x, c_i, t_is, context_mask)

            predict_epsilon = eps
            # predict_epsilon = eps2

            #predict_epsilon = self.diffusion_fn(x, idx)
            sigma = self.sigmas[idx] * self.eta

            predicted_x0 = torch.sqrt(self.alpha_bars[prev_idx]) * (
                        x - torch.sqrt(1 - self.alpha_bars[idx]) * predict_epsilon) / torch.sqrt(self.alpha_bars[idx])
            direction_pointing_to_xt = torch.sqrt(1 - self.alpha_bars[prev_idx] - sigma ** 2) * predict_epsilon
            x = predicted_x0 + direction_pointing_to_xt + sigma * noise

            yield x

    @torch.no_grad()
    def sampling(self, n_sample, sample=None, only_final=False):
        '''
        sampling_number : a number of generation
        only_final      : If True, return is an only output of final schedule step
        '''

        if sample == None:
            sample = torch.randn([n_sample, *self.shape]).to(device=self.device)

        sampling_list = []

        final = None
        for sample in self._one_reverse_diffusion_step(sample, n_sample):
            final = sample
            if not only_final:
                sampling_list.append(final)

        return final if only_final else torch.stack(sampling_list)

    @torch.no_grad()
    def probabilityflow(self, x, reverse):
        '''
        reverse : if True, backward(noise -> data) else forward(data -> noise)
        '''

        def reparameterize_sigma(idx):
            return torch.sqrt((1 - self.alpha_bars[idx]) / self.alpha_bars[idx])

        def reparameterize_x(x, idx):
            return x / torch.sqrt(self.alpha_bars[idx])

        diffusion_process = self._get_process_scheduling(reverse=reverse)
        for idx_delta_t, idx in diffusion_process:
            self.diffusion_fn.eval()
            x_bar_delta_t = reparameterize_x(x, idx) + 0.5 * (
                        reparameterize_sigma(idx_delta_t) ** 2 - reparameterize_sigma(idx) ** 2) / reparameterize_sigma(
                idx) * self.diffusion_fn(x, idx)
            x = x_bar_delta_t * torch.sqrt(self.alpha_bars[idx_delta_t])

        return x