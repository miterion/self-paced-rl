import logging
from functools import partial
from typing import Tuple

import numpy as np
from adaptive_baselines.samplers.kernels import (rbf_kernel,
                                                 rbf_kernel_mahalanobis)
from adaptive_baselines.samplers.steinpoints import (
    OptimizationSteinPointsSampler, SteinPointsSampler)
from adaptive_baselines.samplers.svgd import OptimizationSVGDSampler
from scipy.stats import multivariate_normal
from sprl.distributions.kl_joint import KLGaussian, KLJoint, KLPolicy

log = logging.getLogger(__name__)


class SteinPointsGaussian(KLGaussian):
    def __init__(self, lower_bounds, upper_bounds, mu, sigma):
        super().__init__(lower_bounds, upper_bounds, mu, sigma)
        self._samples = None
        self._kernel_args = dict(bandwidth=1.)
        self._return_last = False

    def sample(self, num_samples=1, aux_samples=-1):
        if self._samples is None:
            log.debug("Complete resample")
            return np.array(super().sample(num_samples))

        elif self._return_last:
            self._return_last = False
            return self._samples
        else:
            raise NotImplementedError(
                'Called sample without setting return last or clearing sample buffer'
            )

    def clear_sample_buffer(self):
        self._samples = None

    def __getstate__(self):
        # For some reason, the sampler is not pickleable,
        # so we need to remove it.
        state = self.__dict__.copy()
        del state['_sampler']
        return state

    def set_buffer_values(self, values: np.ndarray):
        self._samples = values

    def prepare_buffer_with_preselected_values(
            self, values: np.ndarray, num_samples: int,
            cfg: dict) -> Tuple[np.ndarray, int]:
        pass


class SteinPointsOptGaussian(SteinPointsGaussian):
    def __init__(self, lower_bounds, upper_bounds, mu, sigma):
        super().__init__(lower_bounds, upper_bounds, mu, sigma)
        self._sampler = OptimizationSteinPointsSampler(
            kernel=rbf_kernel_mahalanobis, verbose=False)

    def prepare_buffer_with_preselected_values(
            self, values: np.ndarray, num_samples: int,
            cfg: dict) -> Tuple[np.ndarray, int]:

        U, S, V = np.linalg.svd(self.sigma)
        sigma_stretched = U * (
            S * cfg.algorithm.sampler.proposal_stretch_factor) @ V
        dis = multivariate_normal(self.mu, self.sigma)
        proposal = multivariate_normal(self.mu, sigma_stretched)
        self._kernel_args["cov"] = np.linalg.inv(sigma_stretched)
        if cfg.algorithm.sampler.old_sample_ratio == 0.:
            if num_samples > values.shape[0]:
                old_sample_ratio = 1.
            else:
                old_sample_ratio = (values.shape[0] - cfg.env.n_samples) / values.shape[0]
        else:
            old_sample_ratio = cfg.algorithm.sampler.old_sample_ratio
        old_sample_selection, aux = self._sampler.sample_with_bound(
            new=dis,
            old_samples=values,
            n_samples=num_samples,
            bounds=(self.lower_bounds, self.upper_bounds),
            aux_samples_dist=proposal,
            aux_samples_amount=int(values.shape[0] *
                                   cfg.algorithm.sampler.aux_samples_factor),
            return_splitted=True,
            old_sample_ratio=old_sample_ratio,
            kernel_args=self._kernel_args)
        self._samples = aux.copy()
        return old_sample_selection, aux.shape[0]

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._sampler = OptimizationSteinPointsSampler(
            kernel=rbf_kernel_mahalanobis, verbose=False)


class SteinPointsIterGaussian(SteinPointsGaussian):
    def __init__(self, lower_bounds, upper_bounds, mu, sigma):
        super().__init__(lower_bounds, upper_bounds, mu, sigma)
        self._sampler = SteinPointsSampler(kernel=rbf_kernel_mahalanobis,
                                           verbose=False)

    def prepare_buffer_with_preselected_values(
            self, values: np.ndarray, num_samples: int,
            cfg: dict) -> Tuple[np.ndarray, int]:

        added_samples = int(values.shape[0] * cfg.aux_samples_factor)
        log.debug(f"Aux samples: {added_samples}, required samples {num_samples}")
        assert num_samples <= added_samples, "Not enough samples for iSP due to low aux_samples_factor"
        U, S, V = np.linalg.svd(self.sigma)
        sigma_stretched = U * (S * cfg.proposal_stretch_factor) @ V
        dis = multivariate_normal(self.mu, self.sigma)
        proposal = multivariate_normal(self.mu, sigma_stretched)
        self._kernel_args["cov"] = np.linalg.inv(sigma_stretched)
        old_sample_selection, aux = self._sampler.sample_with_bound(
            new=dis,
            old_samples=values,
            n_samples=num_samples,
            bounds=(self.lower_bounds, self.upper_bounds),
            aux_samples_dist=proposal,
            aux_samples_amount=added_samples,
            return_splitted=True,
            kernel_args=self._kernel_args)
        self._samples = aux.copy()
        return old_sample_selection, aux.shape[0]

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._sampler = SteinPointsSampler(kernel=rbf_kernel_mahalanobis,
                                           verbose=False)


class SVGDKLPolicy(KLPolicy):
    def __init__(self, lower_bounds, upper_bounds, mu_init, sigma_init,
                 feature_func):
        super().__init__(lower_bounds, upper_bounds, mu_init, sigma_init,
                         feature_func)
        self._samples = None
        self._distribution_shift = False

    def sample_action(self, state):
        # TODO: Squeeze the output
        if self._samples is None or not self._distribution_shift:
            self._samples = super().sample_action(state)
        else:
            mu = self.compute_greedy_action(state)
            sigma = self.compute_variance(state)
            dis = multivariate_normal(mu, sigma)
            self._samples = OptimizationSVGDSampler(3.).sample(
                dis, self._samples)
            if len(self._samples.shape) == 2 and self._samples.shape[0] == 1:
                self._samples = np.squeeze(self._samples)
        if np.any(self.lower_bounds > self._samples) or np.any(
                self._samples > self.upper_bounds):
            log.debug(
                f"Bounds were violated: {self._samples}\n Bounds: {self.lower_bounds} {self.upper_bounds}"
            )
        return self._samples

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        self._theta = value
        self._distribution_shift = True

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value
        self._distribution_shift = True


class SVGDJoint(KLJoint):
    def __init__(self,
                 lower_bounds_x,
                 upper_bounds_x,
                 mu_x,
                 sigma_x,
                 lower_bounds_y,
                 upper_bounds_y,
                 mu_y,
                 sigma_y,
                 feature_func,
                 epsilon,
                 max_eta=100,
                 svgd_type=None):

        log.debug(f"Using sampler type: {svgd_type}")
        if svgd_type == 'iterative':
            self.distribution = SteinPointsIterGaussian(
                lower_bounds_x, upper_bounds_x, mu_x, sigma_x)
        elif svgd_type == 'optimization':
            self.distribution = SteinPointsOptGaussian(lower_bounds_x,
                                                       upper_bounds_x, mu_x,
                                                       sigma_x)
        else:
            self.distribution = KLGaussian(lower_bounds_x, upper_bounds_x,
                                           mu_x, sigma_x)
        self.policy = KLPolicy(lower_bounds_y, upper_bounds_y, mu_y, sigma_y,
                               feature_func)

        self.epsilon = epsilon
        self.max_eta = max_eta