from typing import Tuple
import numpy as np
from adaptive_baselines.samplers.svgd import OptimizationSVGDSampler
from adaptive_baselines.samplers.steinpoints import OptimizationSteinPointsSampler, SteinPointsSampler
from adaptive_baselines.samplers.kernels import rbf_kernel, rbf_kernel_mahalanobis
from scipy.stats import multivariate_normal
from sprl.distributions.kl_joint import KLGaussian, KLJoint, KLPolicy
from functools import partial


class SteinPointsGaussian(KLGaussian):

    def __init__(self, lower_bounds, upper_bounds, mu, sigma):
        super().__init__(lower_bounds, upper_bounds, mu, sigma)
        self._samples = None
        self._sampler = OptimizationSteinPointsSampler(
            kernel=rbf_kernel_mahalanobis, verbose=True)
        self._kernel_args = dict(bandwidth=1.)
        self._return_last = False
        self._old_sample_ratio = 0.5


    def sample(self, num_samples=1, aux_samples=-1):
        if self._samples is None:
            print("Complete resample")
            return np.array(super().sample(num_samples))

        elif self._return_last:
            self._return_last = False
            return self._samples
        else:
            dis = multivariate_normal(self.mu, self.sigma)
            _, self._samples = self._sampler.sample(
                dis,
                self._samples,
                num_samples,
                self._samples.shape[0],
                old_sample_ratio=self._old_sample_ratio,
                return_splitted=True,
                kernel_args=self._kernel_args)
            return self._samples

    def set_buffer_values(self, values: np.ndarray):
        self._samples = values

    def prepare_buffer_with_preselected_values(self, values: np.ndarray, num_samples: int, old_sample_ratio: float) -> Tuple[np.ndarray, int]:
        dis = multivariate_normal(self.mu, self.sigma)
        self._kernel_args["cov"] = np.linalg.inv(self.sigma)
        old_sample_selection, aux = self._sampler.sample(
            dis,
            values,
            num_samples,
            values.shape[0] * 2,
            return_splitted=True,
            old_sample_ratio=old_sample_ratio,
            kernel_args=self._kernel_args)
        self._samples = aux.copy()
        return old_sample_selection, aux.shape[0]

    def clear_sample_buffer(self):
        self._samples = None

    def __getstate__(self):
        # For some reason, the sampler is not pickleable,
        # so we need to remove it.
        state = self.__dict__.copy()
        del state['_sampler']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._sampler = OptimizationSteinPointsSampler(
            kernel=rbf_kernel_mahalanobis, verbose=True)

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
            print(
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

        print(f"Using sampler type: {svgd_type}")
        if svgd_type == 'prune_old':
            self.distribution = SVGDPruningKLGaussian(lower_bounds_x,
                                                      upper_bounds_x, mu_x,
                                                      sigma_x)
        elif svgd_type == 'simple':
            self.distribution = SteinPointsGaussian(lower_bounds_x, upper_bounds_x,
                                               mu_x, sigma_x)
        else:
            self.distribution = KLGaussian(lower_bounds_x, upper_bounds_x,
                                           mu_x, sigma_x)
        self.policy = KLPolicy(lower_bounds_y, upper_bounds_y, mu_y, sigma_y,
                               feature_func)

        self.epsilon = epsilon
        self.max_eta = max_eta


class SVGDPruningKLGaussian(SteinPointsGaussian):
    def __init__(self, lower_bounds, upper_bounds, mu, sigma, prune_amount=10):
        super().__init__(lower_bounds, upper_bounds, mu, sigma)
        self._prune_amount = prune_amount

    def sample(self, num_samples=1):
        if self._samples is not None:
            self._samples = self._samples[self._prune_amount:]
        return super().sample(num_samples=num_samples)
