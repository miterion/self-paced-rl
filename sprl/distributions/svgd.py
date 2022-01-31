import numpy as np
from adaptive_baselines.samplers.svgd import BandwidthHeuristic, OptimizationSVGDSampler, VanillaSVGDSampler
from scipy.stats import multivariate_normal
from sprl.distributions.kl_joint import KLGaussian, KLJoint, KLPolicy


class SVGDKLGaussian(KLGaussian):

    def __init__(self, lower_bounds, upper_bounds, mu, sigma):
        super().__init__(lower_bounds, upper_bounds, mu, sigma)
        self._samples = None
        self._counter = 0
        self._keep_old_samples = False
        self._sampler = VanillaSVGDSampler(BandwidthHeuristic.HEMETHOD,
                                           stepsize=1e-1)
        

    def sample(self, num_samples=1, mask=None):
        if self._samples is None:
            print("Complete resample")
            return np.array(super().sample(num_samples))

        else:
            dis = multivariate_normal(self.mu, self.sigma)
            if num_samples > self._samples.shape[0]:
                self._samples = np.concatenate(
                    (self._samples,
                     dis.rvs(num_samples - self._samples.shape[0])))
            elif self._keep_old_samples:
                self._samples = np.concatenate(
                    (self._samples, dis.rvs(num_samples)))
                print(f"Keeping old samples, but adding {num_samples} new ones.")
                mask = np.full((self._samples.shape[0]), False)
                mask[-num_samples:] = True
            elif num_samples < self._samples.shape[0]:
                raise RuntimeError(
                    "SVGDKLGaussian: num_samples must (currently) be greater than the number of samples already stored"
                )
            if mask is not None:
                print(f"SVGDKLGaussian: Using mask")
                # self._samples = OptimizationSVGDSampler(8).sample_with_mask(
                #     dis,
                #     self._samples,
                #     bounds=(self.lower_bounds, self.upper_bounds),
                #     mask=mask)[mask]
                self._samples = self._sampler.sample_with_mask(dis,
                                                    self._samples,
                                                    n_iter=100,
                                                    bounds=(self.lower_bounds,
                                                            self.upper_bounds),
                                                    mask=mask)[1][mask]
            else:
                print(f"SVGDKLGaussian: NOT using mask")
                self._samples = OptimizationSVGDSampler(8).sample_with_bounds(
                    dis,
                    self._samples,
                    bounds=(self.lower_bounds, self.upper_bounds))
        self._counter += 1
        return self._samples

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value
        self._distribution_shift = True

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value
        self._distribution_shift = True

    def set_buffer_values(self, values: np.ndarray):
        self._samples = values
        self._keep_old_samples = True

    def clear_sample_buffer(self):
        self._samples = None
        self._keep_old_samples = False


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
            self.distribution = SVGDKLGaussian(lower_bounds_x, upper_bounds_x,
                                               mu_x, sigma_x)
        else:
            self.distribution = KLGaussian(lower_bounds_x, upper_bounds_x,
                                           mu_x, sigma_x)
        self.policy = KLPolicy(lower_bounds_y, upper_bounds_y, mu_y, sigma_y,
                               feature_func)

        self.epsilon = epsilon
        self.max_eta = max_eta


class SVGDPruningKLGaussian(SVGDKLGaussian):
    def __init__(self, lower_bounds, upper_bounds, mu, sigma, prune_amount=10):
        super().__init__(lower_bounds, upper_bounds, mu, sigma)
        self._prune_amount = prune_amount

    def sample(self, num_samples=1):
        if self._samples is not None:
            self._samples = self._samples[self._prune_amount:]
        return super().sample(num_samples=num_samples)
