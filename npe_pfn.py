# Adapted code from https://github.com/mackelab/npe-pfn
import math
import logging
from tqdm.auto import tqdm
from typing import Literal, Mapping, Optional, Callable, Dict, Tuple, Any

import torch
from torch import Tensor
from torch.distributions import Distribution, Independent, Uniform
from sbi.utils import BoxUniform
from PFNs.pfns.model.transformer import TableTransformer

log = logging.getLogger(__name__)


class NPE_PFN_Core:
    """TabPFN-based simulation-based inference that follows SBI-like interface.

    This class provides similar functionality to SBI's NPE (Neural Posterior Estimation)
    but uses TabPFN as the underlying model.
    """

    def __init__(
        self,
        model: TableTransformer,
        show_progress_bars: bool = False,
        prior: Optional[Distribution] = None,
        embedding_net: Optional[torch.nn.Module] = None,
        x_shape: Optional[torch.Size] = None,
        regressor_init_kwargs: Mapping = {},
        classifier_init_kwargs: Mapping = {},
    ) -> None:
        """Initialize TabPFN-based inference."""
        self.show_progress_bars = show_progress_bars
        self.prior = prior
        self.regressor_init_kwargs = regressor_init_kwargs
        self.classifier_init_kwargs = classifier_init_kwargs

        self._model = model
        self._model_classifier = None
        self.embedding_net = embedding_net
        self.x_shape = x_shape

        # Initialize theta, x for storage of parameters and simulations
        self._theta_train = None
        self._x_train = None

    def __getstate__(self):
        """Prepare the object state for pickling."""
        state = self.__dict__.copy()
        # Remove the model and classifier from the state
        state["_model"] = None
        state["_model_classifier"] = None
        return state

    def __setstate__(self, state):
        """Restore the object state after unpickling."""
        self.__dict__.update(state)
        # Reinitialize the model and classifier
        self._model = TableTransformer(**self.regressor_init_kwargs)
        if self._model_classifier is not None:
            self._model_classifier = DensityRatioWrapper(**self.classifier_init_kwargs)

    def append_simulations(self, theta: Tensor, x: Tensor):
        """Append new simulation outputs to training data."""
        self._theta_train = None
        self._x_train = None
        if self.embedding_net:
            x = x.reshape(-1, *self.x_shape)
            x = self.embedding_net(x)
        self._theta_train = self._validate_theta(theta)
        self._x_train = self._validate_x(x)
        return self

    def get_context(self, x: Tensor):
        """Get context used for observation."""
        return self._theta_train, self._x_train

    def _validate_x(self, x: Tensor):
        """Validate x."""
        if x is None:
            raise NotImplementedError("Setting a default x is not yet supported.")

        x = x.unsqueeze(0) if x.ndim == 1 else x
        assert x.ndim == 2, "x must be a 2D tensor."
        if self._x_train is not None:
            assert (
                x.shape[1] == self._x_train.shape[1]
            ), "The number of features in x must match the training data."
        return x

    def _validate_theta(self, theta: Tensor):
        """Validate theta."""
        theta = theta.unsqueeze(0) if theta.ndim == 1 else theta
        assert theta.ndim == 2, "theta must be a 2D tensor."
        if self._theta_train is not None:
            assert (
                theta.shape[1] == self._theta_train.shape[1]
            ), "The number of features in theta must match the training data."
        return theta

    def _sample(
        self,
        sampling_batch_size: int,
        x: Tensor,
        repeat_x: bool = True,
        with_log_prob: bool = False,
        eps=1e-15,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Sample from the posterior p(theta | x)"""

        if repeat_x:
            samples_batch = x.repeat(sampling_batch_size, 1)
        else:
            samples_batch = x
            sampling_batch_size = x.shape[0]

        # Create joint dataset of observations and parameters
        theta_context, x_context = self.get_context(x)
        joint_data = torch.cat([x_context, theta_context], dim=1)
        dim_x = x_context.shape[1]
        dim_theta = theta_context.shape[1]

        log_probs_batch = torch.zeros(sampling_batch_size) if with_log_prob else None
        # Sequentially predict each parameter dimension
        for param_idx in range(dim_theta):
            # Fit model on joint data up to current parameter
            features_end = dim_x + param_idx
            target_idx = dim_x + param_idx

            # self._model.fit(joint_data[:, :features_end], joint_data[:, target_idx])

            # Generate samples for current parameter
            logits = self._model(
                joint_data[None, :, :features_end], joint_data[None, :, target_idx], samples_batch[None]
            ).squeeze(0)
            param_samples = self._model.criterion.sample(logits)

            if with_log_prob:
                dim_log_prob = -self._model.criterion(
                    logits, param_samples
                )

                dim_log_prob = torch.where(
                    dim_log_prob == float("-inf"),
                    torch.log(torch.tensor(eps)),
                    dim_log_prob,
                )

                log_probs_batch += dim_log_prob

            # Append new parameter samples
            samples_batch = torch.cat([samples_batch, param_samples[:, None]], dim=1)

            # Clear cache to avoid memory issues, otherwise can segmentation fault
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        return samples_batch[:, dim_x:], log_probs_batch

    def sample(
        self,
        sample_shape: torch.Size = torch.Size(),
        x: Tensor = None,
        max_sampling_batch_size: int = 10_000,
        with_log_prob: bool = False,
        eps=1e-15,
        max_iter_rejection: int | None = None,
        show_progress_bars: bool = False,  # TODO deal with this
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Sample from the posterior p(theta | x)

        Args:
            sample_shape: Desired shape of samples
            x: Observations to condition on
            max_sampling_batch_size: Maximum batch size for sampling
        """

        # standard input checks
        if self.embedding_net:
            x = x.reshape(-1, *self.x_shape)
            x = self.embedding_net(x)

        x = self._validate_x(x)

        if x.shape[0] > 1:
            raise ValueError(
                ".sample() supports only `batchsize == 1`. If you intend "
                "to sample multiple observations, use `.sample_batched()`. "
            )

        def proposal_fn(max_sampling_batch_size, **kwargs):
            """Generate proposal samples using the original sampling method"""
            return self._sample(
                sampling_batch_size=max_sampling_batch_size,
                x=x,
                repeat_x=True,
                with_log_prob=with_log_prob,
                eps=eps,
            )

        # handles rejection and batching
        samples, log_probs, _ar = accept_reject_sample(
            proposal=proposal_fn,
            accept_reject_fn=self._within_support,
            num_samples=torch.Size(sample_shape).numel(),
            show_progress_bars=self.show_progress_bars,
            max_sampling_batch_size=max_sampling_batch_size,
            proposal_sampling_kwargs={},
            max_iter_rejection=max_iter_rejection,
        )

        if with_log_prob:
            return samples, log_probs
        else:
            return samples

    def sample_batched(
        self,
        x: Tensor,
        sample_shape: torch.Size = torch.Size(),
        max_sampling_batch_size: int = 10_000,
    ):
        """Sample from the posterior p(theta | x) in a batched manner.

        Args:
            x: Observations to condition on
            sample_shape: Desired shape of samples
            max_sampling_batch_size: Maximum batch size for sampling
        """
        raise NotImplementedError

    def log_prob(
        self,
        theta: Tensor,
        x: Tensor,
        max_sampling_batch_size: int = 10_000,
        mode="autoregressive",
        eps=1e-15,
        **ratio_kwargs,
    ):
        """Calculate log probability of parameters p(theta | x)

        Args:
            theta: Parameters to evaluate
            x: Observations to condition on
            mode: Method to use for log probability calculation
        """
        if self.embedding_net:
            x = x.reshape(-1, *self.x_shape)
            x = self.embedding_net(x)

        theta = self._validate_theta(theta)
        x = self._validate_x(x)

        log_probs = torch.zeros(theta.shape[0])
        for i in range(0, theta.shape[0], max_sampling_batch_size):
            if mode == "autoregressive":
                log_probs[i : i + max_sampling_batch_size] = (
                    self._autoregressive_log_prob(
                        theta[i : i + max_sampling_batch_size],
                        x,
                        eps=eps,
                    )
                )
            elif mode == "ratio_based":
                log_probs[i : i + max_sampling_batch_size] = self._ratio_based_log_prob(
                    theta[i : i + max_sampling_batch_size],
                    x,
                    eps=eps,
                    **ratio_kwargs,
                )
            else:
                raise ValueError(f"Invalid mode: {mode}")

        return log_probs

    def log_prob_batched(self, theta: Tensor, x: Tensor):
        """Calculate log probability of parameters in a batched manner."""
        # NOTE: Will only support autoregressive log prob
        raise NotImplementedError

    def _autoregressive_log_prob(
        self,
        theta: Tensor,
        x: Tensor = None,
        repeat_x: bool = True,
        eps: float = 1e-15,
    ) -> Tensor:
        """Calculate log probability of parameters p(theta | x)

        Args:
            theta: Parameters to evaluate
            x: Observations to condition on
        """
        # TODO leakage correction is not implemented, can have density outside of prior support

        # NOTE: repeat_x flag crucial for unconditional models
        num_samples = theta.shape[0]
        if repeat_x:
            x_batch = x.repeat(num_samples, 1)
        else:
            x_batch = x

        assert x_batch.shape[0] == num_samples
        test_joint = torch.cat([x_batch, theta], dim=1)

        # Create training joint data (observations and parameters)
        theta_context, x_context = self.get_context(x)
        joint_data = torch.cat([x_context, theta_context], dim=1)
        dim_x = x_context.shape[1]
        dim_theta = theta_context.shape[1]

        # Initialize log probability tensor
        log_prob = torch.zeros(num_samples)

        # Sequentially compute log prob for each parameter dimension
        for param_idx in range(dim_theta):
            # Fit model on joint data up to current parameter
            features_end = dim_x + param_idx
            target_idx = dim_x + param_idx

            # self._model.fit(joint_data[:, :features_end], joint_data[:, target_idx])

            # Get prediction distribution
            logits = self._model(
                test_joint[:, :features_end], output_type="full", quantiles=[]
            )

            # Compute log probability for this dimension
            dim_log_prob = -self._model.criterion(
                logits, test_joint[:, target_idx]
            )

            # Handle -inf values
            dim_log_prob = torch.where(
                dim_log_prob == float("-inf"),
                torch.log(torch.tensor(eps)),
                dim_log_prob,
            )

            # Add to total log probability
            log_prob += dim_log_prob

        return log_prob

    def _ratio_based_log_prob(
        self,
        theta: Tensor,
        x: Tensor = None,
        num_posterior_samples: int = 5000,
        boundary_padding: float = 0.1,
        reuse_estimator_if_possible: bool = True,
        eps: float = 1e-15,
    ) -> Tensor:
        """Calculate log probability of parameters using ratio-based method.

        Args:
            theta: Parameters to evaluate log prob for
            x: Observation to condition on
            num_posterior_samples: Number of posterior samples to generate (should be not more than 5000 under the normal limits)
            boundary_padding: Padding for uniform reference distribution
            eps: Small constant to avoid numerical issues
            reuse_estimator_if_possible: Reuse classifier if obsvervation was seen before.
                This will ignore the `num_posterior_samples` and `boundary_padding` arguments.
        """

        # initialize classifier if not already initialized
        if self._model_classifier is None:
            self._model_classifier = DensityRatioWrapper(**self.classifier_init_kwargs)

        # get actual context, might be different for filtering
        # NOTE: We need to be rather careful here if the sorting from the filter is not fully deterimistic/"stable"
        theta_context, x_context = self.get_context(x)
        if not reuse_estimator_if_possible or self._model_classifier.refit_necessary(
            x,
            x_context,
            theta_context,
            num_posterior_samples,
            boundary_padding,
        ):
            posterior_samples = self.sample(
                sample_shape=torch.Size([num_posterior_samples]), x=x
            )
            self._model_classifier.fit(
                x, posterior_samples, boundary_padding, x_context, theta_context
            )

        log_probs = self._model_classifier.ratio_log_probs(theta, eps)

        return log_probs

    def _get_classifier_bounds(self):
        """Get the bounds of the classifier if they exist."""
        if self._model_classifier is None:
            return None, None
        return (
            self._model_classifier._padded_dim_min,
            self._model_classifier._padded_dim_max,
        )

    def _within_support(self, theta: Tensor) -> Tensor:
        """Check if samples are within prior support.

        First attempts to use the support property of the prior distribution.
        Falls back to checking if log probability is finite if support check
        is not available.

        Args:
            theta: Parameter samples to check

        Returns:
            Tensor of bools indicating whether each sample is within support
        """
        try:
            sample_check = self.prior.support.check(theta)
            if sample_check.shape == theta.shape:
                sample_check = torch.all(sample_check, dim=-1)
            return sample_check
        except (NotImplementedError, AttributeError):
            return torch.isfinite(self.prior.log_prob(theta))


class DensityRatioWrapper:
    """Wrapper class for the density ratio based log probability calculation.
    This enables reuse of the classifier if the observation (and other parameters) are the same, which will be significantly faster.
    """

    def __init__(self, **init_kwargs):
        super().__init__()
        self._classifier = TableTransformer(**init_kwargs)

        self._ratio_log_prob_x = None
        self._num_posterior_samples = None
        self._boundary_padding = None

        self._padded_dim_min = None
        self._padded_dim_max = None
        self._uniform_log_prob = None

    def fit(
        self,
        x: Tensor,
        posterior_samples: Tensor,
        boundary_padding: float,
        x_context: Tensor,
        theta_context: Tensor,
    ):
        """Fit the classifier on the given data."""

        dim_min = posterior_samples.min(dim=0).values
        dim_max = posterior_samples.max(dim=0).values
        dim_length = dim_max - dim_min
        padded_dim_min = dim_min - boundary_padding * dim_length
        padded_dim_max = dim_max + boundary_padding * dim_length
        padded_dim_length = padded_dim_max - padded_dim_min

        uniform_log_prob = -torch.log(padded_dim_length).sum()

        # Generate uniform samples matching the shape of training data
        uniform_samples = (
            torch.rand_like(posterior_samples) * padded_dim_length + padded_dim_min
        )

        # Prepare classifier training data
        num_posterior_samples = posterior_samples.shape[0]
        train_X_classifier = torch.cat([uniform_samples, posterior_samples], dim=0)
        train_y_classifier = torch.cat(
            [torch.zeros(num_posterior_samples), torch.ones(num_posterior_samples)],
            dim=0,
        )

        self._ratio_log_prob_x = x
        self._num_posterior_samples = num_posterior_samples
        self._boundary_padding = boundary_padding
        self._x_context = x_context
        self._theta_context = theta_context

        self._padded_dim_min = padded_dim_min
        self._padded_dim_max = padded_dim_max
        self._uniform_log_prob = uniform_log_prob
        self._classifier.fit(train_X_classifier, train_y_classifier)

    def refit_necessary(
        self,
        x: Tensor,
        x_context: Tensor,
        theta_context: Tensor,
        num_posterior_samples: int,
        boundary_padding: float,
    ):
        """Reuse classifier if possible. Check whether refitting is necessary."""
        return (
            self._ratio_log_prob_x is None
            or not torch.allclose(x, self._ratio_log_prob_x)
            or not x_context.shape == self._x_context.shape
            or not torch.allclose(x_context, self._x_context)
            or not theta_context.shape == self._theta_context.shape
            or not torch.allclose(theta_context, self._theta_context)
            or not num_posterior_samples == self._num_posterior_samples
            or not math.isclose(boundary_padding, self._boundary_padding)
        )

    def ratio_log_probs(self, theta: Tensor, eps=1e-15):
        """Predict probabilities for the given theta."""
        mask = torch.all(
            (theta >= self._padded_dim_min) & (theta <= self._padded_dim_max), dim=1
        )

        log_probs = torch.full(
            (theta.shape[0],),
            self._uniform_log_prob
            + torch.log(torch.tensor(eps))
            - torch.log(torch.tensor(1 + eps)),
        )

        if mask.any():
            classifier_probs = self._classifier.predict_proba(theta[mask])
            log_probs[mask] = (
                self._uniform_log_prob
                + torch.log(torch.tensor(classifier_probs[:, 1] + eps))
                - torch.log(torch.tensor(classifier_probs[:, 0] + eps))
            )

        return log_probs


# NOTE: Can never support batched sampling, as context depends on x
class NPE_PFN(NPE_PFN_Core):
    def __init__(
        self,
        model: TableTransformer,
        show_progress_bars: bool = False,
        prior: Optional[Distribution] = None,
        filter_type: (
            Literal[
                "latest_filtering",
                "random_filtering",
                "standardized_euclidean_filtering",
            ]
            | callable
        ) = "standardized_euclidean_filtering",
        filter_context_size: int = 10_000,
        regressor_init_kwargs: Mapping = {},
        classifier_init_kwargs: Mapping = {},
        embedding_net: Optional[torch.nn.Module] = None,
        x_shape: Optional[torch.Size] = None,
    ):
        super().__init__(
            model,
            show_progress_bars,
            prior,
            regressor_init_kwargs=regressor_init_kwargs,
            classifier_init_kwargs=classifier_init_kwargs,
            embedding_net=embedding_net,
            x_shape=x_shape,
        )

        self.filter = get_filtering_method(filter_type)
        self.filter_context_size = filter_context_size

    def get_context(self, x: Tensor):
        x = self._validate_x(x)
        theta_context, x_context = self.filter(
            x, self._theta_train, self._x_train, self.filter_context_size
        )
        return theta_context, x_context


class PosteriorSupport:
    def __init__(
        self,
        prior: Any,
        posterior: Any,
        obs: Tensor,
        num_samples_to_estimate_support: int = 10_000,
        batch_size_for_estimate_support: int = 10_000,
        allowed_false_negatives: float = 0.0,
        sampling_method: str = "rejection",
        max_iter_rejection: int = 1000,
        oversample_sir: int = 100,
        log_prob_kwargs: Mapping = {},  # optionally pass some stuff depending on the posterior object
    ) -> None:

        self._prior = prior
        self._posterior = posterior
        self._obs = obs
        self._posterior_thr = None

        self.sampling_method = sampling_method
        self.max_iter = max_iter_rejection
        self.oversample_sir = oversample_sir
        self.allowed_false_negatives = allowed_false_negatives

        self._log_prob_kwargs = log_prob_kwargs

        # NOTE: reuse samples to get quantile and constrained support to save time
        if sampling_method == "rejection":
            samples_to_estimate_support = self._posterior.sample(
                (num_samples_to_estimate_support,),
                self._obs,
                max_sampling_batch_size=batch_size_for_estimate_support,
            )

            self.thr = self.tune_threshold(
                samples_to_estimate_support,
                allowed_false_negatives,
                batch_size=batch_size_for_estimate_support,
            )

    def tune_threshold(
        self,
        samples: Tensor,
        allowed_false_negatives: float = 0.0,
        batch_size: int = 10_000,
    ) -> None:

        log_probs = self._posterior.log_prob(
            samples,
            self._obs,
            max_sampling_batch_size=batch_size,
            **self._log_prob_kwargs,
        )

        # any reason why not torch.quantile previously?
        return torch.quantile(log_probs, allowed_false_negatives)

    def sample(
        self,
        sample_shape: torch.Size = torch.Size(),
        show_progress_bars: bool = True,  # True for now because one cannot set it within `simulate_for_sbi`
        sampling_batch_size: int = 10_000,
        return_acceptance_rate: bool = False,  # TODO could be something like return diagnostic
        return_ess: bool = False,
    ) -> Tensor:

        if self.sampling_method == "rejection":
            return self.sample_rejection(
                sample_shape=sample_shape,
                show_progress_bars=show_progress_bars,
                sampling_batch_size=sampling_batch_size,
                return_acceptance_rate=return_acceptance_rate,
            )
        elif self.sampling_method == "sir":
            return self.sample_sir(
                sample_shape=sample_shape,
                show_progress_bars=show_progress_bars,
                sampling_batch_size=sampling_batch_size,
                return_ess=return_ess,
            )
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")

    def sample_rejection(
        self,
        sample_shape: torch.Size = torch.Size(),
        show_progress_bars: bool = True,  # True for now because one cannot set it within `simulate_for_sbi`
        sampling_batch_size: int = 10_000,
        return_acceptance_rate: bool = False,
    ) -> Tensor:
        """
        Return samples from the `RestrictedPrior`.
        Samples are obtained by sampling from the prior, evaluating them under the
        trained classifier (`RestrictionEstimator`) and using only those that were
        accepted.
        Args:
            sample_shape: Shape of the returned samples.
            show_progress_bars: Whether or not to show a progressbar during sampling.
            max_sampling_batch_size: Batch size for drawing samples from the posterior.
            return_acceptance_rate: Whether to return the acceptance rate.
        Returns:
            Samples from the `RestrictedPrior`.
        """

        # There is no reason to support other shapes here
        assert len(sample_size := torch.Size(sample_shape)) == 1
        num_samples = sample_size[0]

        pbar = tqdm(
            disable=not show_progress_bars,
            total=num_samples,
            desc=f"Drawing {num_samples} restricted posterior samples",
        )

        # minimal supported acceptance is num_samples / (max_iter * sampling_batch_size)
        pre_acceptance_rate = 1.0  # overwriting continously is fine
        lower, upper = None, None
        num_sampled_total, num_remaining = 0, num_samples
        accepted = []
        for _ in range(self.max_iter):
            if num_remaining <= 0:
                break

            if lower is None or upper is None:
                candidates = self._prior.sample((sampling_batch_size,))
                log_probs = self._posterior.log_prob(
                    candidates, self._obs, **self._log_prob_kwargs
                )
                lower, upper = self._posterior._get_classifier_bounds()
            else:
                candidates, pre_acceptance_rate = prereject_with_bounds(
                    self._prior, lower, upper, sampling_batch_size
                )
                log_probs = self._posterior.log_prob(
                    candidates, self._obs, **self._log_prob_kwargs
                )
                sanity_lower, sanity_upper = self._posterior._get_classifier_bounds()
                assert torch.allclose(lower, sanity_lower)
                assert torch.allclose(upper, sanity_upper)

            are_accepted_by_classifier = log_probs > self.thr
            samples = candidates[are_accepted_by_classifier.bool()]
            accepted.append(samples)

            num_sampled_total += sampling_batch_size
            num_remaining -= samples.shape[0]
            pbar.update(samples.shape[0])

        pbar.close()

        acceptance_rate = (num_samples - num_remaining) / num_sampled_total

        log.info(f"Pre-acceptance rate: {pre_acceptance_rate}")
        log.info(f"Log prob acceptance rate: {acceptance_rate}")
        overall_acceptance_rate = pre_acceptance_rate * acceptance_rate
        log.info(f"Overall acceptance rate: {overall_acceptance_rate}")

        if num_remaining > 0:
            remaining_samples = self._prior.sample((num_remaining,))
            accepted.append(remaining_samples)
            log.info(f"Max iter exceeded. Added {remaining_samples} prior samples.")

        samples = torch.cat(accepted)[:num_samples]
        assert samples.shape[0] == num_samples

        if return_acceptance_rate:
            return samples, overall_acceptance_rate
        else:
            return samples

    def sample_sir(
        self,
        sample_shape: torch.Size = torch.Size(),
        show_progress_bars: bool = True,
        sampling_batch_size: int = 10_000,  # divide by oversampling
        return_ess: bool = False,
    ):
        assert len(sample_size := torch.Size(sample_shape)) == 1
        num_samples = sample_size[0]

        pbar = tqdm(
            disable=not show_progress_bars,
            total=num_samples,
            desc=f"Drawing {num_samples} restricted posterior samples",
        )

        oversampling_factor = self.oversample_sir
        assert sampling_batch_size % oversampling_factor == 0
        sir_batch_size = sampling_batch_size // oversampling_factor

        num_remaining = num_samples
        all_samples = []
        all_ess = []
        while num_remaining > 0:

            # use "free" log probs
            posterior_samples, posterior_log_probs = self._posterior.sample(
                (sampling_batch_size,),
                self._obs,
                max_sampling_batch_size=sampling_batch_size,
                with_log_prob=True,
            )
            truncated_prior_log_probs = self._prior.log_prob(posterior_samples)

            # Adaptive threshold instead of pre-computed
            thr = torch.quantile(posterior_log_probs, self.allowed_false_negatives)
            truncated_prior_log_probs[posterior_log_probs < thr] = -float("inf")

            log_ratios = truncated_prior_log_probs - posterior_log_probs
            log_ratios = torch.nan_to_num(log_ratios, -float("inf"))
            reshaped_ratio = torch.reshape(
                log_ratios, (sir_batch_size, oversampling_factor)
            )
            # Save guard
            probs = torch.exp(
                reshaped_ratio - torch.logsumexp(reshaped_ratio, dim=1, keepdim=True)
            )

            all_ess.append(1.0 / torch.sum(probs**2, dim=1))

            cat_dist = torch.distributions.Categorical(logits=reshaped_ratio)
            categorical_samples = cat_dist.sample((1,))[0, :]
            reshaped_posterior_samples = torch.reshape(
                posterior_samples, (sir_batch_size, self.oversample_sir, -1)
            )
            selected_posterior_samples = reshaped_posterior_samples[
                torch.arange(sir_batch_size), categorical_samples
            ]

            all_samples.append(selected_posterior_samples)
            num_remaining -= sir_batch_size
            pbar.update(sir_batch_size)

        pbar.close()

        samples = torch.cat(all_samples)[:num_samples]
        assert samples.shape[0] == num_samples

        ess = torch.cat(all_ess)
        log.info(f"Mean ESS: {ess.mean().item()}")
        log.info(f"Min ESS: {ess.min().item()}")
        if return_ess:
            return samples, ess
        else:
            return samples


# Some utils for pre-rejection and filtering


def prereject_with_bounds(
    proposal: Any,  # proposal distribution with the typical sample method
    lower_bound: Tensor,
    upper_bound: Tensor,
    sampling_batch_size: int = 10_000,
    pre_sampling_batch_size: int = 1_000_000,  # hard coded atm
):
    """
    Pre-reject samples that are not in the support of the posterior.
    Args:
        lower_bound: Lower bound of the support of the posterior.
        upper_bound: Upper bound of the support of the posterior.
    Returns:
        Samples from the proposal that are in the support of the posterior.
    """
    is_uniform = check_for_uniform(proposal)

    num_pre_accepted = 0
    num_sampled_total = 0
    pre_samples = []
    while num_pre_accepted < sampling_batch_size:
        samples = proposal.sample((pre_sampling_batch_size,))
        within_bounds = torch.all(
            (samples >= lower_bound) & (samples <= upper_bound), dim=1
        )
        samples = samples[within_bounds.bool()]
        pre_samples.append(samples)

        num_pre_accepted += samples.shape[0]
        num_sampled_total += pre_sampling_batch_size

        if is_uniform:
            break  # perform one iteration to estimate pre acceptance rate

    pre_acceptance_rate = num_pre_accepted / num_sampled_total

    if is_uniform:
        prop_lower_bound, prop_upper_bound = get_uniform_bounds(proposal)
        max_lower = torch.max(lower_bound, prop_lower_bound)
        min_upper = torch.min(upper_bound, prop_upper_bound)
        return (
            BoxUniform(max_lower, min_upper).sample((sampling_batch_size,)),
            pre_acceptance_rate,
        )
    else:
        return torch.cat(pre_samples)[:sampling_batch_size], pre_acceptance_rate


def check_for_uniform(proposal: Any):
    if isinstance(proposal, BoxUniform):
        return True
    if isinstance(proposal, Independent) and isinstance(proposal.base_dist, Uniform):
        return True
    # anything else?
    return False


def get_uniform_bounds(proposal):
    # anything else?
    return proposal.base_dist.low, proposal.base_dist.high


# NOTE filter functions should always return (theta, x) in that order
def get_filtering_method(name: str):
    if name == "no_filtering":
        return no_filtering
    elif name == "latest_filtering":
        return latest_filtering
    elif name == "random_filtering":
        return random_filtering
    elif name == "standardized_euclidean_filtering":
        return standardized_euclidean_filtering
    elif callable(name):
        return name
    else:
        raise ValueError(f"Unknown filtering method: {name}")


def no_filtering(obs: Tensor, theta: Tensor, x: Tensor, context_size: int):
    return theta, x


def latest_filtering(obs: Tensor, theta: Tensor, x: Tensor, context_size: int):
    # assumes that the latest samples are at the end
    return theta[-context_size:], x[-context_size:]


def random_filtering(obs: Tensor, theta: Tensor, x: Tensor, context_size: int):
    num_samples = theta.shape[0]
    perm = torch.randperm(num_samples)
    return theta[perm[:context_size]], x[perm[:context_size]]


def standardized_euclidean_filtering(
    obs: Tensor, theta: Tensor, x: Tensor, context_size: int
):
    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0)
    x_s = (x - x_mean) / x_std

    obs_s = (obs - x_mean) / x_std

    dists = torch.norm(x_s - obs_s, dim=1)

    _, idx = torch.topk(dists, min(context_size, dists.shape[0]), largest=False)
    return theta[idx], x[idx]


@torch.no_grad()
def accept_reject_sample(
    proposal: Callable,
    accept_reject_fn: Callable,
    num_samples: int,
    show_progress_bars: bool = False,
    max_sampling_batch_size: int = 10_000,
    proposal_sampling_kwargs: Optional[Dict] = None,
    max_iter_rejection: int | None = None,
) -> Tuple[Tensor, float]:
    """Returns samples from a proposal according to an acceptance criterion.

    Args:
        proposal: Function that generates proposal samples
        accept_reject_fn: Function that evaluates which samples are accepted
        num_samples: Desired number of samples
        show_progress_bars: Whether to show a progressbar during sampling
        max_sampling_batch_size: Maximum batch size for sampling
        proposal_sampling_kwargs: Arguments passed to proposal function

    Returns:
        Tuple of (accepted_samples, acceptance_rate)
    """
    if proposal_sampling_kwargs is None:
        proposal_sampling_kwargs = {}

    pbar = tqdm(
        disable=not show_progress_bars,
        total=num_samples,
        desc=f"Drawing {num_samples} posterior samples",
    )

    accepted = []
    accepted_log_probs = []

    num_remaining = num_samples
    num_sampled_total = 0
    sampling_batch_size = min(num_samples, max_sampling_batch_size)
    i = 0

    while num_remaining > 0:
        i += 1
        # Sample and reject
        candidates, log_probs = proposal(
            sampling_batch_size, **proposal_sampling_kwargs
        )
        are_accepted = accept_reject_fn(candidates)

        # Store accepted samples
        accepted.append(candidates[are_accepted])
        if log_probs is not None:
            accepted_log_probs.append(log_probs[are_accepted])

        # Update counters
        num_accepted = are_accepted.sum().item()
        num_sampled_total += sampling_batch_size
        num_remaining -= num_accepted
        pbar.update(num_accepted)

        # Adjust batch size based on acceptance rate
        acceptance_rate = sum(len(x) for x in accepted) / num_sampled_total
        sampling_batch_size = min(
            max_sampling_batch_size,
            max(int(1.5 * num_remaining / max(acceptance_rate, 1e-12)), 100),
        )

        if max_iter_rejection is not None and i > max_iter_rejection:
            accepted.append(candidates)
            accepted_log_probs.append(log_probs)
            break

    pbar.close()

    # Concatenate and trim to exact number of samples
    samples = torch.cat(accepted, dim=0)[:num_samples]
    log_probs = (
        torch.cat(accepted_log_probs, dim=0)[:num_samples]
        if accepted_log_probs  # if list is not empty
        else None
    )

    final_acceptance_rate = len(samples) / num_sampled_total

    return samples, log_probs, final_acceptance_rate
