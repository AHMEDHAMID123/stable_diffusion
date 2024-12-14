import torch
import numpy as np


class DDPMsampler:
    def __init__(
        # Params "beta_start" and "beta_end" taken from:
        # https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L5C8-L5C8
        # For the naming conventions, refer to the DDPM paper (https://arxiv.org/pdf/2006.11239.pdf)
        self,
        generator: torch.Generator,
        num_training_steps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
    ):

        self.beta = torch.linspace(
            beta_start, beta_end, num_training_steps, dtype=torch.float32
        )
        self.alpha = 1 - self.beta

        self.alpha_bar = self.alpha.cumprod(0, dtype=torch.float32)

        self.generator = generator

        self.num_training_steps = num_training_steps

        self.timesteps = torch.arange(0, 1000, 1, dtype=torch.int).flip(0)

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step = self.num_training_steps // self.num_inference_steps
        self.timesteps = torch.arange(
            0, self.num_training_steps, step, dtype=torch.int
        ).flip(0)

    def set_strength(self, strength=1):
        starting_step = int(strength * self.num_inference_steps)
        self.timesteps = self.timesteps[starting_step:]

    def add_noise(
        self,
        original_sample: torch.tensor,
        timestep,
    ):
        # take a latent and time step and add noise to this latent at this timestep
        # adding noise formula from the paper -> q(x_t|x_0) = N~(x_0 * sqrt(alpha_bar), (1-alpha_bar) * I)
        image_shape = original_sample.shape
        random_noise = torch.randn(
            image_shape,
            generator=self.generator,
            dtype=torch.float32,
            device=original_sample.device,
        )
        self.variance = 1 - self.alpha_bar[timestep].to(
            original_sample.device, dtype=original_sample.dtype
        )
        self.mean_scale = (
            self.alpha_bar[timestep].to(
                original_sample.device, dtype=original_sample.dtype
            )
            ** 0.5
        )

        added_noise = original_sample * self.mean_scale + self.variance * random_noise
        return added_noise

    def _get_prev_time_step(self, timestep):
        prev_timestep = timestep - self.num_training_steps // self.num_training_steps
        return prev_timestep

    def step(self, timesteps, x_t, noise):
        # the reverse process , removing the noise (model_output) from the latents at given time step
        # x_0 = (x_t - sqrt(1-alpha_hat) * noise(x_t))/sqrt(alpha_hat)
        # q(x_{t-1} | x_{t}, X_0) = N(x_{t-1}; \mu_t(x_t, X_0), \beta_t I)
        # where \mu_t(x_t, X_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} X_0 + \frac{\sqrt{\bar{\alpha}_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} X_t
        # and \beta_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta

        # using eq 6, 7, 15 from the paper DDPM

        t = timesteps
        prev_t = self._get_prev_time_step(timesteps)

        x_0_inference = (x_t - self.alpha_bar[t] ** 0.5 * noise) / self.alpha_bar**0.5
        x_0_inference = x_0_inference.to(x_t.device, dtype=x_t.dtype)
        latent_shape = x_t.shape
        if t > 0:
            mean_t = (
                self.alpha_bar[prev_t] ** 0.5
                * self.beta
                / (1 - self.alpha_bar[t])
                * x_0_inference
                + (self.alpha[t] ** 0.5 * (1 - self.alpha_bar[prev_t]))
                / (1 - self.alpha_bar[t])
                * x_t
            )
            variance_t = (
                (1 - self.alpha_bar[prev_t]) / (1 - self.alpha_bar[t]) * self.beta[t]
            )

        else:
            mean_t = (
                self.alpha_bar[prev_t] ** 0.5
                * self.beta
                / (1 - self.alpha_bar[t])
                * x_0_inference
            )
            variance_t = 0

        random_sample = torch.randn(
            latent_shape, generator=self.generator, dtype=x_t.dtype, device=x_t.device
        )

        x_t_1 = mean_t + variance_t * random_sample
        return x_t_1
