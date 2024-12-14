import torch
from torch import nn
from torch.nn import functional as F

from decoder import vae_residule_block, vae_attention_block


class vae_encoder(nn.Sequential):
    def __init__(self):
        ## it is a sequnce  of submodels
        super().__init__(
            # 1st block
            # dilation -> to allow to skip some pixels
            # input (batch_size, 3, 512(height), 512(width)) -> output (batch_size, 128, 512, 512)
            nn.Conv2d(
                in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            # input (batch_size, 128, 512, 512) -> output (batch_size, 128, 512, 512)
            vae_residule_block(128, 128),
            vae_residule_block(128, 128),
            # input (batch_size, 128, 512, 512) -> output (batch_size, 128, 512/2, 512/2)
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0
            ),
            # input (batch_size, 128, 512/2, 512/2) -> output (batch_size, 256, 512/2, 512/2)
            vae_residule_block(128, 265),
            # input (batch_size, 256, 512/2, 512/2) -> output (batch_size, 256, 512/2, 512/2)
            # increase the number of channels but decreasing the number of pixels
            vae_residule_block(256, 256),
            # input (batch_size, 256, 512/2, 512/2) -> output (batch_size, 256, 512/4, 512/4)
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0
            ),
            # input (batch_size, 256, 512/4, 512/4) -> output (batch_size, 512, 512/4, 512/4)
            vae_residule_block(256, 512),
            # input (batch_size, 512, 512/4, 512/4) -> output (batch_size, 512, 512/4, 512/4)
            vae_residule_block(512, 512),
            # input (batch_size, 512, 512/4, 512/4) -> output (batch_size, 512, 512/8, 512/8)
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0
            ),
            vae_residule_block(512, 512),
            vae_residule_block(512, 512),
            vae_residule_block(512, 512),
            # to relate pixels to each other, even if convolutional
            # layers are used it relate local features, but the attintion is used to relate global features
            vae_attention_block(512),
            # input (batch_size, 512, 512/8, 512/8) -> output (batch_size, 512, 512/8, 512/8)
            nn.GroupNorm(num_groups=32, num_channels=512),
            nn.SiLU(),  # works better in practice than ReLU
            # this one keeps the spatial dimensions but reduces the number of channels(features)
            # this the bottleneck layer
            # input (batch_size, 512, 512/8, 512/8) -> output (batch_size, 8, 512/8, 512/8)
            nn.Conv2d(512, 8, kernel_size=1, padding=1),
            # input (batch_size, 8, 512/8, 512/8) -> output (batch_size, 8, 512/8, 512/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:

        # x is the input image(batch_size, 3, 512, 512)
        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                # padding (left, right, top, bottom) asymmetric padding.
                x = F.pad(
                    x,
                    (0, 1, 0, 1),
                )
            x = module(x)
        # for variational autoencoder, we need to return the mean and the log of the variance
        # we learn the latent space distribution
        # the mean and the log of the variance are used to sample the latent space
        # (batch_size, 8, 512/8, 512/8) -> 2 * (batch_size, 4, 512/8, 512/8)
        mean, log_var = torch.chunk(x, 2, dim=1)
        # (batch_size, 4, 512/8, 512/8)
        # we do clamp to avoid the log_var to be too large or too small
        log_var = torch.clamp(log_var, -30, 20)
        stdev = torch.exp(0.5 * log_var)
        x = mean + noise * stdev

        # scaling the output by a constant factor, from diffusers repo
        x = x * 0.18215
        return x
