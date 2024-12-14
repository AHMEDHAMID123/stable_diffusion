import torch
from torch import nn
from torch.nn import functional as F
from attention import self_attention


class vae_attention_block(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # mean and variance are calculated for each group of features
        # the number of groups is 32
        # so we use group because closer features coming from the same convolution
        # normalization making the training faster and more stable by making the layers outputs have same same scale

        self.group_norm = nn.GroupNorm(32, in_channels)
        self.attention = self_attention(1, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : batch_size, in_channels, height, width
        residual = x
        x = self.group_norm(x)
        n, c, h, w = x.shape
        # batch_size, in_channels, height*width
        x = x.view(n, c, h * w)
        # we transpose the tensor to make the pixels the sequence of the tensor
        # batch_size, height*width, in_channels -> sequence of pixels and each pixel has its own embedding
        x = x.transpose(-1, -2)
        x = self.attention(x)
        # we transpose the tensor back to the original shape
        x = x.transpose(-1, -2)
        # batch_size, in_channels, height*width -> batch_size, in_channels, height, width
        x = x.view(n, c, h, w)
        x += residual
        return x


class vae_residule_block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # made of normalizations
        self.group_norm1 = nn.GroupNorm(32, in_channels)
        # convolutional layer that change the number of features keeping the same spatial dimensions
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.group_norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        # residual connections
        if in_channels != out_channels:
            # if the number of features is different, we need to change the number of features to match for addition
            self.skip = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # X : batch_size, in_channels, height, width

        residual = x

        x = self.group_norm1(x)
        # activation after the normalization
        x = F.silu(x)
        x = self.conv1(x)
        x = self.group_norm2(x)
        # activation after the normalization
        x = F.silu(x)
        x = self.conv2(x)
        x = x + self.skip(residual)
        return x

    class vae_decoder(nn.Sequential):
        def __init__(self):
            super.__init__(
                # reverse the encoder
                nn.Conv2d(
                    in_channels=4, out_channels=4, kernel_size=1, stride=1, padding=0
                ),
                nn.Conv2d(
                    in_channels=4, out_channels=512, kernel_size=3, stride=1, padding=0
                ),
                vae_residule_block(512, 512),
                vae_attention_block(512),
                vae_residule_block(512, 512),
                vae_residule_block(512, 512),
                vae_residule_block(512, 512),
                vae_residule_block(512, 512),
                nn.Upsample(
                    scale_factor=2,
                ),
                nn.Conv2d(
                    in_channels=512,
                    out_channels=512,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                vae_residule_block(512, 512),
                vae_residule_block(512, 512),
                vae_residule_block(512, 512),
                nn.Upsample(
                    scale_factor=2,
                ),
                nn.Conv2d(
                    in_channels=512,
                    out_channels=512,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                vae_residule_block(512, 265),
                vae_residule_block(256, 256),
                vae_residule_block(256, 256),
                nn.Upsample(
                    scale_factor=2,
                ),
                nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                vae_residule_block(256, 128),
                vae_residule_block(128, 128),
                vae_residule_block(128, 128),
                nn.GroupNorm(32, 128),
                nn.SiLU(),
                nn.Conv2d(
                    in_channels=128, out_channels=3, kernel_size=1, stride=1, padding=0
                ),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:

            # input : batch_size, 4, 512/8, 512/8
            x /= 0.18215
            for module in self:
                x = module(x)
            # batch_size, 3, 512, 512
            return x
