import torch
from torch import nn
from torch.nn import functional as F
from attention import self_attention, cross_attention


class time_embedding(nn.Module):
    def __init__(self, n_embed):
        super.__int__()
        self.linear1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear2 = nn.Linear(4 * n_embed, 4 * n_embed)

    def forward(self, time: torch.Tensor):

        # (1, 320) -> (1, 1280)
        time = self.linear1(time)
        time = F.silu(time)
        time = self.linear2(time)
        # (1,1280)
        return time


class UNET_attentionblock(nn.Module):
    def __init__(self, num_heads, embedding_dim, d_context=768):
        super.__init__()
        channels = num_heads * embedding_dim  # embedding per head
        self.groupnorm1 = nn.GroupNorm(32, channels)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.layer_norm1 = nn.LayerNorm(channels)
        self.attention1 = self_attention(num_heads, embedding_dim, in_proj_bias=False)
        self.layer_norm2 = nn.LayerNorm(channels)
        self.attention2 = cross_attention(
            num_heads, embedding_dim, d_context, in_proj_bias=False
        )
        self.layer_norm3 = nn.LayerNorm(channels)
        self.linear_gelu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_gelu_2 = nn.Linear(4 * channels, channels)
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        # x: (batch_size, features, height, width)
        # context : (batch_size, seq_len, d_context) = (77, 768)

        residual_long = x
        n, c, h, w = x.shape
        x = self.groupnorm1(x)
        x = self.conv_input(x)

        # normalization + self attention
        x = x.view(n, c, h * w)
        x = x.transpose(-1, -2)
        residual_short = x
        x = self.attention1(x)
        x = self.layer_norm1(x)
        x += residual_short
        # normalization + corss attention
        residual_short = x
        x = self.layer_norm2(x)
        x = self.attention2(x)
        x += residual_short

        # feed forward
        residual_short = x
        x = self.layer_norm3(x)
        x, gate = self.linear_gelu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linear_gelu_2(x)
        x += residual_short

        x = x.transpose(-1, -2)
        x = x.view(n, c, h, w)

        return self.conv_output(x) + residual_long


class UNET_residualblock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super.__init__()

        self.group_norm1 = nn.GroupNorm(32, in_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.linear_time = nn.Linear(n_time, out_channels)

        self.group_norm2 = nn.GroupNorm(32, num_channels=out_channels)

        self.conv_merged = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

        if in_channels == out_channels:
            self.skip_layer = nn.Identity()
        else:
            self.skip_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:

        # x (batch_size , features, lenght, width)
        # time -> (1, 1280)

        residuals = x
        x = self.group_norm1(x)
        x = self.conv1(x)
        x = F.silu(x)

        time = self.linear_time(time)

        time = F.silu(time)

        # adding dims to time for broadcasting
        merged = x + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.group_norm2(merged)

        merged = self.conv_merged(merged)

        return merged + self.skip_layer(residuals)


class switch_sequntial(nn.Sequential):
    """
    A custom sequential container that switches the forward pass behavior based on the type of layer.
    Methods
    -------
    __init__():
        Initializes the switch_sequntial container.
    forward(latent, context, time):
        Defines the forward pass through the container, applying different operations based on the type of layer.
        Parameters:
        latent : torch.Tensor
            The input tensor to be processed by the layers.
        context : torch.Tensor
            The context tensor used by UNET_attentionblock layers.
        time : torch.Tensor
            The time tensor used by UNET_residualblock layers.
        Returns:
        torch.Tensor
            The output tensor after processing through all layers in the container.
    """

    def __init__(self):
        super.__init__()

    def forward(self, latent, context, time):
        for layer in self:
            if isinstance(layer, UNET_residualblock):
                latent = layer(latent, time)
            elif isinstance(layer, UNET_attentionblock):
                latent = layer(latent, context)
            else:
                latent = layer(latent)
        return latent


class Upsample(nn.Module):
    def __init__(self, num_channels):
        super.__init__()
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

    def forwward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNET(nn.module):
    def __init__(self):
        super.__init__()
        self.encoder = nn.ModuleList(
            [
                # 1
                switch_sequntial(
                    nn.Conv2d(in_channels=4, out_channels=320, kernel_size=3, padding=1)
                ),
                # 2
                switch_sequntial(
                    UNET_residualblock(320, 320), UNET_attentionblock(8, 40)
                ),
                # 3
                switch_sequntial(
                    UNET_residualblock(320, 320), UNET_attentionblock(8, 40)
                ),
                # 4change in dimensions
                switch_sequntial(
                    nn.Conv2d(
                        in_channels=320,
                        out_channels=320,
                        stride=2,
                        kernel_size=3,
                        padding=1,
                    )
                ),
                # 5change in feature space
                switch_sequntial(
                    UNET_residualblock(320, 640), UNET_attentionblock(8, 80)
                ),
                # 6
                switch_sequntial(
                    UNET_residualblock(640, 640), UNET_attentionblock(8, 80)
                ),
                # 7 change in dimensions
                switch_sequntial(
                    nn.Conv2d(
                        in_channels=640,
                        out_channels=640,
                        stride=2,
                        kernel_size=3,
                        padding=1,
                    )
                ),
                # 8 change in feature space
                switch_sequntial(
                    UNET_residualblock(640, 1280), UNET_attentionblock(8, 160)
                ),
                # 9
                switch_sequntial(
                    UNET_residualblock(1280, 1280), UNET_attentionblock(8, 160)
                ),
                # 10 change in dimension
                switch_sequntial(
                    nn.Conv2d(
                        in_channels=1280,
                        out_channels=1280,
                        stride=2,
                        kernel_size=3,
                        padding=1,
                    )
                ),
                # 11 change in feature space
                switch_sequntial(
                    UNET_residualblock(1280, 1280), UNET_attentionblock(8, 160)
                ),
                # 12
                switch_sequntial(
                    UNET_residualblock(1280, 1280), UNET_attentionblock(8, 160)
                ),
            ]
        )
        self.bottleneck = switch_sequntial(
            UNET_residualblock(1280, 1280),
            UNET_attentionblock(1280, 160),
            UNET_residualblock(1280, 1280),
        )

        self.decoder = nn.ModuleList(
            [  # 12
                switch_sequntial(UNET_residualblock(2560, 1280)),
                # 11
                switch_sequntial(UNET_residualblock(2560, 1280)),
                # 10
                switch_sequntial(UNET_residualblock(2560, 1280), Upsample(1280)),
                # 9
                switch_sequntial(
                    UNET_residualblock(2560, 1280), UNET_attentionblock(8, 160)
                ),
                # 8
                switch_sequntial(
                    UNET_residualblock(2560, 1280), UNET_attentionblock(8, 160)
                ),
                # 7
                switch_sequntial(
                    UNET_residualblock(1920, 1280),
                    UNET_attentionblock(8, 160),
                    Upsample(1280),
                ),
                # 6
                switch_sequntial(
                    UNET_residualblock(1920, 640), UNET_attentionblock(8, 160)
                ),
                # 5
                switch_sequntial(
                    UNET_residualblock(1280, 640), UNET_attentionblock(8, 80)
                ),
                # 4
                switch_sequntial(
                    UNET_residualblock(960, 640),
                    UNET_attentionblock(8, 80),
                    Upsample(640),
                ),
                # 3
                switch_sequntial(
                    UNET_residualblock(960, 320), UNET_attentionblock(8, 80)
                ),
                # 2
                switch_sequntial(
                    UNET_residualblock(640, 320), UNET_attentionblock(8, 80)
                ),
                # 1
                switch_sequntial(
                    UNET_residualblock(640, 320), UNET_attentionblock(8, 80)
                ),
            ]
        )

    def forward(self, latent, context, time):
        # latent coming from encoder vae -> (batch_size, features (4) , height/8 , width/8 )
        # context coming from clip (batch_size, seq_len (77) , d_embedd (768) )
        # time coming from time embedding (1, 1280)
        skip_connections = []
        for layer in self.decoder:
            x = layer(latent, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x)

        for layer in self.decoder:
            residual = skip_connections.pop()
            x = torch.concat([x, residual], dim=1)
            x = layer(x)

        return x


class UNET_outputlayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super.__init__()
        self.group_norm = nn.GroupNorm(32, in_channel)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        # x (batch, 320, length, width)
        x = self.group_norm(x)
        x = F.silu(x)
        x = self.conv(x)

        return x


class diffusion(nn.Module):
    def __init__(self):
        super.__init__()

        self.time_embedding = time_embedding(320)
        self.unet = UNET()
        # convert the output of the unet to be input to the unet again.
        self.final = UNET_outputlayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):

        # latent -> (batch , 4, hight/8, width/8)
        # context -> (batch, seq_len, d_embedding)
        # time -> (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        # (batch , 4, hight/8, width/8) -> (batch , 320, hight/8, width/8)
        output = self.unet(latent, context, time)
        # (batch , 320, hight/8, width/8) -> (batch , 4, hight/8, width/8)
        output = self.final(output)

        return output
