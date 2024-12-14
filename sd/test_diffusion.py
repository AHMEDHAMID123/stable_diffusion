import unittest
import torch
from torch import nn
from diffusion import switch_sequntial, UNET_residualblock, UNET_attentionblock

class TestSwitchSequential(unittest.TestCase):
    def setUp(self):
        self.latent = torch.randn(1, 320, 8, 8)
        self.context = torch.randn(1, 8, 40)
        self.time = torch.randn(1, 1280)

        self.residual_block = UNET_residualblock(320, 320)
        self.attention_block = UNET_attentionblock()
        self.conv_layer = nn.Conv2d(320, 320, kernel_size=3, padding=1)

        self.model = switch_sequntial(
            self.residual_block,
            self.attention_block,
            self.conv_layer
        )

    def test_forward_pass(self):
        output = self.model(self.latent, self.context, self.time)
        self.assertEqual(output.shape, self.latent.shape)

if __name__ == '__main__':
    unittest.main()