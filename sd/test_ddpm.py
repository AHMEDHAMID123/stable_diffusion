import unittest
import torch
from sd.ddpm import DDPMsampler


class TestDDPMsampler(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(42)
        self.ddpm_sampler = DDPMsampler(self.generator)

    def test_add_noise(self):
        original_sample = torch.randn(
            (3, 64, 64),
            generator=self.generator,
            dtype=torch.float32,
        )
        timestep = 10
        added_noise = self.ddpm_sampler.add_noise(original_sample, timestep)
        print(added_noise.shape)
        self.assertEqual(added_noise.shape, original_sample.shape)
        self.assertTrue(torch.is_tensor(added_noise))


if __name__ == "__main__":
    unittest.main()
