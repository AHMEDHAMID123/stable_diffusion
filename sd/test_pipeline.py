import unittest
import torch
from utilis import rescale


class TestPipeline(unittest.TestCase):

    def test_rescale_no_clamp(self):
        x = torch.tensor([0.0, 0.5, 1.0])
        old_range = (0.0, 1.0)
        new_range = (0.0, 10.0)
        expected = torch.tensor([0.0, 5.0, 10.0])
        result = rescale(x, old_range, new_range, clamp=False)
        self.assertTrue(torch.allclose(result, expected))

    def test_rescale_with_clamp(self):
        x = torch.tensor([-0.5, 0.5, 1.5])
        old_range = (0.0, 1.0)
        new_range = (0.0, 10.0)
        expected = torch.tensor([0.0, 5.0, 10.0])
        result = rescale(x, old_range, new_range, clamp=True)
        self.assertTrue(torch.allclose(result, expected))


if __name__ == "__main__":
    unittest.main()
