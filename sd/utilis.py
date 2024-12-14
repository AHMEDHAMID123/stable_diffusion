import torch


def rescale(x: torch.Tensor, old_range: tuple, new_range: tuple, clamp: bool):

    x_min, x_max = old_range
    y_min, y_max = new_range

    x = (x - x_min) * (y_max - y_min) / (x_max - x_min) + y_min
    if clamp:
        x = x.clamp(y_min, y_max)
    return x


def get_time_embedding(timestep: float):
    # uses time embedding similar to the postional embedding in the origional transformer paper
    # d embedding is 320 / 2 -> 160
    # sin(t/n^(2i/d)) concat cos(t/n^(2i/d)) n = 10000 , d = 160
    freq = torch.pow(10000, -torch.arange(0, 160, dtype=torch.float32) / 160).unsqueeze(
        0
    )
    timestep = torch.tensor([timestep], dtype=torch.float32).unsqueeze(0)
    freq *= timestep
    # (1, 320)
    return torch.concat([torch.sin(freq), torch.cos(freq)], dim=-1)
