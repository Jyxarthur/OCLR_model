import torch
import einops
import numpy as np
import torch.nn as nn



def build_time_grid(frames):
    # input: t (number of frames)
    # output dimension: 1 1 t 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    grid = np.linspace(-1., 1., num=frames)
    grid = np.reshape(grid, [frames, -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(grid).to(device)

def build_spacetime_grid(frames, resolution):
    # input: t,h,w (number of frames, height, width)
    # output dimension: 1 t h w 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ranges = [np.linspace(-1., 1., num=frames)] + [np.linspace(-1., 1., num=res) for res in resolution] 
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [frames, resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(grid).to(device)


class SoftTimeEmbed(nn.Module):
   #input dimension: b q t c (batch_size number_of_queries frames channels)
   #output dimension: b q t c
    def __init__(self, hidden_size):
        super(SoftTimeEmbed, self).__init__()
        self.proj = nn.Linear(1, hidden_size)
        self.grid = build_time_grid(5)

    def forward(self, inputs, frames):
        if self.grid.size()[1] != frames:
            self.grid = build_time_grid(frames)
        b = inputs.size()[0]
        inputs = einops.rearrange(inputs, 'b q t c -> (b q) t c')
        inputs = inputs + self.proj(self.grid)
        inputs = einops.rearrange(inputs, '(b q) t c -> b q t c', b = b)
        return inputs

class SoftPositionTimeEmbed(nn.Module):
    # input dimension: b t h w c (batch_size frames height width channels)
    # output dimension: b t h w c
    def __init__(self, hidden_size):
        super(SoftPositionTimeEmbed, self).__init__()
        self.proj = nn.Linear(3, hidden_size)
        self.grid = build_spacetime_grid(5, (8, 14))

    def forward(self, inputs, frames, resolution):
        if self.grid.size()[1:4] != (frames, resolution[0], resolution[1]):
            self.grid = build_spacetime_grid(frames, resolution)
        return inputs + self.proj(self.grid)

  
def attn_mask(frames, resolution):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vec = torch.unsqueeze(torch.unsqueeze(torch.arange(frames) + 1, 1), 2)
    vec = vec.expand(frames, resolution[0], resolution[1])
    vec = torch.reshape(vec, [vec.shape[0] * vec.shape[1] * vec.shape[2]])
    mask = (vec.reshape(1, -1) - vec.reshape(-1, 1)) == 0
    return mask.to(device)


