# adapted from https://github.com/facebookresearch/flow_matching/blob/main/examples/2d_flow_matching.ipynb

import mlx.nn as nn
import mlx.core as mx

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x: mx.array) -> mx.array: 
        return mx.sigmoid(x) * x

# Model class
class MLP(nn.Module):
    def __init__(self, input_dim: int = 16, time_dim: int = 1, hidden_dim: int = 128):
        """
        Args:
            input_dim: The dimensionality of the latent representation. This should be the same size as the VAE latent space.


            TODO: add an action embedding as input.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim),
        )
    

    def __call__(self, x: mx.array, t: mx.array) -> mx.array:
        sz = x.shape
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim)

        h = mx.concat([x, t], axis=1)
        output = self.main(h)
        
        return output.reshape(*sz)