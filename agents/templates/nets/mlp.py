# adapted from https://github.com/facebookresearch/flow_matching/blob/main/examples/2d_flow_matching.ipynb

import mlx.nn as nn
import mlx.core as mx

from arcengine import GameAction

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x: mx.array) -> mx.array: 
        return mx.sigmoid(x) * x

# Model class
class MLP(nn.Module):
    def __init__(self, input_dim: int = 16, time_dim: int = 1, action_dim: int = 16, hidden_dim: int = 128):
        """
        Args:
            input_dim: The dimensionality of the latent representation. This should be the same size as the VAE latent space.
            action_dim: The dimensionality of the action space.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(input_dim + time_dim + action_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.action_embedding = nn.Embedding(len(GameAction), action_dim)
        # TODO: vary the hidden size of this
        self.complex_action_embedding = nn.Sequential(
            nn.Linear(action_dim + 2, action_dim),
            Swish(),
            nn.Linear(action_dim, action_dim),
            Swish(),
            nn.Linear(action_dim, action_dim),
        )

    def __call__(self, x: mx.array, t: mx.array, actions: list[GameAction]) -> mx.array:
        sz = x.shape
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim)

        # Handle multiple actions (e.g., for batch processing)
        action_vecs = []
        for action in actions:
            if action.is_complex():
                action_data = action.action_data
                position = mx.array([[action_data.x, action_data.y]], dtype=mx.float32)
                action_embed = self.action_embedding(mx.array([action.value]))
                action_vec = self.complex_action_embedding(mx.concat([action_embed, position], axis=1))
            else:
                action_vec = self.action_embedding(mx.array([action.value]))
            action_vecs.append(action_vec)

        h = mx.concat([x, t, mx.concat(action_vecs)], axis=1)
        output = self.main(h)
        
        return output.reshape(*sz)
