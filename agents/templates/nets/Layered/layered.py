import mlx.core as mx
import mlx.nn as nn
from arcengine import GameAction


class NextDynamicLayerPrediction(nn.Module):
    def __init__(self, action_dim: int, token_embedding: nn.Embedding):
        super().__init__()

        self.action_dim = action_dim
        self.token_embedding = token_embedding
        self.embedding_dim = token_embedding.weight.shape[1]

        self.net = nn.Sequential(
            nn.Conv2d(
                2 * self.embedding_dim + 1 + action_dim,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.RMSNorm(128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.RMSNorm(64),
            nn.Conv2d(64, self.embedding_dim, kernel_size=3, padding=1),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(
                2 * self.embedding_dim + 1 + action_dim,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.RMSNorm(64),
            nn.Conv2d(64, self.embedding_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.RMSNorm(self.embedding_dim),
            nn.Conv2d(self.embedding_dim, 1, kernel_size=3, padding=1),
        )

        self.action_embedding = nn.Embedding(len(GameAction), action_dim)

    def __call__(
        self,
        static_layer: mx.array,
        dynamic_layer: mx.array,
        gate_logits: mx.array,
        actions: list[GameAction],
    ) -> tuple[mx.array, mx.array]:
        # static_layer shape: (batch_size, height, width, embedding_dim)
        # dynamic_layer shape: (batch_size, height, width, embedding_dim)
        # gate_logits shape: (batch_size, height, width, 1)
        # actions is a list of GameAction enums of length batch_size

        action_embeddings = self.action_embedding(mx.array([action.value for action in actions]))  # shape: (batch_size, action_dim)
        action_embeddings = mx.broadcast_to(
            action_embeddings[:, None, None, :],
            dynamic_layer.shape[:-1] + (self.action_dim,),
        )  # shape: (batch_size, height, width, action_dim)

        x = mx.concat(
            [static_layer, dynamic_layer, gate_logits, action_embeddings],
            axis=-1,
        )  # shape: (batch_size, height, width, 2 * embedding_dim + 1 + action_dim)
        features = self.net(x)  # shape: (batch_size, height, width, embedding_dim)

        logits = self.token_embedding.as_linear(features)  # shape: (batch_size, height, width, vocab_size)
        gate_logits = self.gate(x)  # shape: (batch_size, height, width, 1)

        return logits, gate_logits


class Layered(nn.Module):
    def __init__(self, vocab_size: int, action_dim: int = 16):
        """
        Args:
            vocab_size: The size of the vocabulary for the input states. This should be the same as the number of possible colours for a frame pixel.

            action_dim: The dimensionality of the action representation.
        """
        super().__init__()

        assert vocab_size == 16

        self.vocab_size = vocab_size
        self.action_dim = action_dim

        self.token_embedding = nn.Embedding(vocab_size, 32)
        self.embedding_dim = self.token_embedding.weight.shape[1]

        self.backbone = nn.Sequential(
            nn.Conv2d(self.embedding_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.RMSNorm(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.RMSNorm(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.RMSNorm(256),
        )

        self.static_layer = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.RMSNorm(128),
            nn.Conv2d(128, self.embedding_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.RMSNorm(self.embedding_dim),
        )

        # dynamic layer needs more global context, larger kernel size and more channels
        self.dynamic_layer = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.RMSNorm(128),
            nn.Conv2d(128, self.embedding_dim, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.RMSNorm(self.embedding_dim),
        )

        self.dynamic_gate = nn.Sequential(
            nn.Conv2d(2 * self.embedding_dim, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.RMSNorm(64),
            nn.Conv2d(64, self.embedding_dim, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.RMSNorm(self.embedding_dim),
            nn.Conv2d(self.embedding_dim, 1, kernel_size=3, padding=1),
        )

        self.next_dynamic_layer_predictor = NextDynamicLayerPrediction(
            action_dim,
            self.token_embedding,
        )

    def __call__(
        self,
        x_0: mx.array,
        actions: list[GameAction],
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        """
            returns: static_layer_logits, dynamic_layer_logits, dynamic_gate_logits,
                next_dynamic_layer_logits, next_dynamic_gate_logits

            shapes:
            (batch_size, height, width, vocab_size),
            (batch_size, height, width, vocab_size),
            (batch_size, height, width, 1),
            (batch_size, height, width, vocab_size),
            (batch_size, height, width, 1)
        """
        (
            static_layer_features,
            dynamic_layer_features,
            static_layer_logits,
            dynamic_layer_logits,
            dynamic_gate_logits,
        ) = self.decompose(x_0)

        # next frame dynamic layer prediction
        (
            next_dynamic_layer_logits,
            next_dynamic_gate_logits,
        ) = self.next_dynamic_layer_predictor(
            static_layer_features,
            dynamic_layer_features,
            dynamic_gate_logits,
            actions,
        )

        return (
            static_layer_logits,
            dynamic_layer_logits,
            dynamic_gate_logits,
            next_dynamic_layer_logits,
            next_dynamic_gate_logits,
        )

    def decompose(
        self,
        x_0: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        # Embed the input tokens. mlx Conv2d expects shape
        # (batch_size, height, width, embedding_dim), so no permutation is needed.
        x_0 = self.token_embedding(x_0)

        features = self.backbone(x_0)  # shape: (batch_size, height, width, 256)

        static_layer_features = self.static_layer(features)
        dynamic_layer_features = self.dynamic_layer(features)

        static_layer_logits = self.token_embedding.as_linear(static_layer_features)
        dynamic_layer_logits = self.token_embedding.as_linear(dynamic_layer_features)
        dynamic_gate_logits = self.dynamic_gate(
            mx.concat([static_layer_features, dynamic_layer_features], axis=-1)
        )

        return (
            static_layer_features,
            dynamic_layer_features,
            static_layer_logits,
            dynamic_layer_logits,
            dynamic_gate_logits,
        )
