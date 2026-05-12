import mlx.nn as nn
import mlx.core as mx
from arcengine import GameAction


class NextDynamicLayerPrediction(nn.Module):
    def __init__(self, action_dim: int, token_embedding: nn.Embedding):
        super().__init__()

        self.action_dim = action_dim
        self.token_embedding = token_embedding
        self.embedding_dim = token_embedding.weight.shape[1]

        self.net = nn.Sequential(
            nn.Conv2d(2 * self.embedding_dim + action_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.RMSNorm(128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.RMSNorm(64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.RMSNorm(32),
            nn.Conv2d(32, self.embedding_dim, kernel_size=3, padding=1), # +1 for <blank> token
        )

        self.action_embedding = nn.Embedding(len(GameAction), action_dim)

    def __call__(self, static_layer: mx.array, dynamic_layer: mx.array, actions: list[GameAction]) -> mx.array:
        # static_layer shape: (batch_size, height, width, 16)
        # dynamic_layer shape: (batch_size, height, width, 16)
        # actions is a list of GameAction enums of length batch_size

        action_embeddings = self.action_embedding(mx.array([action.value for action in actions]))  # shape: (batch_size, action_dim)
        action_embeddings = mx.broadcast_to(
            action_embeddings[:, None, None, :],
            dynamic_layer.shape[:-1] + (self.action_dim,),
        )  # shape: (batch_size, height, width, action_dim)

        x = mx.concat([static_layer, dynamic_layer, action_embeddings], axis=-1)  # shape: (batch_size, height, width, vocab_size + vocab_size + action_dim)
        features = self.net(x)  # shape: (batch_size, height, width, vocab_size + 1)

        logits = self.token_embedding.as_linear(features)  # shape: (batch_size, height, width, vocab_size + 1)

        return logits

class Layered(nn.Module):
    def __init__(self, vocab_size: int, action_dim: int = 16):
        """
        Args:
            vocab_size: The size of the vocabulary for the input states. This should be the same as the number of possible tokens in the state representation.
            
            static_layer_vocab_size: The size of the vocabulary for the static layer. This should be the same as the number of possible tokens in the static layer representation.
            
            dynamic_layer_vocab_size: The size of the vocabulary for the dynamic layer. This should be the same as the number of possible tokens in the dynamic layer representation and additional tokens for <blank>.

            action_dim: The dimensionality of the action representation.
        """
        super().__init__()

        assert vocab_size == 16
        
        self.vocab_size = vocab_size
        self.static_layer_vocab_size = vocab_size
        self.dynamic_layer_vocab_size = vocab_size + 1 # +1 for <blank> token
        self.blank_token_id = vocab_size # the id for the <blank> token in the dynamic layer

        self.action_dim = action_dim

        self.token_embedding = nn.Embedding(vocab_size + 1, 16)

        self.backbone = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.RMSNorm(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
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
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.RMSNorm(64),    
            nn.Conv2d(64, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.RMSNorm(16),
        )

        self.dynamic_layer = nn.Sequential(
            nn.Conv2d(256 + 16, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.RMSNorm(64),    
            nn.Conv2d(64, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.RMSNorm(16),
        )

        self.next_dynamic_layer_predictor = NextDynamicLayerPrediction(action_dim, self.token_embedding)


    def __call__(self, x_0: mx.array, actions: list[GameAction]) -> mx.array:
        """
            returns: static_layer_logits, dynamic_layer_logits, next_dynamic_layer_logits

            shapes: 
            (batch_size, height, width, static_layer_vocab_size), 
            (batch_size, height, width, dynamic_layer_vocab_size), 
            (batch_size, height, width, dynamic_layer_vocab_size)
        """
        # Embed the input tokens
        x_0 = self.token_embedding(x_0)  # mlx Conv2d expect shape: (batch_size, height, width, embedding_dim), no need to permute

        # Pass through the backbone
        features = self.backbone(x_0)  # shape: (batch_size, height, width, 256)

        static_layer_features = self.static_layer(features)  # shape: (batch_size, height, width, 16)
        dynamic_layer_features = self.dynamic_layer(mx.concatenate([features, static_layer_features], axis=-1))  # shape: (batch_size, height, width, 16)
        
        
        static_layer_logits = self.token_embedding.as_linear(static_layer_features)  # shape: (batch_size, height, width, vocab_size)
        dynamic_layer_logits = self.token_embedding.as_linear(dynamic_layer_features)  # shape: (batch_size, height, width, vocab_size + 1)

        static_layer_logits = static_layer_logits[:, :, :, :self.static_layer_vocab_size]  # remove logit for <blank> token


        # next frame dynamic layer prediction
        next_dynamic_layer_logits = self.next_dynamic_layer_predictor(static_layer_features, dynamic_layer_features, actions)  # shape: (batch_size, height, width, vocab_size + 1)


        return static_layer_logits, dynamic_layer_logits, next_dynamic_layer_logits








        

        
        
