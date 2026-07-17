# adapted from https://github.com/DriesSmit/ARC3-solution/blob/main/custom_agents/action.py

import mlx.core as mx
import mlx.nn as nn


class ActionModel(nn.Module):
    """CNN that predicts which actions will result in new frames with shared conv backbone."""
    
    def __init__(self, num_colors=16, num_simple_action_types=6, is_coord_action_allowed=False):
        super().__init__()
        self.num_simple_action_types = num_simple_action_types
        self.is_coord_action_allowed = is_coord_action_allowed

        self.embedding = nn.Embedding(num_colors, 16)  # Embed frame values into channels
        
        # Shared convolutional backbone
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        if self.num_simple_action_types > 0:
            self.action_pool = nn.MaxPool2d(4, 4)  # Reduce to 16x16 like coordinates

            # Action head - preserve spatial information
            action_flattened_size = 256 * 16 * 16  # 65,536
            self.action_head = nn.Sequential(
                nn.Linear(action_flattened_size, 512),        # Flatten spatial dimensions
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, self.num_simple_action_types)
            )

        
        # Coordinate head - enhanced spatial reasoning (64x64 action space)
        if is_coord_action_allowed:
            self.coord_conv = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Spatial processing
                nn.ReLU(),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),   # More spatial processing
                nn.ReLU(),
                nn.Conv2d(64, 32, kernel_size=1),               # Channel reduction
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=1),                # Final logits
            )
        
        # No special initialization - let coordinates start naturally
        
        self.dropout = nn.Dropout(0.2)
        
    def __call__(self, x):
        # x shape: (batch_size, height, width, channels)

        x = self.embedding(x)  # (batch, height, width, input_channels)
        
        # Shared convolutional backbone
        x = nn.relu(self.conv1(x))  # (batch, 64, 64, 32)
        x = nn.relu(self.conv2(x))  # (batch, 64, 64, 64)
        x = nn.relu(self.conv3(x))  # (batch, 64, 64, 128)
        conv_features = nn.relu(self.conv4(x))  # (batch, 64, 64, 256)

        if self.num_simple_action_types:
            # Action head - preserve spatial information (5 actions)
            action_features = self.action_pool(conv_features)  # (batch, 16, 16, 256)
            action_features = action_features.reshape((action_features.shape[0], -1))  # Flatten spatial dimensions (batch, 65536)
            action_logits = self.action_head(action_features)  # (batch, self.num_simple_action_types)
        else:
            action_logits = mx.zeros((conv_features.shape[0], 0))  # No simple actions
        
        # Coordinate head - enhanced 64x64 action space
        if self.is_coord_action_allowed:
            coord_logits = self.coord_conv(conv_features)            # (batch, 64, 64, 1)
            coord_logits = mx.reshape(coord_logits, (coord_logits.shape[0], -1)) # (batch, 4096)
        else:
            coord_logits = mx.zeros((conv_features.shape[0], 0))  # No coordinate actions
        
        # Return combined logits: [6 action logits, 4096 coordinate logits]
        combined_logits = mx.concatenate([action_logits, coord_logits], axis=1)  # (batch, 6 + 4096)
        
        return combined_logits


class ConnectedComponentConvNet(nn.Module):
    """Masked CNN that embeds one connected component per input frame."""

    def __init__(
        self,
        num_colors: int,
        color_embedding_dim: int = 16,
        hidden_dim: int = 64,
        component_embedding_dim: int = 48,
        position_feature_dim: int = 4,
    ):
        super().__init__()
        self.num_colors = num_colors
        self.color_embedding_dim = color_embedding_dim
        self.hidden_dim = hidden_dim
        self.component_embedding_dim = component_embedding_dim
        self.position_feature_dim = position_feature_dim

        self.color_embedding = nn.Embedding(num_colors, color_embedding_dim)
        self.conv1 = nn.Conv2d(
            color_embedding_dim + 1 + position_feature_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
        )
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm3 = nn.RMSNorm(hidden_dim)

        self.component_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, component_embedding_dim),
        )

    def __call__(
        self,
        frames: mx.array,
        dynamic_mask: mx.array,
    ) -> mx.array:
        color_features = self.color_embedding(frames)
        color_features = mx.where(
            dynamic_mask[..., None],
            color_features,
            mx.zeros_like(color_features),
        )
        component_mask_float = dynamic_mask.astype(mx.float32)
        inputs = [color_features, component_mask_float[..., None]]
        if self.position_feature_dim:
            inputs.append(self._pixel_position_features(dynamic_mask))
        x = mx.concatenate(inputs, axis=-1)

        x = self._masked_relu(self.norm1(self.conv1(x)), dynamic_mask)
        x = self._masked_relu(self.norm2(self.conv2(x)), dynamic_mask)
        x = self._masked_relu(self.norm3(self.conv3(x)), dynamic_mask)

        return x

    def _masked_relu(
        self,
        features: mx.array,
        dynamic_mask: mx.array,
    ) -> mx.array:
        return mx.where(
            dynamic_mask[..., None],
            nn.relu(features),
            mx.zeros_like(features),
        )
    
    def cc_embedding_from_features(self, component_features: mx.array, component_mask: mx.array) -> mx.array:
        component_denominator = mx.sum(component_mask, axis=(1, 2))[:, None]
        aggregated_component_features = (
            mx.sum(component_features * component_mask[..., None], axis=(1, 2))
            / (component_denominator + 1e-6)
        )
        return self.component_projection(aggregated_component_features)

    def _pixel_position_features(self, dynamic_mask: mx.array) -> mx.array:
        batch_size, height, width = dynamic_mask.shape
        mask = dynamic_mask.astype(mx.float32)

        y_coords = (
            mx.arange(height, dtype=mx.float32)[None, :, None]
            / max(height - 1, 1)
            * 2.0
            - 1.0
        )
        x_coords = (
            mx.arange(width, dtype=mx.float32)[None, None, :]
            / max(width - 1, 1)
            * 2.0
            - 1.0
        )

        pi = mx.array(3.141592653589793, dtype=mx.float32)
        features = mx.concatenate(
            [
                mx.broadcast_to(mx.sin(pi * y_coords), (batch_size, height, width))[..., None],
                mx.broadcast_to(mx.cos(pi * y_coords), (batch_size, height, width))[..., None],
                mx.broadcast_to(mx.sin(pi * x_coords), (batch_size, height, width))[..., None],
                mx.broadcast_to(mx.cos(pi * x_coords), (batch_size, height, width))[..., None],
            ],
            axis=-1,
        )
        return features[..., : self.position_feature_dim] * mask[..., None]
