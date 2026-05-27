# adapted from https://github.com/DriesSmit/ARC3-solution/blob/main/custom_agents/action.py

import mlx.nn as nn
import mlx.core as mx

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