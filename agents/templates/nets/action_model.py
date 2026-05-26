# adapted from https://github.com/DriesSmit/ARC3-solution/blob/main/custom_agents/action.py

import mlx.nn as nn
import mlx.core as mx

class ActionModel(nn.Module):
    """CNN that predicts which actions will result in new frames with shared conv backbone."""
    
    def __init__(self, num_colors=16, input_channels=16, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.num_action_types = 6  # ACTION1-ACTION5, ACTION7

        self.embedding = nn.Embedding(num_colors, input_channels)  # Embed frame values into channels
        
        # Shared convolutional backbone
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Action head - preserve spatial information
        self.action_pool = nn.MaxPool2d(4, 4)  # Reduce to 16x16 like coordinates
        action_flattened_size = 256 * 16 * 16  # 65,536
        self.action_fc = nn.Linear(action_flattened_size, 512)
        self.action_head = nn.Linear(512, self.num_action_types)
        
        # Coordinate head - enhanced spatial reasoning (64x64 action space)
        self.coord_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # Spatial processing
        self.coord_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)   # More spatial processing
        self.coord_conv3 = nn.Conv2d(64, 32, kernel_size=1)               # Channel reduction
        self.coord_conv4 = nn.Conv2d(32, 1, kernel_size=1)                # Final logits
        
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
        
        # Action head - preserve spatial information (5 actions)
        action_features = self.action_pool(conv_features)  # (batch, 16, 16, 256)
        action_features = mx.reshape(action_features, (action_features.shape[0], -1))  # (batch, 65536)
        action_features = nn.relu(self.action_fc(action_features))  # (batch, 512)
        action_features = self.dropout(action_features)
        action_logits = self.action_head(action_features)  # (batch, 6)
        
        # Coordinate head - enhanced 64x64 action space
        coord_features = nn.relu(self.coord_conv1(conv_features))  # (batch, 64, 64, 128)
        coord_features = nn.relu(self.coord_conv2(coord_features))  # (batch, 64, 64, 64)
        coord_features = nn.relu(self.coord_conv3(coord_features))  # (batch, 64, 64, 32)
        coord_logits = self.coord_conv4(coord_features)            # (batch, 64, 64, 1)
        coord_logits = mx.reshape(coord_logits, (coord_logits.shape[0], -1)) # (batch, 4096)
        
        # Return combined logits: [6 action logits, 4096 coordinate logits]
        combined_logits = mx.concatenate([action_logits, coord_logits], axis=1)  # (batch, 6 + 4096)
        
        return combined_logits