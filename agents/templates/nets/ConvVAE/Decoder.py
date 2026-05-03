import mlx.nn as nn
import mlx.core as mx

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_channels: int, shared_embedding: nn.Embedding = None):
        super().__init__()
        # dense: latent_dim -> 1x1x1024
        self.fc = nn.Linear(latent_dim, 1024)

        # 1x1x1024 -> 5x5x128
        self.deconv1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=1, padding=0)
        # 5x5x128 -> 13x13x64
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=0)
        # 13x13x64 -> 30x30x32
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=0)
        # 30x30x32 -> 64x64xoutput_channels
        self.deconv4 = nn.ConvTranspose2d(32, output_channels, kernel_size=6, stride=2, padding=0)

        self.rmsNorm1 = nn.RMSNorm(128)
        self.rmsNorm2 = nn.RMSNorm(64)
        self.rmsNorm3 = nn.RMSNorm(32)
        self.rmsNorm4 = nn.RMSNorm(output_channels)

        self.embedding = shared_embedding if shared_embedding is not None else nn.Embedding(output_channels, output_channels)

    def __call__(self, z: mx.array) -> mx.array:
        x = self.fc(z)                           # 1024
        x = x.reshape(x.shape[0], 1, 1, 1024)   # 1x1x1024

        x = nn.relu(self.deconv1(x))             # 5x5x128
        x = self.rmsNorm1(x)
        x = nn.relu(self.deconv2(x))             # 13x13x64
        x = self.rmsNorm2(x)
        x = nn.relu(self.deconv3(x))             # 30x30x32
        x = self.rmsNorm3(x)
        x = self.deconv4(x)                      # 64x64xoutput_channels, logits
        x = self.rmsNorm4(x)
        # project back into embedding space
        x = self.embedding.as_linear(x)

        return x