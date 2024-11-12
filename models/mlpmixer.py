import torch
import torch.nn as nn
import torch.nn.functional as F

class MixerBlock(nn.Module):
    def __init__(self, num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super(MixerBlock, self).__init__()
        self.token_mixer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(num_patches, tokens_mlp_dim),
            nn.GELU(),
            nn.Linear(tokens_mlp_dim, num_patches)
        )
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, channels_mlp_dim),
            nn.GELU(),
            nn.Linear(channels_mlp_dim, hidden_dim)
        )

    def forward(self, x):
        y = self.token_mixer(x.transpose(1, 2)).transpose(1, 2)  # Token mixing along the patches
        x = x + y
        y = self.channel_mixer(x)  # Channel mixing along the hidden dimensions
        x = x + y
        return x

class MLP_Mixer(nn.Module):
    def __init__(self, num_class=19, img_size=768, patch_size=32, in_channels=3, hidden_dim=256, num_blocks=8, tokens_mlp_dim=256, channels_mlp_dim=1024):
        super(MLP_Mixer, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        self.image_to_patch_embedding = nn.Linear(self.patch_dim, hidden_dim)
        self.mixer_blocks = nn.Sequential(*[
            MixerBlock(self.num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim)
            for _ in range(num_blocks)
        ])
        self.classifier = nn.Linear(hidden_dim, num_class)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, x.size(1), -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(batch_size, -1, self.patch_dim)
        x = self.image_to_patch_embedding(x)  # [batch_size, num_patches, hidden_dim]
        x = self.mixer_blocks(x)  # Ensure output shape is [batch_size, num_patches, hidden_dim]
        x = x.mean(dim=1)  # Global average pooling
        x = self.classifier(x)  # [batch_size, num_classes]
        return x

