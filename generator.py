import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor
        '''
        DCGAN For input size 𝑊, kernel size 𝐾, padding p, and stride s, ((𝑊−1)×𝑠−𝑝×2+𝐾)
        Starting with input noise, gradually expand using the inverse convolution operation.
        Calculate the initial size and work backwards: 128/32=4, 87/32≈2.7→3
        Therefore, the starting size is 4x3.
        '''

        # Projection layer - Create small feature maps from noise vectors
        self.projection = nn.Linear(self.preprocessor.n_noise, 256 * 4 * 3)
        self.bn_proj = nn.BatchNorm1d(256 * 4 * 3)

        # Expansion using convolutional transposition layers
        # 4x3 → 8x6
        self.convt1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        # 8x6 → 16x11 ((6 - 1)* 2 - 0 + 3) = 13
        self.convt2 = nn.ConvTranspose2d(128, 64, kernel_size=(4, 3), stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 16x11 → 32x22
        self.convt3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        # 32x22 → 64x44
        self.convt4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(16)

        # 64x44 → 128x87
        self.convt5 = nn.ConvTranspose2d(16, 1, kernel_size=(4, 3), stride=2, padding=1)

    def forward(self, x):

        # Flatten to two dimensions before putting into Linear
        x = x.view(-1, self.preprocessor.n_noise)  # [batch_size, n_noise]

        # Projection & Reshape
        x = self.projection(x)                # [batch_size, 256*4*3]
        x = self.bn_proj(x)
        x = F.relu(x)
        x = x.view(-1, 256, 4, 3)             # [batch_size, 256, 4, 3]

        # Stepwise upscaling using reverse convolution layers
        x = F.relu(self.bn1(self.convt1(x)))  # [batch_size, 128, 8, 6]
        # print(f"1: {x.shape}")
        x = F.relu(self.bn2(self.convt2(x)))  # [batch_size, 64, 16, 11]
        # print(f"2: {x.shape}")
        x = F.relu(self.bn3(self.convt3(x)))  # [batch_size, 32, 32, 22]
        # print(f"3: {x.shape}")
        x = F.relu(self.bn4(self.convt4(x)))  # [batch_size, 16, 64, 44]
        # print(f"4: {x.shape}")

        # Final layer - Output restricted to [-1, 1] with tanh activation function
        x = torch.tanh(self.convt5(x))        # [batch_size, 1, 128, 87]
        # print(f"5: {x.shape}")
        return x

