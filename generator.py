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
        Calculate the initial size and work backwards: 17 -> 33 -> 65 -> 129 -> 257 -> 513,
        3 -> 6 -> 11 -> 22 -> 44 -> 87
        '''

        # Projection layer - Create small feature maps from noise vectors
        self.init_h = 17
        self.init_w = 3
        self.projection = nn.Linear(self.preprocessor.n_noise, 256 * self.init_h * self.init_w)
        self.bn_proj = nn.BatchNorm1d(256 * self.init_h * self.init_w)

        # Expansion using convolutional transposition layers
        # 17x3 → 33x6 ((3 - 1)* 2 - 0 + 4) = 8
        self.convt1 = nn.ConvTranspose2d(256, 128, kernel_size=(3, 4), stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        # 33x6 → 65x11 ((6 - 1)* 2 - 0 + 4) = 16
        self.convt2 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 65x11 → 129x22 ((11 - 1)* 2 - 0 + 4) = 32
        self.convt3 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 4), stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        # 129x22 → 257x44 ((22 - 1)* 2 - 0 + 4) = 64
        self.convt4 = nn.ConvTranspose2d(32, 16, kernel_size=(3, 4), stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(16)

        # 257x44 → 513x87 ((44 - 1)* 2 - 0 + 4) = 128
        self.convt5 = nn.ConvTranspose2d(16, 2, kernel_size=(3, 3), stride=2, padding=1)

    def forward(self, x):

        # Flatten to two dimensions before putting into Linear
        x = x.view(-1, self.preprocessor.n_noise)  # [batch_size, n_noise]

        # Projection & Reshape
        x = self.projection(x)               
        x = self.bn_proj(x)
        x = F.relu(x)
        x = x.view(-1, 256, self.init_h, self.init_w)           

        # Stepwise upscaling using reverse convolution layers
        x = F.relu(self.bn1(self.convt1(x)))  
        # print(f"1: {x.shape}")
        x = F.relu(self.bn2(self.convt2(x)))  
        # print(f"2: {x.shape}")
        x = F.relu(self.bn3(self.convt3(x))) 
        # print(f"3: {x.shape}")
        x = F.relu(self.bn4(self.convt4(x))) 
        # print(f"4: {x.shape}")

        # Final layer - Output restricted to [-1, 1] with tanh activation function
        x = torch.tanh(self.convt5(x))   
        # print(f"5: {x.shape}")
        return x

