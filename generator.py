import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor
        #Retrieve an attribute (variable) from an object by name
        self.mode = getattr(preprocessor, 'mode', 'stft')
        
        # Projection layer
        if self.mode == 'mel':
            self.init_h = 4
            self.out_channels = 1
        else:
            self.init_h = 17
            self.out_channels = 2
            
        self.init_w = 3
        self.projection = nn.Linear(self.preprocessor.n_noise, 256 * self.init_h * self.init_w)
        self.bn_proj = nn.BatchNorm1d(256 * self.init_h * self.init_w)

        # Expansion using convolutional transposition layers
        if self.mode == 'mel':
            # 4x3 -> 8x6
            self.convt1 = nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=2, padding=1)
            # 8x6 -> 16x11
            self.convt2 = nn.ConvTranspose2d(128, 64, kernel_size=(4, 3), stride=2, padding=1)
            # 16x11 -> 32x22
            self.convt3 = nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=2, padding=1)
            # 32x22 -> 64x44
            self.convt4 = nn.ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=2, padding=1)
            # 64x44 -> 128x87
            self.convt5 = nn.ConvTranspose2d(16, self.out_channels, kernel_size=(4, 3), stride=2, padding=1)
        else:
            # 17x3 → 33x6
            self.convt1 = nn.ConvTranspose2d(256, 128, kernel_size=(3, 4), stride=2, padding=1)
            # 33x6 → 65x11
            self.convt2 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=2, padding=1)
            # 65x11 → 129x22
            self.convt3 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 4), stride=2, padding=1)
            # 129x22 → 257x44
            self.convt4 = nn.ConvTranspose2d(32, 16, kernel_size=(3, 4), stride=2, padding=1)
            # 257x44 → 513x87
            self.convt5 = nn.ConvTranspose2d(16, self.out_channels, kernel_size=(3, 3), stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(16)

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
        #x = x[:, :, :, :173]
        #print(f"5: {x.shape}")
        return x

