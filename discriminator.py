import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor
        '''
        DCGAN ((𝑊+{𝑝×2}−𝐾)/𝑠+1)
        Feature extraction and downsampling using convolutional layers
        In the DCGAN paper, BatchNorm is recommended for Discriminator layers other than the first layer.
        '''
        # 128x87 → 64x43　 ((87 + 2 - 4) / 2 + 1)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)
        # LeakyReLU only (do not apply BatchNorm to the first layer - recommended by DCGAN paper)

        # 64x43 → 32x21　torch.Size([16, 32, 32, 26])
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # 32x21 → 16x10
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # 16x10 → 8x5
        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # 8x5 → 4x2
        self.conv5 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        # final determination layer - 4x2 → 1x1
        self.conv6 = nn.Conv2d(256, 1, kernel_size=(4, 2), stride=1, padding=0)

        # Dropout (optional - improves model stability)
        self.dropout = nn.Dropout2d(0.3)


    def forward(self, x):
        #Convert to one dimension.

        x = x.view(x.size(0), 1, self.preprocessor.n_mels, -1)  #(batch size, number of channels, height, width)
        # print(f"Input shape Discriminator: {x.shape}")

        # Downsampling through a convolutional layer
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        # print(f"After conv1: {x.shape}")

        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        # print(f"After conv2: {x.shape}")

        # Add dropout (optional)
        x = self.dropout(x)

        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        # print(f"After conv3: {x.shape}")

        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        # print(f"After conv4: {x.shape}")

        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.2)
        # print(f"After conv5: {x.shape}")

        # Final layer - Outputs probabilities between 0 and 1 using sigmoid activation
        x = torch.sigmoid(self.conv6(x))
        # print(f"After conv6: {x.shape}")

        # Transform into a shape with a batch size * 1
        x = x.view(-1, 1)
        # print(f"Final output: {x.shape}")
        return x

