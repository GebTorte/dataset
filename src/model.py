import torch.nn as nn
import torch

class DoubleConv(nn.Module):
    """Two consecutive convolution blocks with BatchNorm and ReLU.

    This is the basic building block of U-Net.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Basic U-Net for semantic segmentation.

    Args:
        in_channels: Number of input channels (1 for binary/grayscale)
        num_classes: Number of output classes (3 for our problem)
        base_channels: Number of channels in first layer (default: 32)
    """

    def __init__(self, in_channels=1, num_classes=4, base_channels=32):
        super().__init__()

        # ENCODER
        self.enc1 = DoubleConv(in_channels, base_channels)  # 32 channels
        self.enc2 = DoubleConv(base_channels, base_channels * 2)  # 64 channels
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)  # 128 channels
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)  # 256 channels

        self.pool = nn.MaxPool2d(2)  # Downsampling by 2

        # BOTTLENECK (Bottom of U)
        self.bottleneck = DoubleConv(
            base_channels * 8, base_channels * 16
        )  # 512 channels

        # DECODER
        self.up4 = nn.ConvTranspose2d(
            base_channels * 16, base_channels * 8, kernel_size=2, stride=2
        )
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        # FINAL LAYER: 1×1 convolution to get class scores
        self.out = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path (save features for skip connections)
        enc1 = self.enc1(x)  # 1024×1024×32
        enc2 = self.enc2(self.pool(enc1))  # 512×512×64
        enc3 = self.enc3(self.pool(enc2))  # 256×256×128
        enc4 = self.enc4(self.pool(enc3))  # 128×128×256

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))  # 64×64×512

        # Decoder path (with skip connections)
        dec4 = self.up4(bottleneck)  # 128×128×256
        dec4 = torch.cat([dec4, enc4], dim=1)  # Concatenate skip connection
        dec4 = self.dec4(dec4)  # 128×128×256

        dec3 = self.up3(dec4)  # 256×256×128
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)  # 256×256×128

        dec2 = self.up2(dec3)  # 512×512×64
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)  # 512×512×64

        dec1 = self.up1(dec2)  # 1024×1024×32
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)  # 1024×1024×32

        # Final output
        out = self.out(dec1)  # 1024×1024×3
        return out




if __name__ == "__main__":
    # Create model and inspect
    model = UNet(in_channels=11, num_classes=3, base_channels=32).to(device)
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with 1024×1024 input
    dummy_input = torch.randn(1, 1, 1024, 1024).to(device)
    dummy_output = model(dummy_input)
