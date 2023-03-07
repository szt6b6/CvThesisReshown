import torch
from torch import nn

# make modification, only the channels are changed.
# input C1xHxW, output C2xHxW
class Double_Conv(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.seq(x)


# Set the input 1x512x512, then after down: 64x256x256->128x128x128->256x64x64->512x32x32
class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.dconv1 = Double_Conv(1, 64)
        self.dconv2 = Double_Conv(64, 128)
        self.dconv3 = Double_Conv(128, 256)
        self.dconv4 = Double_Conv(256, 512)

        self.down1 = nn.MaxPool2d(2, 2)
        self.down2 = nn.MaxPool2d(2, 2)
        self.down3 = nn.MaxPool2d(2, 2)
        self.down4 = nn.MaxPool2d(2, 2)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2, 0)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2, 0)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2, 0)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2, 0)

        self.dconv4_ = Double_Conv(1024, 512)
        self.dconv3_ = Double_Conv(512, 256)
        self.dconv2_ = Double_Conv(256, 128)
        self.dconv1_ = Double_Conv(128, 64)

        self.bottom_conv = Double_Conv(512, 1024)

        
        self.final_conv = nn.Conv2d(64, 2, 1, 1)



    def forward(self, x):
        x1 = self.dconv1(x)
        x2 = self.dconv2(self.down1(x1))
        x3 = self.dconv3(self.down2(x2))
        x4 = self.dconv4(self.down3(x3))
        
        x5 = self.up4(self.bottom_conv(self.down4(x4)))
        
        x_out = self.up3(self.dconv4_(torch.concat([x4, x5], dim=1)))
        x_out = self.up2(self.dconv3_(torch.concat([x_out, x3], dim=1)))
        x_out = self.up1(self.dconv2_(torch.concat([x_out, x2], dim=1)))
        x_out = self.final_conv(self.dconv1_(torch.concat([x_out, x1], dim=1)))

        return x_out


if __name__ == "__main__":
    x = torch.randn((3, 1, 512, 512))


    # Set the C=1, H=512, W=512
    # Have modification refet to U-net img in model_architectures.md 
    unet = UNet()

    out = unet(x)

    print(out.shape)