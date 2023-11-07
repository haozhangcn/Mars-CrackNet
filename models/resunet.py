import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class block(nn.Module):
    def __init__(self, input_channels, output_channels, stride, dilation=1, mode='A'):
        super(block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_channels),
        )
        self.mode = mode
        if self.mode == 'A':
            self.skip = nn.Sequential()
        elif self.mode == 'B':
            self.skip = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1)
            )

    def forward(self, x):
        x_conv = self.conv(x)
        out = x_conv + self.skip(x)
        return F.relu(out)


class decoder(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_channels),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_e, x_up):
        x_up = F.interpolate(x_up, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat((x_e, x_up), 1)
        x_c = self.conv(x)
        skip = self.skip(x)
        x_c += skip
        x_c = self.relu(x_c)
        return x_c


class Res_UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet18(pretrained=False)
        self.conv_original_size0 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
            )

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.block4 = nn.Sequential(
            block(512, 512, 1, 2),
        )

        self.decoder1 = decoder(filters[3] + filters[2], filters[2], stride=1)
        self.decoder2 = decoder(filters[2] + filters[1], filters[1], stride=1)
        self.decoder3 = decoder(filters[1] + filters[0], filters[0], stride=1)
        self.decoder4 = decoder(filters[0] + filters[0], filters[0], stride=1)

        # for p in self.parameters():
        #     p.requires_grad = False

        self.final = nn.Conv2d(filters[0], num_classes, kernel_size=1, stride=1)


    def forward(self, x):
        x_ = self.conv_original_size0(x)

        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        b = self.encoder4(e3)

        b = self.block4(b)
        up1 = self.decoder1(e3, b)
        up2 = self.decoder2(e2, up1)
        up3 = self.decoder3(e1, up2)
        up4 = self.decoder4(x_, up3)

        f = self.final(up4)
        # return torch.sigmoid(f)
        return b, up1, up2, up3, up4, torch.sigmoid(f) # only for test


if __name__ == '__main__':
    model = Res_UNet(1)
    print(model)
    from torchsummary import summary
    summary(model.cuda(), input_size=(3, 224, 224))
