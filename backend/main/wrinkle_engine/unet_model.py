"""Full assembly of the U-Net parts (vendored from rmsandu/FFHQ-detect-face-wrinkles)"""

import torch
import torch.nn as nn
from wrinkle_engine.unet_parts import Up, OutConv
from torchvision.models import resnet50, ResNet50_Weights


class UNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        bilinear=False,
        pretrained=True,
        freeze_encoder=False,
        use_attention=False,
    ):

        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_attention = use_attention
        self.pretrained = pretrained
        self.freeze_encoder = freeze_encoder

        # Load pretrained ResNet50 as encoder
        if pretrained is True:
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            resnet = resnet50(weights=None)

        self.input_layer = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.pool = resnet.maxpool
        self.encoder1 = resnet.layer1  # 256 channels
        self.encoder2 = resnet.layer2  # 512 channels
        self.encoder3 = resnet.layer3  # 1024 channels
        self.encoder4 = resnet.layer4  # 2048 channels

        if freeze_encoder is True:
            for param in resnet.parameters():
                param.requires_grad = False
        else:
            for param in resnet.parameters():
                param.requires_grad = True

        if not pretrained:
            self._initialize_weights()

        # Decoder path
        self.up1 = Up(
            x1_channels=2048, x2_channels=1024, out_channels=1024, bilinear=bilinear
        )
        self.up2 = Up(1024, 512, 512, bilinear=bilinear)
        self.up3 = Up(512, 256, 256, bilinear=bilinear)
        self.up4 = Up(256, 64, 128, bilinear=bilinear)
        self.up5 = Up(128, 0, 64, bilinear=bilinear)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x0 = self.input_layer(x)
        x1 = self.encoder1(self.pool(x0))
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        if self.use_attention:
            x3_att = self.att1(g=x4, x=x3)
            x2_att = self.att2(g=x3, x=x2)
            x1_att = self.att3(g=x2, x=x1)
            x0_att = self.att4(g=x1, x=x0)
        else:
            x3_att, x2_att, x1_att, x0_att = x3, x2, x1, x0

        x = self.up1(x4, x3_att)
        x = self.up2(x, x2_att)
        x = self.up3(x, x1_att)
        x = self.up4(x, x0_att)
        x = self.up5(x, None)
        logits = self.outc(x)
        return logits if self.n_classes == 1 else torch.softmax(logits, dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
