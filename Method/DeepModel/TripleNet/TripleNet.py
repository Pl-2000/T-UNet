import torch
import torch.nn as nn
from torchvision.models import vgg16

from Method.Utils.VGG16 import VGG16_layer1, VGG16_layer2, VGG16_layer3, VGG16_layer4, VGG16_layer5
from Decoder import decoder_layer5, decoder_layer4, decoder_layer3, decoder_layer2, decoder_layer1


class TripleNet(nn.Module):
    def __init__(self):
        super(TripleNet, self).__init__()

        self.backbone = backbone_vgg16()
        self.main_stream_layer1 = VGG16_layer1()  # 3-64
        self.main_stream_layer2 = VGG16_layer2()  # 64-128
        self.main_stream_layer3 = VGG16_layer3()  # 128-256
        self.main_stream_layer4 = VGG16_layer4()  # 256-512
        self.main_stream_layer5 = VGG16_layer5()  # 512-512

        self.mbssca1 = MBSSCA(64)
        self.mbssca2 = MBSSCA(128)
        self.mbssca3 = MBSSCA(256)
        self.mbssca4 = MBSSCA(512)
        self.mbssca5 = MBSSCA(512)

        self.decoder_layer5 = decoder_layer5(512, 512)
        self.sa5 = SpatialAttention()
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_trans5 = nn.ConvTranspose2d(512, 512, kernel_size=(2, 2), stride=(2, 2))

        self.ca4 = ChannelAttention(in_channels=512 * 2, ratio=8)
        self.decoder_layer4 = decoder_layer4(512, 256)
        self.sa4 = SpatialAttention()
        self.bn4 = nn.BatchNorm2d(256)
        self.conv_trans4 = nn.ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2))

        self.ca3 = ChannelAttention(in_channels=256 * 2, ratio=8)
        self.decoder_layer3 = decoder_layer3(256, 128)
        self.sa3 = SpatialAttention()
        self.bn3 = nn.BatchNorm2d(128)
        self.conv_trans3 = nn.ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2))

        self.ca2 = ChannelAttention(in_channels=128 * 2, ratio=8)
        self.decoder_layer2 = decoder_layer2(128, 64)
        self.sa2 = SpatialAttention()
        self.bn2 = nn.BatchNorm2d(64)
        self.conv_trans2 = nn.ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=(2, 2))

        self.ca1 = ChannelAttention(in_channels=64 * 2, ratio=8)
        self.decoder_layer1 = decoder_layer1(64, 3)
        self.sa1 = SpatialAttention()
        self.bn1 = nn.BatchNorm2d(3)

        self.output = nn.Conv2d(3, 1, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_t1, x_t2):
        t1_stream_layer1, t1_stream_layer2, t1_stream_layer3, t1_stream_layer4, t1_stream_layer5 = self.backbone(x_t1)
        t2_stream_layer1, t2_stream_layer2, t2_stream_layer3, t2_stream_layer4, t2_stream_layer5 = self.backbone(x_t2)

        main_stream_input = torch.abs(x_t1 - x_t2)
        main_stream_layer1 = self.main_stream_layer1(main_stream_input)
        main_stream_layer1 = self.mbssca1(main_stream_layer1, t1_stream_layer1, t2_stream_layer1)
        main_stream_layer2 = self.main_stream_layer2(main_stream_layer1)
        main_stream_layer2 = self.mbssca2(main_stream_layer2, t1_stream_layer2, t2_stream_layer2)
        main_stream_layer3 = self.main_stream_layer3(main_stream_layer2)
        main_stream_layer3 = self.mbssca3(main_stream_layer3, t1_stream_layer3, t2_stream_layer3)
        main_stream_layer4 = self.main_stream_layer4(main_stream_layer3)
        main_stream_layer4 = self.mbssca4(main_stream_layer4, t1_stream_layer4, t2_stream_layer4)
        main_stream_layer5 = self.main_stream_layer5(main_stream_layer4)
        main_stream_layer5 = self.mbssca5(main_stream_layer5, t1_stream_layer5, t2_stream_layer5)

        decoder_layer5 = self.decoder_layer5(main_stream_layer5)
        decoder_layer5 = self.bn5(self.sa5(decoder_layer5) * decoder_layer5)
        decoder_layer5 = self.conv_trans5(decoder_layer5)

        decoder_cat4 = torch.cat([decoder_layer5, main_stream_layer4], dim=1)
        decoder_layer4 = self.ca4(decoder_cat4) * decoder_cat4
        decoder_layer4 = self.decoder_layer4(decoder_layer4)
        decoder_layer4 = self.bn4(self.sa4(decoder_layer4) * decoder_layer4)
        decoder_layer4 = self.conv_trans4(decoder_layer4)

        decoder_cat3 = torch.cat([decoder_layer4, main_stream_layer3], dim=1)
        decoder_layer3 = self.ca3(decoder_cat3) * decoder_cat3
        decoder_layer3 = self.decoder_layer3(decoder_layer3)
        decoder_layer3 = self.bn3(self.sa3(decoder_layer3) * decoder_layer3)
        decoder_layer3 = self.conv_trans3(decoder_layer3)

        decoder_cat2 = torch.cat([decoder_layer3, main_stream_layer2], dim=1)
        decoder_layer2 = self.ca2(decoder_cat2) * decoder_cat2
        decoder_layer2 = self.decoder_layer2(decoder_layer2)
        decoder_layer2 = self.bn2(self.sa2(decoder_layer2) * decoder_layer2)
        decoder_layer2 = self.conv_trans2(decoder_layer2)

        decoder_cat1 = torch.cat([decoder_layer2, main_stream_layer1], dim=1)
        decoder_layer1 = self.ca1(decoder_cat1) * decoder_cat1
        decoder_layer1 = self.decoder_layer1(decoder_layer1)
        decoder_layer1 = self.bn1(self.sa1(decoder_layer1) * decoder_layer1)

        output = self.output(decoder_layer1)

        return self.sigmoid(output)
