import torch.nn as nn


class decoder_layer5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_layer5, self).__init__()
        self.nn_modules = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=0.6),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=0.6),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=0.6),
        )

    def forward(self, x):
        return self.nn_modules(x)


class decoder_layer4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_layer4, self).__init__()
        self.nn_modules = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels),
            nn.Dropout(p=0.6),
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels),
            nn.Dropout(p=0.6),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=0.6),
        )

    def forward(self, x):
        return self.nn_modules(x)


class decoder_layer3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_layer3, self).__init__()
        self.nn_modules = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels),
            nn.Dropout(p=0.6),
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels),
            nn.Dropout(p=0.6),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=0.6),
        )

    def forward(self, x):
        return self.nn_modules(x)


class decoder_layer2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_layer2, self).__init__()
        self.nn_modules = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels),
            nn.Dropout(p=0.6),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=0.6),
        )

    def forward(self, x):
        return self.nn_modules(x)


class decoder_layer1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_layer1, self).__init__()
        self.nn_modules = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels),
            nn.Dropout(p=0.6),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=0.6),
        )

    def forward(self, x):
        return self.nn_modules(x)
