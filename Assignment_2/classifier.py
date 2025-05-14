import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

NUM_CLASSES = 21


class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.conv3 = nn.Conv2d(32, 16, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 26 * 26, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size()[0], 16 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, base_channels: int,
                 stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        out_channels = base_channels * 4

        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels)

        self.conv3 = nn.Conv2d(base_channels, out_channels, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ResNet50(nn.Module):
    def __init__(self, num_classes: int = 21):
        super().__init__()
        ############################# Stem ###################################
        # Conv1 7x7 kernel, output channel 64, stride 2로 112x112 해상도로 생성
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 한번 더 maxpool을 2x2마다 수행하면, 112 -> 56으로 해상도가 줄어듦
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ########################### Residual stage ###########################
        # Stem에서 64개의 채널로 변환되었으므로 conv2_x에 넣을때 64개의 채널로 설정
        self.layer1 = self._make_layer(64,  64, block_cnt=3, stride=1)  # 최종 256 채널 출력
        self.layer2 = self._make_layer(256, 128, block_cnt=4, stride=2) # 최종 512 채널
        self.layer3 = self._make_layer(512, 256, block_cnt=6, stride=2) # 1024
        self.layer4 = self._make_layer(1024, 512, block_cnt=3, stride=2) # 2048

        # ---------------- Classification head ----------------
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * 4, num_classes)

        # ---------------- Init ----------------
        self._init_weights()

    def _make_layer(self, in_channels: int, base_channels: int,
                    block_cnt: int, stride: int) -> nn.Sequential:
        """
        Residual stage를 만들 때 사용하는 함수
        in_channels : 이전 레이어에서 계산된 채널의 개수 (입력 채널 개수)
        base_channels : Bottleneck블록의 위에 두개 Convolution 채널 개수
        block_cnt : 논문에 나와있는 블록 반복 수 설정 (conv2_x : 3개, 3_x : 4개, 4_x : 6개, 5_x : 3개)
        stride : 기본적으로 2로 설정
        """
        downsample = None
        out_channels = base_channels * 4 # 출력 채널
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers: list[nn.Module] = []
        layers.append(Bottleneck(in_channels, base_channels,
                                 stride=stride, downsample=downsample))
        for _ in range(1, block_cnt):
            layers.append(Bottleneck(out_channels, base_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)  # logits; apply sigmoid in loss fn

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)