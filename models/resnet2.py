import torch
from torch import nn

class ResNet18(nn.Module):

    def __init__(self, num_classes=200):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # -------- STAGE 1 (64) --------
        self.block1_1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64)
        )

        self.block1_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64)
        )

        # -------- STAGE 2 (128) --------
        self.s2 = nn.Sequential(
            nn.Conv2d(64, 128, 1, 2, bias=False),
            nn.BatchNorm2d(128)
        )

        self.block2_1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.block2_2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128)
        )

        # -------- STAGE 3 (256) --------
        self.s3 = nn.Sequential(
            nn.Conv2d(128, 256, 1, 2, bias=False),
            nn.BatchNorm2d(256)
        )

        self.block3_1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256)
        )

        self.block3_2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256)
        )

        # -------- STAGE 4 (512) --------
        self.s4 = nn.Sequential(
            nn.Conv2d(256, 512, 1, 2, bias=False),
            nn.BatchNorm2d(512)
        )

        self.block4_1 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512)
        )

        self.block4_2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):

        out = self.relu(self.bn1(self.conv1(x)))

        # ---- STAGE 1 ----
        out = self.relu(self.block1_1(out) + out)
        out = self.relu(self.block1_2(out) + out)

        # ---- STAGE 2 ----
        identity = self.s2(out)
        out = self.relu(self.block2_1(out) + identity)
        out = self.relu(self.block2_2(out) + out)

        # ---- STAGE 3 ----
        identity = self.s3(out)
        out = self.relu(self.block3_1(out) + identity)
        out = self.relu(self.block3_2(out) + out)

        # ---- STAGE 4 ----
        identity = self.s4(out)
        out = self.relu(self.block4_1(out) + identity)
        out = self.relu(self.block4_2(out) + out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out