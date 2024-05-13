# based on https://github.com/dansuh17/alexnet-pytorch/blob/d0c1b1c/model.py
import torch
from torch import nn


# LocalResponseNorm hyperparameters (section 3.3)
LRN_k = 2
LRN_n = 5
LRN_alpha = 10e-4
LRN_beta = 0.75

DROPOUT_p = 0.5 # section 4.2


class AlexNet(nn.Module):
    def __init__(self, num_classes=10_000):
        super().__init__()

        # 논문 상에서는 input size가 224 x 224 이지만
        # 1st conv layer에서 kernel_size=11, stride=4 적용 시 55 x 55가 되려면
        # input size가 227 x 227가 되어야한다.
        # (227pixel - 11kernel + 0padding) / 4stride + 1 = 55
        self.conv_net = nn.Sequential(
            # 1st layer
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4), # (b, 96, 55, 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=LRN_n, alpha=LRN_alpha, beta=LRN_beta, k=LRN_k),
            nn.MaxPool2d(kernel_size=3, stride=2), #  (b, 96, 27, 27)

            # 2nd layer
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2), # (b, 256, 27, 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=LRN_n, alpha=LRN_alpha, beta=LRN_beta, k=LRN_k),
            nn.MaxPool2d(kernel_size=3, stride=2), # (b, 256, 13, 13)

            # 3rd layer
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1), # (b, 384, 13, 13)
            nn.ReLU(),
            
            # 4th layer
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1), # (b, 384, 13, 13)
            nn.ReLU(),

            # 5th layer
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1), # (b, 256, 13, 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # (b, 256, 6, 6)
        )

        # FC layer 이전에 Dropout이 먼저 나오는 구조는 reference에 있는 아래 논문의 구조를 사용한 것 같다
        # https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
        self.classifier = nn.Sequential(
            # 1st layer
            nn.Dropout(p=DROPOUT_p, inplace=True),
            nn.Linear(in_features=256*6*6, out_features=4096),
            nn.ReLU(),

            # 2nd layer
            nn.Dropout(p=DROPOUT_p, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),

            # 3rd layer
            nn.Linear(in_features=4096, out_features=num_classes),
        )

        self.init_bias()

    # section 5
    def init_bias(self):
        for layer in self.conv_net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        
        nn.init.constant_(self.conv_net[4].bias, 1) # 2nd conv layer
        nn.init.constant_(self.conv_net[10].bias, 1) # 4th conv layer
        nn.init.constant_(self.conv_net[12].bias, 1) # 5th conv layer

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(-1, 256*6*6) # flatten
        y_hat = self.classifier(x)
        return y_hat
