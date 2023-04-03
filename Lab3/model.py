import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self,activate) -> None:
        super().__init__()
        
        if activate == "Relu":
            self.activate = nn.ReLU()
        elif activate == "LeakyRelu":
            self.activate = nn.LeakyReLU()
        elif activate == "Elu":
            self.activate = nn.ELU()
        
        self.firstconv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(1, 51),
                stride=(1, 1),
                padding=(0, 25),
                bias=False
            ),
            nn.BatchNorm2d(16)
        )

        self.depthwiseconv = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(2, 1),
                stride=(1, 1),
                groups=16,
                bias=False
            ),
            nn.BatchNorm2d(32),
            self.activate,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25)
        )

        self.separableconv = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(1, 15),
                stride=(1, 1),
                padding=(0, 7),
                bias=False
            ),
            nn.BatchNorm2d(32),
            self.activate,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25)
        )

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=736, out_features=2, bias=True)
        )


    def forward(self,x):
        x = self.firstconv(x)
        x = self.depthwiseconv(x)
        x = self.separableconv(x)
        x = self.classify(x)
        return x
    


class DeepConvNet(nn.Module):
    def __init__(self,activate) -> None:
        super().__init__()
        if activate == "Relu":
            self.activate = nn.ReLU()
        elif activate == "LeakyRelu":
            self.activate = nn.LeakyReLU()
        else:
            self.activate = nn.ELU()

        self.conv_0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=25,
                kernel_size=(1, 5),
            ),
            nn.Conv2d(
                in_channels=25,
                out_channels=25,
                kernel_size=(2, 1),
            ),
            nn.BatchNorm2d(25),
            self.activate,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=25,
                out_channels=50,
                kernel_size=(1, 5),
            ),
            nn.BatchNorm2d(50),
            self.activate,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=50,
                out_channels=100,
                kernel_size=(1, 5),
            ),
            nn.BatchNorm2d(100),
            self.activate,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=100,
                out_channels=200,
                kernel_size=(1, 5),
            ),
            nn.BatchNorm2d(200),
            self.activate,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=8600, out_features=2)
        )


    def forward(self,input):
        x = self.conv_0(input)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.classify(x)
        return x