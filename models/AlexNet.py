import torch
from torch import nn
from torch.nn import functional as F

from MiCoModel import MiCoModel

class AlexNet(MiCoModel):

    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
        self.n_layers = len(self.get_qlayers())

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits
    
# if __name__ == "__main__":
#     model = AlexNet(10)
#     from MiCoUtils import get_model_size
#     model.set_qscheme(
#         [[8]*model.n_layers, [8]*model.n_layers]
#     )
#     print(get_model_size(model)/8/1024/1024, "MB")

#     from models import resnet_alt_18

#     model = resnet_alt_18(100)
#     model.set_qscheme(
#         [[8]*model.n_layers, [8]*model.n_layers]
#     )
#     print(get_model_size(model)/8/1024/1024, "MB")

#     from models import VGG

#     model = VGG(3, 100)
#     model.set_qscheme(
#         [[8]*model.n_layers, [8]*model.n_layers]
#     )
#     print(get_model_size(model)/8/1024/1024, "MB")