import torch
import torchvision.models as models
from torch import Tensor, nn
from torchvision.models import ResNet50_Weights


class Encoder(nn.Module):
    """Encoder model."""

    def __init__(self, encoded_size=(7, 7), finetune=False):
        super().__init__()
        # 加载预训练 resnet
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # 移除最后两层（平均池化和全连接层）
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        # 自适应池化到固定尺寸
        self.pool = nn.AdaptiveAvgPool2d(encoded_size)
        # 默认不微调
        self.finetune(finetune)

    def forward(self, images: Tensor):
        """Extract image features.

        Args:
            images (Tensor): Input images with shape (batch, channels, height, width).

        Returns:
            Tensor: Extracted features with shape (batch, num_pixels, feature_dim).
        """
        # 提取特征
        x: Tensor = self.resnet(images)
        # 池化到固定尺寸
        x: Tensor = self.pool(x)
        # B C H W -> B (H W) C
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        return x

    def finetune(self, finetune=True):
        """
        Fine-tune the later convolutional blocks of ResNet.
    
        Since self.resnet is an nn.Sequential, its children are indexed as:
        0: conv1
        1: bn1
        2: relu
        3: maxpool
        4: layer1
        5: layer2
        6: layer3
        7: layer4
        """
    
        # 先冻结全部 ResNet 参数
        for p in self.resnet.parameters():
            p.requires_grad = False
    
        if finetune:
            # 解冻 layer2, layer3, layer4
            for child in list(self.resnet.children())[5:]:
                for p in child.parameters():
                    p.requires_grad = True
    
            # 整个 backbone 设为 train
            # 如果之后发现不稳定，可以只让 layer2/layer3/layer4 train
            self.resnet.train()
        else:
            self.resnet.eval()                
