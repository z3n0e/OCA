#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#
from typing import Optional, List, Tuple
from cv2 import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import orthogonal
from torchvision.models import alexnet, vgg19_bn, vgg13_bn

class BasicBlock(nn.Module):
    """Resnet basic block module."""

    expansion = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            base_width: int = 64,
            nonlin: bool = True,
            embedding_dim: Optional[int] = None,
    ) -> None:
        """Construct a BasicBlock module.

        :param inplanes: Number of input channels.
        :param planes: Number of output channels.
        :param stride: Stride size.
        :param downsample: Down-sampling for residual path.
        :param base_width: Base width of the block.
        :param nonlin: Whether to apply non-linearity before output.
        :param embedding_dim: Size of the output embedding dimension.
        """
        super(BasicBlock, self).__init__()
        if base_width / 64 > 1:
            raise ValueError("Base width >64 does not work for BasicBlock")
        if embedding_dim is not None:
            planes = embedding_dim
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1,
                               stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.nonlin = nonlin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        residual = x

        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        if self.bn2 is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.nonlin:
            out = self.relu(out)

        return out


class BasisChange(nn.Module):
    """
    applying a series of change of basis
    """

    def __init__(self, embedding_dim, to_add):
        super().__init__()
        self.to_add = to_add
        self.ortholinear_p = orthogonal(
            nn.Linear(embedding_dim, embedding_dim, bias=False))
        self.ortholinear_old = orthogonal(
            nn.Linear(embedding_dim, to_add, bias=False))
        self.ortholinear_old2 = orthogonal(
            nn.Linear(embedding_dim, embedding_dim, bias=False))
        self.criterion = nn.CosineSimilarity()

    def forward(self, x, y):
        """
        x is the old feature while y is the new feature
        """
        new_feature = torch.cat(
            [F.normalize(self.ortholinear_old(x)), self.ortholinear_p(y)], dim=1)
        bct_feature = self.ortholinear_old2(new_feature[:, :-self.to_add])
        to_add_feature = new_feature[:, -self.to_add:]
        return bct_feature, to_add_feature


class Bottleneck(nn.Module):
    """Resnet bottleneck block module."""

    expansion = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            base_width: int = 64,
            nonlin: bool = True,
            embedding_dim: Optional[int] = None,
    ) -> None:
        """Construct a Bottleneck module.

        :param inplanes: Number of input channels.
        :param planes: Number of output channels.
        :param stride: Stride size.
        :param downsample: Down-sampling for residual path.
        :param base_width: Base width of the block.
        :param nonlin: Whether to apply non-linearity before output.
        :param embedding_dim: Size of the output embedding dimension.
        """
        super(Bottleneck, self).__init__()
        width = int(planes * base_width / 64)
        if embedding_dim is not None:
            out_dim = embedding_dim
        else:
            out_dim = planes * self.expansion
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1,
                               stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_dim, kernel_size=1, stride=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.nonlin = nonlin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.nonlin:
            out = self.relu(out)

        return out


class ResNet(nn.Module):
    """Resnet module."""

    def __init__(
            self,
            block: nn.Module,
            layers: List[int],
            num_classes: int = 1000,
            base_width: int = 64,
            embedding_dim: Optional[int] = None,
            last_nonlin: bool = True,
            norm_feature: bool = False,
    ) -> None:
        """Construct a ResNet module.

        :param block: Block module to use in Resnet architecture.
        :param layers: List of number of blocks per layer.
        :param num_classes: Number of classes in the dataset. It is used to
            form linear classifier weights.
        :param base_width: Base width of the blocks.
        :param embedding_dim: Size of the output embedding dimension.
        :param last_nonlin: Whether to apply non-linearity before output.
        :param norm_feature: Whether to normalized output embeddings.
        """
        self.inplanes = 64
        super(ResNet, self).__init__()
        print("==> Using last_nonlin model", last_nonlin)

        self.OUTPUT_SHAPE = [embedding_dim, 1, 1]
        self.is_normalized = norm_feature
        self.base_width = base_width
        if self.base_width // 64 > 1:
            print(f"==> Using {self.base_width // 64}x wide model")

        if embedding_dim is not None:
            print("Using given embedding dimension = {}".format(embedding_dim))
            self.embedding_dim = embedding_dim
        else:
            self.embedding_dim = 512 * block.expansion

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 64, layers[0], embedding_dim=64 * block.expansion
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            embedding_dim=128 * block.expansion,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            embedding_dim=256 * block.expansion,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            nonlin=last_nonlin,
            embedding_dim=self.embedding_dim,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Conv2d(self.embedding_dim, num_classes, kernel_size=1,
                            stride=1, bias=False)

    def _make_layer(
            self,
            block: nn.Module,
            planes: int,
            blocks: int,
            embedding_dim: int,
            stride: int = 1,
            nonlin: bool = True
    ):
        """Make a layer of resnet architecture.

        :param block: Block module to use in this layer.
        :param planes: Number of output channels.
        :param blocks: Number of blocks in this layer.
        :param embedding_dim: Size of the output embedding dimension.
        :param stride: Stride size.
        :param nonlin: Whether to apply non-linearity before output.
        :return:
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            dconv = nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1,
                              stride=stride, bias=False)
            dbn = nn.BatchNorm2d(planes * block.expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        last_downsample = None

        layers = []
        if blocks == 1:  # If this layer has only one-block
            if stride != 1 or self.inplanes != embedding_dim:
                dconv = nn.Conv2d(self.inplanes, embedding_dim, kernel_size=1,
                                  stride=stride, bias=False)
                dbn = nn.BatchNorm2d(embedding_dim)
                if dbn is not None:
                    last_downsample = nn.Sequential(dconv, dbn)
                else:
                    last_downsample = dconv
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride,
                    last_downsample,
                    base_width=self.base_width,
                    nonlin=nonlin,
                    embedding_dim=embedding_dim,
                )
            )
            return nn.Sequential(*layers)
        else:
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride,
                    downsample,
                    base_width=self.base_width,
                )
            )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks - 1):
            layers.append(
                block(self.inplanes, planes,
                      base_width=self.base_width)
            )

        if self.inplanes != embedding_dim:
            dconv = nn.Conv2d(self.inplanes, embedding_dim, stride=1,
                              kernel_size=1,
                              bias=False)
            dbn = nn.BatchNorm2d(embedding_dim)
            if dbn is not None:
                last_downsample = nn.Sequential(dconv, dbn)
            else:
                last_downsample = dconv
        layers.append(
            block(
                self.inplanes,
                planes,
                downsample=last_downsample,
                base_width=self.base_width,
                nonlin=nonlin,
                embedding_dim=embedding_dim,
            )
        )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply forward pass.

        :param x: input to the model with shape (N, C, H, W).
        :return: Tuple of (logits, embedding)
        """
        x = self.conv1(x)

        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        feature = self.avgpool(x)
        if self.is_normalized:
            feature = F.normalize(feature)

        x = self.fc(feature)
        x = x.view(x.size(0), -1)

        return x, feature


def ResNet18(num_classes: int,
             embedding_dim: int,
             last_nonlin: bool = True,
             norm_feature: bool = True,
             **kwargs) -> nn.Module:
    """Get a ResNet18 model.

    :param num_classes: Number of classes in the dataset.
    :param embedding_dim: Size of the output embedding dimension.
    :param last_nonlin: Whether to apply non-linearity before output.
    :return: ResNet18 Model.
    """
    return ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        last_nonlin=last_nonlin,
        norm_feature=norm_feature
    )


def ResNet50(num_classes: int,
             embedding_dim: int,
             last_nonlin: bool = True,
             norm_feature: bool = True,
             **kwargs) -> nn.Module:
    """Get a ResNet50 model.

    :param num_classes: Number of classes in the dataset.
    :param embedding_dim: Size of the output embedding dimension.
    :param last_nonlin: Whether to apply non-linearity before output.
    :return: ResNet50 Model.
    """
    return ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        last_nonlin=last_nonlin,
        norm_feature=norm_feature
    )


class Backbone(nn.Module):
    """Construct a Backbone module, where the backbone can be supported
        with alexnet, vgg13_bn, vgg19_bn.

        :param num_classes: Number of classes in the dataset. It is used to
            form linear classifier weights.
        :param embedding_dim: Size of the output embedding dimension.
        :param norm_feature: Whether to normalized output embeddings.
        :param architecture: Network architecture of the backbone model.
    """

    def __init__(self,
                 num_classes: int = 1000,
                 embedding_dim: Optional[int] = None,
                 norm_feature: bool = False,
                 architecture='alexnet') -> None:
        super(Backbone, self).__init__()
        if architecture == 'alexnet':
            self.alexnet = alexnet(num_classes=embedding_dim)
        elif architecture == 'vgg19_bn':
            self.alexnet = vgg19_bn(num_classes=embedding_dim)
        elif architecture == 'vgg13_bn':
            self.alexnet = vgg13_bn(num_classes=embedding_dim)
        self.embedding_dim = embedding_dim
        self.norm_feature = norm_feature
        self.fc = nn.Linear(embedding_dim, num_classes, bias=False)

    def forward(self, x):
        feature = self.alexnet(x)
        if self.norm_feature:
            feature = F.normalize(feature)
        output = self.fc(feature).reshape(feature.size(0), -1)
        return output, feature


def BackboneBase(num_classes: int,
                 embedding_dim: int,
                 norm_feature: bool = True,
                 architecture='alexnet',
                 **kwargs) -> nn.Module:
    """Get a Backbone module.

    :param num_classes: Number of classes in the dataset. It is used to
            form linear classifier weights.
    :param embedding_dim: Size of the output embedding dimension.
    :param norm_feature: Whether to normalized output embeddings.
    :param architecture: Network architecture of the backbone model.
    :return: a Backbone model with architecture as specified.
    """
    return Backbone(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        norm_feature=norm_feature,
        architecture=architecture
    )


class ResNet_BT2(nn.Module):
    """
    a Backbone+Basis Transformation module
    """

    def __init__(self,
                 block: nn.Module,
                 layers: List[int],
                 num_classes: int = 1000,
                 base_width: int = 64,
                 embedding_dim: Optional[int] = None,
                 embedding_dim_old: Optional[int] = None,
                 last_nonlin: bool = True,
                 norm_feature: bool = False,
                 to_add_dim: int = 32,
                 C: float = 3,
                 architecture: str = 'resnet50') -> None:
        """Construct a Backbone+Basis Transformation module

        :param block: Block module to use in Resnet architecture 
            (if architecture is specified as resnet50).
        :param layers: List of number of blocks per layer
            (if architecture is specified as resnet50).
        :param num_classes: Number of classes in the dataset. It is used to
            form linear classifier weights.
        :param base_width: Base width of the blocks (for resnet50).
        :param embedding_dim: Size of the new embedding dimension.
        :param embedding_dim_old: Size of the old embedding dimension.
        :param last_nonlin: Whether to apply non-linearity before output.
        :param norm_feature: Whether to normalized output embeddings.
        :param to_add_dim: Size of the additional dimension, so that the final 
            output dimension is to_add_dim + embedding_dim.
        :param C: float.
        :param architecture: the architecture to use for backbone, chosen from
            resnet50, alexnet, , vgg13_bn and vgg19_bn.
        """
        super(ResNet_BT2, self).__init__()
        if embedding_dim_old == None:
            embedding_dim_old = embedding_dim
        self.embedding_dim_old = embedding_dim_old

        self.resnet = ResNet(
                block=block,
                layers=layers,
                embedding_dim=embedding_dim+embedding_dim_old,
                last_nonlin=last_nonlin,
                norm_feature=False)
        
        self.embedding_dim = embedding_dim
        self.norm_feature = norm_feature
        self.fc3 = nn.Conv2d(embedding_dim+embedding_dim_old, embedding_dim_old, kernel_size=1,
                             stride=1, bias=True)
        self.fc_out = nn.Linear(embedding_dim, num_classes, bias=False)
        self.to_add_dim = to_add_dim
        self.ortholinear_p = orthogonal(
            nn.Linear(embedding_dim, embedding_dim, bias=False))
        self.ortholinear_old2 = orthogonal(
            nn.Linear(embedding_dim_old, embedding_dim_old, bias=False))
        self.C = C

    def forward(self, x):
        """
        Forward pass
        :return output: classification logits.
        :return bct_feature: the embedding layer that is trained to
            match the old embedding.
        :return feature: the embedding layer that is trained to
            match the new embedding.
        :return new_feature: the merged embedding that is simultaneously
            backward compatible with the old embedding and retains inforation
            of an independently trained new model.
        """
        _, feature = self.resnet(x)
        old_feature = self.fc3(F.relu(feature))
        feature = feature.reshape(feature.size(0), -1)[:,
                                                       :self.embedding_dim]
        old_feature = old_feature.reshape(old_feature.size(0), -1)

        if self.norm_feature:
            old_feature = F.normalize(old_feature)
            feature = F.normalize(feature)

        new_feature = torch.cat([F.normalize(old_feature[:, :self.to_add_dim]),
                                 self.ortholinear_p(self.C*feature)], dim=1)
        bct_feature = self.ortholinear_old2(
            new_feature[:, :self.embedding_dim_old])
        
        # final embedding size is self.embedding_dim + self.to_add_dim
        to_add_feature = new_feature[:, -(self.embedding_dim +
                                          self.to_add_dim - self.embedding_dim_old):]
        new_feature = torch.cat([bct_feature, to_add_feature], dim=1)

        bct_feature = bct_feature.view(bct_feature.size(0), -1)
        to_add_feature = to_add_feature.view(to_add_feature.size(0), -1)
        new_feature = torch.cat([bct_feature, to_add_feature], dim=1)
        output = self.fc_out(feature)

        return output, bct_feature, feature, new_feature


def ResNet50_BT2(num_classes: int,
                   embedding_dim: int,
                   embedding_dim_old: int = None,
                   last_nonlin: bool = True,
                   norm_feature: bool = True,
                   to_add_dim=32,
                   C: float = 3,
                   architecture: str = 'resnet50',
                   **kwargs) -> nn.Module:
    """Get a Backbone+Basis Transformation model.
        (it has been extended to
         support other model architectures other than resnet50)
        :param block: Block module to use in Resnet architecture 
            (if architecture is specified as resnet50).
        :param layers: List of number of blocks per layer
            (if architecture is specified as resnet50).
        :param num_classes: Number of classes in the dataset. It is used to
            form linear classifier weights.
        :param base_width: Base width of the blocks (for resnet50).
        :param embedding_dim: Size of the new embedding dimension.
        :param embedding_dim_old: Size of the old embedding dimension.
        :param last_nonlin: Whether to apply non-linearity before output.
        :param norm_feature: Whether to normalized output embeddings.
        :param to_add_dim: Size of the additional dimension, so that the final 
            output dimension is to_add_dim + embedding_dim.
        :param C: float.
        :param architecture: the architecture to use for backbone, chosen from
            resnet50, alexnet, , vgg13_bn and vgg19_bn.

        :return: Backbone+Basis Transformation model.
    """
    return ResNet_BT2(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        embedding_dim_old=embedding_dim_old,
        last_nonlin=last_nonlin,
        norm_feature=norm_feature,
        to_add_dim=to_add_dim,
        C=C,
        architecture=architecture,
    )

def ResNet18_BT2(num_classes: int,
                   embedding_dim: int,
                   embedding_dim_old: int = None,
                   last_nonlin: bool = True,
                   norm_feature: bool = True,
                   to_add_dim=32,
                   C: float = 3,
                   architecture: str = 'resnet18',
                   **kwargs) -> nn.Module:
    """Get a Backbone+Basis Transformation model.
        (it has been extended to
         support other model architectures other than resnet50)
        :param block: Block module to use in Resnet architecture 
            (if architecture is specified as resnet50).
        :param layers: List of number of blocks per layer
            (if architecture is specified as resnet50).
        :param num_classes: Number of classes in the dataset. It is used to
            form linear classifier weights.
        :param base_width: Base width of the blocks (for resnet50).
        :param embedding_dim: Size of the new embedding dimension.
        :param embedding_dim_old: Size of the old embedding dimension.
        :param last_nonlin: Whether to apply non-linearity before output.
        :param norm_feature: Whether to normalized output embeddings.
        :param to_add_dim: Size of the additional dimension, so that the final 
            output dimension is to_add_dim + embedding_dim.
        :param C: float.
        :param architecture: the architecture to use for backbone, chosen from
            resnet50, alexnet, , vgg13_bn and vgg19_bn.

        :return: Backbone+Basis Transformation model.
    """

    return ResNet_BT2(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        embedding_dim_old=embedding_dim_old,
        last_nonlin=last_nonlin,
        norm_feature=norm_feature,
        to_add_dim=to_add_dim,
        C=C,
        architecture=architecture,
    )

def ResNet101(num_classes: int,
              embedding_dim: int,
              last_nonlin: bool = True,
              **kwargs) -> nn.Module:
    """Get a ResNet101 model.

    :param num_classes: Number of classes in the dataset.
    :param embedding_dim: Size of the output embedding dimension.
    :param last_nonlin: Whether to apply non-linearity before output.
    :return: ResNet18 Model.
    """
    return ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        num_classes=num_classes,
        eembedding_dim=embedding_dim,
        last_nonlin=last_nonlin
    )


def WideResNet50_2(num_classes: int,
                   embedding_dim: int,
                   last_nonlin: bool = True,
                   **kwargs) -> nn.Module:
    """Get a WideResNet50 model.

    :param num_classes: Number of classes in the dataset.
    :param embedding_dim: Size of the output embedding dimension.
    :param last_nonlin: Whether to apply non-linearity before output.
    :return: ResNet18 Model.
    """
    return ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        base_width=64 * 2,
        embedding_dim=embedding_dim,
        last_nonlin=last_nonlin
    )


def WideResNet101_2(num_classes: int,
                    embedding_dim: int,
                    last_nonlin: bool = True,
                    **kwargs) -> nn.Module:
    """Get a WideResNet101 model.

    :param num_classes: Number of classes in the dataset.
    :param embedding_dim: Size of the output embedding dimension.
    :param last_nonlin: Whether to apply non-linearity before output.
    :return: ResNet18 Model.
    """
    return ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        num_classes=num_classes,
        base_width=64 * 2,
        embedding_dim=embedding_dim,
        last_nonlin=last_nonlin
    )

class ResNet_OCA(nn.Module):


    def __init__(self,
                 block: nn.Module,
                 layers: List[int],
                 num_classes: int = 1000,
                 embedding_dim: Optional[int] = None,
                 embedding_dim_old: Optional[int] = None,
                 last_nonlin: bool = True,
                 norm_feature: bool = False,
                 to_add_dim: int = 32,
                 architecture: str = 'resnet') -> None:

        super(ResNet_OCA, self).__init__()
        if embedding_dim_old == None:
            embedding_dim_old = embedding_dim
        self.embedding_dim_old = embedding_dim_old
        self.increment = to_add_dim
        self.norm_feature = norm_feature


        self.resnet = ResNet(
                block=block,
                layers=layers,
                embedding_dim=self.embedding_dim_old+self.increment,
                last_nonlin=last_nonlin,
                norm_feature=False)

        self.embedding_dim = embedding_dim
        
        self.fc_out = nn.Linear(self.embedding_dim_old + self.increment, num_classes, bias=False)
        self.fc_orth = orthogonal(nn.Linear(self.embedding_dim_old + self.increment, self.embedding_dim_old + self.increment))
    
    def forward(self, x):
        _, feature = self.resnet(x)

        new_feature = feature.reshape(feature.size(0), -1)

        if self.norm_feature:
            new_feature = F.normalize(new_feature)

        output_feat = self.fc_orth(new_feature)

        bct_feature = new_feature[:, :self.embedding_dim_old]

        output = self.fc_out(output_feat)

        return output, bct_feature, output_feat, new_feature
    

def ResNet18_OCA(num_classes: int,
                   embedding_dim: int,
                   embedding_dim_old: int = None,
                   last_nonlin: bool = True,
                   norm_feature: bool = True,
                   to_add_dim=32,
                   architecture: str = 'resnet18',
                   **kwargs) -> nn.Module:

    return ResNet_OCA(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        embedding_dim_old=embedding_dim_old,
        last_nonlin=last_nonlin,
        norm_feature=norm_feature,
        to_add_dim=to_add_dim,
        architecture=architecture
    )

def ResNet50_OCA(num_classes: int,
                   embedding_dim: int,
                   embedding_dim_old: int = None,
                   last_nonlin: bool = True,
                   norm_feature: bool = True,
                   to_add_dim=32,
                   architecture: str = 'resnet50',
                   **kwargs) -> nn.Module:

    return ResNet_OCA(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        embedding_dim_old=embedding_dim_old,
        last_nonlin=last_nonlin,
        norm_feature=norm_feature,
        to_add_dim=to_add_dim,
        architecture=architecture,
    )