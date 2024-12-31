import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from layers import CurvatureLayerRW, STN, SCConv_Block, InterChannelCrissCrossAttention, CurvatureLayerCA
import torchvision
from layers import NormLinear

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, loss='softmax', cat_flag = False):
        super(ResNet, self).__init__()
        self.loss = loss
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.fc = nn.Linear(512 * block.expansion, num_classes, bias=False)
        # if self.loss == 'cosface' or self.loss == 'fusion':
        # from layers import NormLinear
        # self.fc = NormLinear(512 * block.expansion,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.mp = nn.MaxPool2d(kernel_size=3,stride=2, padding=1)
        self.cal = CurvatureLayerRW(1)
        self.cat_flag = cat_flag

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        if self.cat_flag:
            y, x = normalize_data(x)
            curv_att = self.cal(y)
            curv_att = self.mp(curv_att)
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
            x = normalize_01(x)

        # See note [TorchScript super()]
        x = self.conv1(x) # (150,50)        
        if self.cat_flag:
            x = x + (curv_att * x)
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x) # (75,25) 
        if self.cat_flag:
            curv_att = self.mp(curv_att)
            y = curv_att * x
            x = x + (curv_att * x)        
        
        x = self.layer2(x) # (38,13)   
        
        x = self.layer3(x) # (19, 7) 
        
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # y = self.fc(x)
        return x, curv_att

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = load_state_dict_from_url(model_urls[arch],
                                                   progress=progress)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k != "fc.weight" and k != "fc.bias"}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def normalize_data(tensor):
    """
    Normalize each individual tensor slice (x[0, 0, :, :]) to the range [0, 1].
    Assumes input tensor has shape (batch_size, channels, height, width).
    """
    tensor = tensor.float()  # Ensure the tensor is in float32 format
    min_vals = tensor.amin(dim=(2, 3), keepdim=True)
    max_vals = tensor.amax(dim=(2, 3), keepdim=True)
    normalized_tensor = (tensor - min_vals) / (max_vals - min_vals + 1e-8)  # Add epsilon to avoid division by zero
    return normalized_tensor, tensor

def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def main():
    import torch.nn as nn
    from torchvision import transforms
    from datasets.dataloader import VisionDataset
    from torch.utils.data.dataloader import DataLoader

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])

    cars_trainset = VisionDataset('../datasets/cars196_trainlist.txt', '/home/weifeng/Desktop/datasets/fine-grained-datasets/Cars196/', transform_train)
    cars_trainloader = DataLoader(cars_trainset, 16, False)

    model = resnet50(pretrained=True)
    model = nn.DataParallel(model).cuda()

    for i, (data, target) in enumerate(cars_trainloader):
        data, target = data.cuda(), target.cuda()
        x, y = model(data)
        print(x.shape)


class SimSiam(nn.Module):
    def __init__(self, dim=128*4, pred_dim=128):
        super(SimSiam, self).__init__()
        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer
    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = x1
        z2 = x2
        
        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()


class SimSiamWrapper(nn.Module):
    def __init__(self, encoder_network):
        super(SimSiamWrapper, self).__init__()
        self.encoder_network = encoder_network
        self.simple_siamese = SimSiam()

    def forward(self, x1, x2):
        emb1, y1 = self.encoder_network(x1)
        emb2, y2 = self.encoder_network(x2)

        # emb1 = self.encoder_network(x1)
        # emb2 = self.encoder_network(x2)

        ### SimSiamese:
        p1, p2, z1, z2 = self.simple_siamese(emb1, emb2)

        return p1, p2, z1, z2 ,y1, y2

"""
class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone, self.embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.projector = Projector(args, self.embedding)

    def forward(self, x, y):
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        repr_loss = torch.nn.functional.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(torch.nn.functional.relu(1 - std_x)) / 2 + torch.mean(torch.nn.functional.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss
"""
def normalize_01(tensor):
    means = torch.zeros((tensor.shape[0], *(1,)*len(tensor.shape[1:]))).to('cuda')
    stds = torch.zeros((tensor.shape[0], *(1,)*len(tensor.shape[1:]))).to('cuda')
    for i in range(tensor.shape[0]):
        means[i] = 0.3 #MMCBNU -> 0.4
        stds[i] = 0.13 # MMCBNU -> 0.138
    tensor = (tensor - means) / (stds)
    # tensor_min = tensor.min()
    # tensor_max = tensor.max()
    return tensor#(tensor - tensor_min) / (tensor_max - tensor_min)

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class EnhancedDenseNet121(nn.Module):
    def __init__(self, pretrained=True, num_classes=1000, cat_flag = True):
        super(EnhancedDenseNet121, self).__init__()

        self.cat_flag = cat_flag

        # Load pre-trained DenseNet121 without the classifier
        self.densenet121 = torchvision.models.densenet121(pretrained=pretrained)
        self.features = self.densenet121.features
        
        # Replace the classifier for the desired number of classes
        self.classifier = nn.Linear(self.densenet121.classifier.in_features, num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.cat_flag:
            self.mp = nn.MaxPool2d(kernel_size=3,stride=2, padding=1)
            self.cal = CurvatureLayerRW(1)

    def forward(self, x):

        if self.cat_flag:
            curv_att = self.cal(x)
            curv_att = self.mp(curv_att)


        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
            # x = normalize_01(x)

        
        # First convolution layer
        x = self.features.conv0(x)
        if self.cat_flag:
            x = x + curv_att * x
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)
        
        x = self.features.denseblock1(x)
        if self.cat_flag:
            curv_att = self.mp(curv_att)
            x = x + curv_att * x
        
        # Remaining DenseNet layers
        x = self.features.transition1(x)
        x = self.features.denseblock2(x)
        x = self.features.transition2(x)
        x = self.features.denseblock3(x)
        x = self.features.transition3(x)
        x = self.features.denseblock4(x)

        # Classification layer
        x = self.features.norm5(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        return torch.abs(out)




class MobilenetV3Large(nn.Module):
    def __init__(self, pretrained=True, num_classes=1000, cat_flag=True):
        super(MobilenetV3Large, self).__init__()
        
        self.cat_flag = cat_flag
        
        # Load pre-trained MobileNetV3 Large
        self.mobilenetv3 = torchvision.models.mobilenet_v3_large(pretrained=pretrained)

        if self.cat_flag:
            self.mp = nn.MaxPool2d(kernel_size=3,stride=2, padding=1)
            self.cal = CurvatureLayerRW(1)

        # Replace the classifier for the desired number of classes
        self.mobilenetv3.classifier[3] = NormLinear(self.mobilenetv3.classifier[3].in_features, num_classes)

    def forward(self, x):
        if self.cat_flag:
            curv_att = self.cal(x)
            curv_att = self.mp(curv_att)
        
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
            x = normalize_01(x)

        x = self.mobilenetv3.features[0][0](x)
        # Apply the first curvature layer after the first convolutional layer
        if self.cat_flag:
            x = x + curv_att*x  # Adding as a residual

        x = self.mobilenetv3.features[0][1](x)  # Normalization layer
        x = self.mobilenetv3.features[0][2](x)  # Activation layer

        
        x = self.mobilenetv3.features[1].block[0][0](x)
        # Apply the second curvature layer after the first layer of the second block
        if self.cat_flag:
            # curv_att = self.mp(curv_att)
            x = x + curv_att*x  # Adding as a residual

        # Processing the rest of the layers in the block
        for i in range(len(self.mobilenetv3.features[1].block[0])):
            if i >0:
                x = self.mobilenetv3.features[1].block[0][i](x)

        # Remaining layers of MobileNetV3
        for i in range(2, len(self.mobilenetv3.features)):
            x = self.mobilenetv3.features[i](x)

        # Classifier
        x = self.mobilenetv3.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.mobilenetv3.classifier(x)
        return torch.abs(x)
    

class EnhancedEfficientNetB0(nn.Module):
    def __init__(self, pretrained=True, num_classes=1000, cat_flag=True):
        super(EnhancedEfficientNetB0, self).__init__()
        
        self.cat_flag = cat_flag
        
        # Load pre-trained EfficientNet
        self.efficientnet = torchvision.models.efficientnet_b0(pretrained=pretrained)

        if self.cat_flag:
            self.mp = nn.MaxPool2d(kernel_size=3,stride=2, padding=1)
            self.cal = CurvatureLayerRW(1)

        # Replace the classifier for the desired number of classes
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        if self.cat_flag:
            curv_att = self.cal(x)
            curv_att = self.mp(curv_att)
        
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
            # x = normalize_01(x)

        x = self.efficientnet.features[0][0](x)  # Initial convolution
        # Apply the first curvature layer after the first convolutional layer
        if self.cat_flag:
            x = x + curv_att*x  # Adding as a residual
        x = self.efficientnet.features[0][1](x)
        x = self.efficientnet.features[0][2](x)
        
        self.efficientnet.features[1][0].block[0][0](x)
        # Apply the second curvature layer
        if self.cat_flag:
            # curv_att = self.mp(curv_att)
            x = x + curv_att*x  # Adding as a residual
        x = self.efficientnet.features[1][0].block[0][1](x)
        x = self.efficientnet.features[1][0].block[0][2](x)
        x = self.efficientnet.features[1][0].block[1](x)
        x = self.efficientnet.features[1][0].block[2](x)
        
        # Remaining layers of EfficientNet
        for i in range(2, len(self.efficientnet.features)):
            x = self.efficientnet.features[i](x)

        # Classifier
        x = self.efficientnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.efficientnet.classifier(x)
        return torch.abs(x)


















class ModifiedGoogleNet(nn.Module):
    def __init__(self, pretrained=True, num_classes=1000, cat_flag=True):
        super(ModifiedGoogleNet, self).__init__()
        
        self.cat_flag = cat_flag
        
        # Load pre-trained GoogleNet
        self.google_net = torchvision.models.googlenet(pretrained=pretrained, num_classes=1000, aux_logits=False)

        if self.cat_flag:
            self.mp = nn.MaxPool2d(kernel_size=3,stride=2, padding=1)
            self.cal = CurvatureLayerRW(1)

        # Num classes:
        self.google_net.fc = nn.Linear(self.google_net.fc.in_features, num_classes)

    def forward(self, x):
        if self.cat_flag:
            curv_att = self.cal(x)
            curv_att = self.mp(curv_att)
        
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
            x = normalize_01(x)

        x = self.google_net.conv1(x) # Initial convolution
        # Apply the first curvature layer after the first convolutional layer
        if self.cat_flag:
            x = x + curv_att*x  # Adding as a residual
        x = self.google_net.conv1.bn(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.google_net.maxpool1(x)

        x = self.google_net.conv2(x)

        # Apply the second curvature layer
        if self.cat_flag:
            curv_att = self.mp(curv_att)
            x = x + curv_att*x  # Adding as a residual
        x = self.google_net.conv2.bn(x)
        x = nn.functional.relu(x, inplace=True)

        # Dynamically apply the remaining layers using a counter until before the fc
        num_layers_to_skip = 2  # Skip conv1, maxpool1, conv2:
        idx = 0
        for name, layer in self.google_net.named_children():
            if name == 'fc':
                break
            if idx > num_layers_to_skip:
                x = layer(x)
            idx += 1

        # Flatten the output for the fully connected layers
        x = torch.flatten(x, start_dim=1)

        # Fully connected layer:
        x = self.google_net.fc(x)

        return torch.abs(x)


class ModifiedMNASNet1_3(nn.Module):
    def __init__(self, pretrained=True, num_classes=1000, cat_flag=True):
        super(ModifiedMNASNet1_3, self).__init__()
        
        self.cat_flag = cat_flag

        # Load pre-trained MNASNet
        self.mnasnet = torchvision.models.mnasnet1_3(pretrained=pretrained)

        if self.cat_flag:
            self.cal = CurvatureLayerRW(1)
            self.mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Modify the classifier for the desired number of classes
        self.mnasnet.classifier[1] = nn.Linear(1280, num_classes)

    def create_custom_layers(self):
        if self.cat_flag:
            # Placeholder for curvature attention
            self.catt = None
            # Wrap the first and fourth layers with CALWrapper
            self.mnasnet.layers[0] = nn.Sequential(self.mnasnet.layers[0], CALWrapper(self))
            self.mnasnet.layers[3] = nn.Sequential(self.mnasnet.layers[3], CALWrapper(self))

    def forward(self, x):
        if self.cat_flag:
            self.catt = self.mp(self.cal(x))
            self.create_custom_layers()

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            x = normalize_01(x)

        x = self.mnasnet(x)
        return x

class ModifiedMNASNet1_3(nn.Module):
    def __init__(self, pretrained=True, num_classes=1000, cat_flag=True):
        super(ModifiedMNASNet1_3, self).__init__()
        self.cat_flag = cat_flag

        # Load pre-trained MNASNet
        self.mnasnet = torchvision.models.mnasnet1_3(pretrained=pretrained)

        if self.cat_flag:
            self.cal = CurvatureLayerRW(1)
            self.mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.modify_first_layers()

        # Modify the classifier for the desired number of classes
        self.mnasnet.classifier[1] = nn.Linear(1280, num_classes)

    def modify_first_layers(self):
        # Clone and modify the first four layers
        first_layers = list(self.mnasnet.layers[:4])
        first_layers[0] = self.modify_layer(first_layers[0], mp_flag= True)
        # first_layers[3] = self.modify_layer(first_layers[3], mp_flag = False)
        
        # Replace the first four layers in the original network
        for i in range(4):
            self.mnasnet.layers[i] = first_layers[i]

    def modify_layer(self, layer, mp_flag):
        # Create a sequential layer with the original layer and CALWrapper
        return nn.Sequential(layer, CALWrapper(self.cal, self.mp, mp_flag=mp_flag))

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        x = self.mnasnet(x)
        return x

class CALWrapper(nn.Module):
    def __init__(self, cal_layer, mp_layer, mp_flag):
        super(CALWrapper, self).__init__()
        self.cal_layer = cal_layer
        self.mp_layer = mp_layer
        self.mp_flag = mp_flag

    def forward(self, x):
        if self.mp_flag:
            curv_att = self.mp_layer(self.cal_layer(x[:,0,:,:].unsqueeze(1)))
        else:
            curv_att = self.cal_layer(x[:,0,:,:].unsqueeze(1))
        return x + curv_att * x


class ModifiedInceptionV3(nn.Module):
    def __init__(self, pretrained=True, num_classes=1000, cat_flag=True):
        super(ModifiedInceptionV3, self).__init__()

        self.cat_flag = cat_flag
        # Load pre-trained InceptionV3
        self.inception = torchvision.models.inception_v3(pretrained=pretrained, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, num_classes)

        if self.cat_flag:
            self.mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.cal = CurvatureLayerRW(1)

    def forward(self, x):
        if self.cat_flag:
            curv_att = self.cal(x)
            curv_att = self.mp(curv_att)
        
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            x = normalize_01(x)

        # First two convolutional layers with CurvatureLayerRW
        x = self.inception.Conv2d_1a_3x3(x)
        if self.cat_flag:
            x = x + curv_att.resize_as_(x[:,0,:,:]).unsqueeze(1) * x

        x = self.inception.Conv2d_2a_3x3(x)
        if self.cat_flag:
            x = x + curv_att.resize_as_(x[:,0,:,:]).unsqueeze(1) * x

        x = self.inception.Conv2d_2b_3x3(x)
        x = self.inception.maxpool1(x)
        x = self.inception.Conv2d_3b_1x1(x)
        x = self.inception.Conv2d_4a_3x3(x)
        x = self.inception.maxpool2(x)

        # Inception modules
        x = self.inception.Mixed_5b(x)
        x = self.inception.Mixed_5c(x)
        x = self.inception.Mixed_5d(x)
        x = self.inception.Mixed_6a(x)
        x = self.inception.Mixed_6b(x)
        x = self.inception.Mixed_6c(x)
        x = self.inception.Mixed_6d(x)
        x = self.inception.Mixed_6e(x)
        x = self.inception.Mixed_7a(x)
        x = self.inception.Mixed_7b(x)
        x = self.inception.Mixed_7c(x)

        # Adaptive average pooling and fully connected layer
        x = self.inception.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.inception.dropout(x)
        x = self.inception.fc(x)

        return x


class ModifiedViT(nn.Module):
    def __init__(self, pretrained=True, num_classes=1000, cat_flag=True):
        super(ModifiedViT, self).__init__()
        self.cat_flag = cat_flag

        # Load the pre-trained ViT model
        self.vit = torchvision.models.vit_b_16(pretrained=pretrained)
        self.vit.heads[0] = nn.Linear(self.vit.heads[0].in_features, num_classes)

        # Curvature Attention Layer
        if self.cat_flag:
            self.cal = CurvatureLayerRW(1)
            self.mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.resize_transform = torchvision.transforms.Resize((224, 224))
    def forward(self, x):
        x = self.pad_and_resize(x)
        if self.cat_flag:
            curv_att = self.cal(x)
            # curv_att = self.mp(curv_att)

        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
            x = normalize_01(x)
        
        if self.cat_flag:
            # Apply Curvature Attention after conv_proj
            x = self.vit(x+x*curv_att)
        else:
            x = self.vit(x)
        return x
    
    def pad_and_resize(self, x):
        # Assuming x is a batch of images of shape (batch_size, channels, height, width)
        _, _, height, width = x.shape
        max_side = max(height, width)
        padding = [0, 0, 0, 0]  # left, top, right, bottom

        if width < max_side:
            total_pad = max_side - width
            padding[0] = total_pad // 2  # left padding
            padding[2] = total_pad - padding[0]  # right padding

        elif height < max_side:
            total_pad = max_side - height
            padding[1] = total_pad // 2  # top padding
            padding[3] = total_pad - padding[1]  # bottom padding

        # Apply padding
        pad_transform = torchvision.transforms.Pad(padding, fill=0, padding_mode='constant')
        x_padded = torch.stack([pad_transform(image) for image in x])

        # Resize the padded image
        x_resized = self.resize_transform(x_padded)
        return x_resized

class EncoderWithFC(nn.Module):
    def __init__(self, base_encoder, output_dim=512):
        super(EncoderWithFC, self).__init__()
        # Use the base encoder, e.g., EfficientNet
        self.encoder = base_encoder
        # Add a fully connected layer to reduce to `output_dim`
        self.fc = nn.Sequential(
            nn.Linear(self._get_feature_dim(), output_dim),
            nn.ReLU(inplace=True)
        )

    def _get_feature_dim(self):
        # Pass a dummy input through the encoder to get the feature dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224
            features = self.encoder(dummy_input)
            return features.view(features.size(0), -1).size(1)

    def forward(self, x):
        # Pass through the encoder
        features = self.encoder(x)
        # Flatten the features
        features = features.view(features.size(0), -1)
        # Pass through the FC layer
        return self.fc(features)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        se = x.mean((2, 3))  # Global average pooling
        se = self.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se)).unsqueeze(2).unsqueeze(3)
        return x * se


if __name__ == '__main__':
    main()
