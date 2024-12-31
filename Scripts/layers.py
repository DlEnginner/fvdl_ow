import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torchvision.models as models

class NormLinear(nn.Module):
    def __init__(self, input, output):
        super(NormLinear, self).__init__()
        self.input = input
        self.output = output
        self.weight = nn.Parameter(torch.Tensor(output, input))
        self.reset_parameters()

    def forward(self, input):
        weight_normalized = F.normalize(self.weight, p=2, dim=1)
        input_normalized = F.normalize(input, p=2, dim=1)
        output = input_normalized.matmul(weight_normalized.t())
        return output

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

# class CurvatureLayerRW(nn.Module):
#     def __init__(self, channels=1, device='cuda'):
#         super(CurvatureLayerRW, self).__init__()
#         self.channels = channels
#         self.device = device

#         self.register_buffer('sigma_a', torch.tensor(2.0))
#         self.register_buffer('sigma_b', torch.tensor(2.5))
#         self.register_buffer('sigma_c', torch.tensor(3.0))

#         self.kernel_size = self.calculate_max_kernel_size([self.sigma_a, self.sigma_b, self.sigma_c])
#         self.initialize_conv_layers()

#         self.activation = nn.ReLU()

#     def calculate_max_kernel_size(self, sigmas):
#         return int(2 * torch.ceil(4 * max(sigmas)).item() + 1)

#     def initialize_conv_layers(self):
#         padding_size = self.kernel_size // 2
#         self.conv_gx = nn.Conv2d(self.channels, self.channels, kernel_size=self.kernel_size, padding=padding_size, bias=False).to(self.device)
#         self.conv_gxx = nn.Conv2d(self.channels, self.channels, kernel_size=self.kernel_size, padding=padding_size, bias=False).to(self.device)
#         self.conv_gy = nn.Conv2d(self.channels, self.channels, kernel_size=self.kernel_size, padding=padding_size, bias=False).to(self.device)
#         self.conv_gyy = nn.Conv2d(self.channels, self.channels, kernel_size=self.kernel_size, padding=padding_size, bias=False).to(self.device)
#         self.conv_gxy = nn.Conv2d(self.channels, self.channels, kernel_size=self.kernel_size, padding=padding_size, bias=False).to(self.device)

#         self.initialize_with_gaussian(self.sigma_a)
#         self.initialize_with_gaussian(self.sigma_b)
#         self.initialize_with_gaussian(self.sigma_c)

#     def initialize_with_gaussian(self, sigma):
#         winsize = torch.ceil(4 * sigma).item()
#         kernel_size = int(2 * winsize + 1)
#         padding = (self.kernel_size - kernel_size) // 2

#         x_grid, y_grid = torch.meshgrid(torch.arange(-winsize, winsize + 1, dtype=torch.float32), 
#                                         torch.arange(-winsize, winsize + 1, dtype=torch.float32))

#         g = (1 / (2 * math.pi * sigma**2)) * torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
#         gx = (-x_grid / sigma**2) * g
#         gxx = ((x_grid**2 - sigma**2) / sigma**4) * g
#         gy = (-y_grid / sigma**2) * g
#         gyy = ((y_grid**2 - sigma**2) / sigma**4) * g
#         gxy = ((x_grid * y_grid) / sigma**4) * g

#         kernel_gx = gx.view(1, 1, *gx.size()).repeat(1, self.channels, 1, 1)
#         kernel_gx = F.pad(kernel_gx, (padding, padding, padding, padding), "constant", 0)

#         kernel_gxx = gxx.view(1, 1, *gxx.size()).repeat(1, self.channels, 1, 1)
#         kernel_gxx = F.pad(kernel_gxx, (padding, padding, padding, padding), "constant", 0)

#         kernel_gy = gy.view(1, 1, *gy.size()).repeat(1, self.channels, 1, 1)
#         kernel_gy = F.pad(kernel_gy, (padding, padding, padding, padding), "constant", 0)

#         kernel_gyy = gyy.view(1, 1, *gyy.size()).repeat(1, self.channels, 1, 1)
#         kernel_gyy = F.pad(kernel_gyy, (padding, padding, padding, padding), "constant", 0)

#         kernel_gxy = gxy.view(1, 1, *gxy.size()).repeat(1, self.channels, 1, 1)
#         kernel_gxy = F.pad(kernel_gxy, (padding, padding, padding, padding), "constant", 0)

#         self.conv_gx.weight = nn.Parameter(kernel_gx, requires_grad=False)
#         self.conv_gxx.weight = nn.Parameter(kernel_gxx, requires_grad=False)
#         self.conv_gy.weight = nn.Parameter(kernel_gy, requires_grad=False)
#         self.conv_gyy.weight = nn.Parameter(kernel_gyy, requires_grad=False)
#         self.conv_gxy.weight = nn.Parameter(kernel_gxy, requires_grad=False)

#     def forward(self, input, device='cuda'):
#         fx = self.conv_gx(input)
#         fxx = self.conv_gxx(input)
#         fy = self.conv_gy(input)
#         fyy = self.conv_gyy(input)
#         fxy = self.conv_gxy(input)

#         f1 = 0.5 * torch.sqrt(torch.tensor(2.0).to(device)) * (fx + fy)
#         f2 = 0.5 * torch.sqrt(torch.tensor(2.0).to(device)) * (fx - fy)
#         f11 = 0.5 * fxx + fxy + 0.5 * fyy
#         f22 = 0.5 * fxx - fxy + 0.5 * fyy

#         h = fxx / ((1 + fx**2)**1.5)
#         v = fyy / ((1 + fy**2)**1.5)
#         d1 = f11 / ((1 + f1**2)**1.5)
#         d2 = f22 / ((1 + f2**2)**1.5)

#         curvature = h + v + d1 + d2
#         curvature = self.activation(curvature)
#         curvature = torch.sigmoid(curvature)
#         return curvature

class CurvatureLayerRW(nn.Module):
    def __init__(self, channels=1, device='cuda'):
        super(CurvatureLayerRW, self).__init__()
        self.channels = channels
        self.device = device

        # Make sigma trainable
        self.sigma_a = nn.Parameter(torch.tensor(2.0, device=device), requires_grad=True)

        self.kernel_size = self.calculate_max_kernel_size()
        self.padding_size = self.kernel_size // 2  # Adjust padding size
        self.initialize_conv_layers()

        self.activation = nn.ReLU()

    def calculate_max_kernel_size(self):
        return int(2 * torch.ceil(4 * self.sigma_a).item() + 1)

    def initialize_conv_layers(self):
        self.conv_gx = nn.Conv2d(self.channels, self.channels, kernel_size=self.kernel_size, padding=0, bias=False).to(self.device)
        self.conv_gxx = nn.Conv2d(self.channels, self.channels, kernel_size=self.kernel_size, padding=0, bias=False).to(self.device)
        self.conv_gy = nn.Conv2d(self.channels, self.channels, kernel_size=self.kernel_size, padding=0, bias=False).to(self.device)
        self.conv_gyy = nn.Conv2d(self.channels, self.channels, kernel_size=self.kernel_size, padding=0, bias=False).to(self.device)
        self.conv_gxy = nn.Conv2d(self.channels, self.channels, kernel_size=self.kernel_size, padding=0, bias=False).to(self.device)

        self.initialize_with_gaussian()

    def initialize_with_gaussian(self):
        self.kernels = []
    
        winsize = torch.ceil(4 * self.sigma_a).item()
        x_grid, y_grid = torch.meshgrid(torch.arange(-winsize, winsize + 1, dtype=torch.float32, device=self.device), 
                                        torch.arange(-winsize, winsize + 1, dtype=torch.float32, device=self.device))
        g = (1 / (2 * math.pi * self.sigma_a**2)) * torch.exp(-(x_grid**2 + y_grid**2) / (2 * self.sigma_a**2))
        gx = (-x_grid / self.sigma_a**2) * g
        gxx = ((x_grid**2 - self.sigma_a**2) / self.sigma_a**4) * g
        gy = (-y_grid / self.sigma_a**2) * g
        gyy = ((y_grid**2 - self.sigma_a**2) / self.sigma_a**4) * g
        gxy = ((x_grid * y_grid) / self.sigma_a**4) * g

        kernel_gx = gx.view(1, 1, *gx.size()).repeat(1, self.channels, 1, 1)
        kernel_gxx = gxx.view(1, 1, *gxx.size()).repeat(1, self.channels, 1, 1)
        kernel_gy = gy.view(1, 1, *gy.size()).repeat(1, self.channels, 1, 1)
        kernel_gyy = gyy.view(1, 1, *gyy.size()).repeat(1, self.channels, 1, 1)
        kernel_gxy = gxy.view(1, 1, *gxy.size()).repeat(1, self.channels, 1, 1)

        self.kernels.append((kernel_gx, kernel_gxx, kernel_gy, kernel_gyy, kernel_gxy))

        self.conv_gx.weight = nn.Parameter(self.kernels[0][0], requires_grad=False)
        self.conv_gxx.weight = nn.Parameter(self.kernels[0][1], requires_grad=False)
        self.conv_gy.weight = nn.Parameter(self.kernels[0][2], requires_grad=False)
        self.conv_gyy.weight = nn.Parameter(self.kernels[0][3], requires_grad=False)
        self.conv_gxy.weight = nn.Parameter(self.kernels[0][4], requires_grad=False)

    def forward(self, input, device='cuda'):
        self.initialize_with_gaussian()

        # Apply reflection padding to input to handle border effects
        padded_input = F.pad(input, (self.padding_size, self.padding_size, self.padding_size, self.padding_size), mode='reflect')

        fx = self.conv_gx(padded_input)
        fxx = self.conv_gxx(padded_input)
        fy = self.conv_gy(padded_input)
        fyy = self.conv_gyy(padded_input)
        fxy = self.conv_gxy(padded_input)

        # Crop the padded output to match the input size
        fx = fx[:, :, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]
        fxx = fxx[:, :, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]
        fy = fy[:, :, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]
        fyy = fyy[:, :, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]
        fxy = fxy[:, :, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]

        f1 = 0.5 * torch.sqrt(torch.tensor(2.0, device=self.device)) * (fx + fy)
        f2 = 0.5 * torch.sqrt(torch.tensor(2.0, device=self.device)) * (fx - fy)
        f11 = 0.5 * fxx + fxy + 0.5 * fyy
        f22 = 0.5 * fxx - fxy + 0.5 * fyy

        h = fxx / ((1 + fx**2)**1.5)
        v = fyy / ((1 + fy**2)**1.5)
        d1 = f11 / ((1 + f1**2)**1.5)
        d2 = f22 / ((1 + f2**2)**1.5)

        curvature = h + v + d1 + d2
        curvature = self.activation(curvature)
        # curvature = torch.sigmoid(curvature)
        curvature  = min_max_scaler(curvature)
        curvature = F.interpolate(curvature, size=input.size()[-2:], mode='bilinear', align_corners=False)
        # mask = (input != 0).float()
        # curvature = curvature * mask
        return curvature

# def min_max_scaler(tensor):
#     min_val = tensor.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
#     max_val = tensor.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
#     normalized_tensor = (tensor - min_val) / (max_val - min_val)
#     return normalized_tensor

def min_max_scaler(tensor):
    min_val = tensor.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    max_val = tensor.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]

    # Create a mask for regions where max_val == min_val
    scale_range = max_val - min_val
    scale_range[scale_range == 0] = 1  # Prevent division by zero by assigning eps only for constant regions
    
    # Normalize only where max_val != min_val
    normalized_tensor = (tensor - min_val) / scale_range

    return normalized_tensor
    
def find_and_sort_peaks(curvature_output, window_size=3, curvature_threshold=0.5):
    pad = window_size // 2
    local_max = F.max_pool2d(curvature_output, kernel_size=window_size, stride=1, padding=pad)
    peaks = (curvature_output == local_max) & (curvature_output > curvature_threshold)

    # Get indices of peaks
    peak_indices = peaks.nonzero(as_tuple=False)

    # Sort peaks based on curvature values
    intensities = curvature_output.view(-1)[peaks.view(-1)]
    sorted_indices = intensities.argsort(descending=True)
    sorted_peak_indices = peak_indices[sorted_indices]

    return sorted_peak_indices

class STN(nn.Module):

    def __init__(self, in_channels, size=10*12*32):

        super(STN, self).__init__()
        self.size = size

        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.SELU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.SELU(True)
        )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.size, 32),
            nn.SELU(True),
            nn.Linear(32, 3),#2 * 2),
            # nn.Tanh(),
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([5, 5, 0], dtype=torch.float))
        self.tanh = nn.Tanh()
        
    def stn(self, x, y):

        # Spatial transformer network forward function
        xs = self.localization(x)
        xs = xs.view(-1, self.size)
        preds = self.fc_loc(xs)
        # preds = self.tanh(preds)
        cx = preds[:,0]#2*torch.sigmoid(preds[:,0])
        cy = preds[:,1]#2*torch.sigmoid(preds[:,1])
        theta = (2*torch.sigmoid(preds[:,2]) - 1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        aff_mat = torch.zeros((xs.size()[0], 2, 2))
        aff_mat[:,0,0] = cx * cos_theta
        aff_mat[:,0,1] = -sin_theta
        aff_mat[:,1,0] = sin_theta
        aff_mat[:,1,1] = cy * cos_theta
        displacements = torch.tensor([0,0]).repeat(theta.size()[0],).view(-1,2,1).cuda()
        theta = torch.cat([aff_mat.cuda(), displacements],-1)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        y = F.grid_sample(y, grid)
        return x, y

    def forward(self, x, y):
        x, y = self.stn(x, y)
        return x, y

class SCConv_Block(nn.Module):
    def __init__(self, inplanes, planes, stride, dilation, groups, pooling_r, hw):
        super(SCConv_Block, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    Conv2dSamePadding(inplanes, inplanes, kernel_size=3, stride=stride,
                                      dilation=dilation, groups=groups, bias=True),
                    nn.BatchNorm2d(inplanes)
                    )
        self.k3 = nn.Sequential(
                    Conv2dSamePadding(inplanes, inplanes, kernel_size=3, stride=stride,
                                      dilation=dilation, groups=groups, bias=True),
                    nn.BatchNorm2d(inplanes)
                    )
        self.k4 = nn.Sequential(
                    Conv2dSamePadding(inplanes, planes, kernel_size=3, stride=stride,
                                      dilation=dilation, groups=groups, bias=True),
                    nn.BatchNorm2d(planes)
                    )

    def forward(self, x):
        identity = x
        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4
        # Swish:
        out = out*torch.sigmoid(out)
        return out

class Conv2dSamePadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super(Conv2dSamePadding, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        padding_height = conv2d_same_padding(x.size(2), self.kernel_size, self.stride)
        padding_width = conv2d_same_padding(x.size(3), self.kernel_size, self.stride)

        x = F.pad(x, (padding_width, padding_width, padding_height, padding_height))
        x = self.conv(x)

        return x


def conv2d_same_padding(input_dim, kernel_size, stride):
    numerator = input_dim * (stride - 1) + kernel_size - stride
    padding = max(0, (numerator + 1) // 2)
    return padding

### ConvGatedBlock:
class ConvGatedBlock(nn.Module):
    """  
    In the given layer, the 1x1 convolution serves as a channel-wise feature recalibration or refinement step. 
    After the 5x5 convolution extracts spatial information, the 1x1 convolution acts to blend or reweight these 
    spatial features across channels, allowing for a combination of feature maps. This can enhance the representation 
    power of the feature maps and make them more discriminative for the subsequent tasks.
    
    The layer can be categorized as a gating mechanism. After processing through the 5x5 and 1x1 convolutions, 
    the output undergoes a Swish-like operation using the sigmoid of the 1x1 convolution's output. This process 
    allows certain features to be scaled (or "gated") based on the output of the sigmoid function, giving different 
    importances to different features. This gating mechanism allows the network to dynamically recalibrate the feature 
    maps, effectively attending to more important features while downplaying others.
    
    Gating mechanisms and attention mechanisms share some conceptual similarities as they both weigh input features. 
    However, there are differences:

    Nature of Weighting: In gating mechanisms like the one shown, the weights (or gates) are often generated from the same input or immediate preceding layer, as in this case. In attention mechanisms, the weights are typically calculated based on interactions between a query, key, and value, often from different parts of the data or network.

    Purpose: Gating mechanisms typically aim to control or modulate the flow of information in the network, allowing certain features to pass while blocking others based on the data itself. Attention mechanisms, on the other hand, are more about focusing on certain parts of the input based on their relevance or importance to a particular context or query.
    """
    def __init__(self, five_by_five_input_channels, five_by_five_output_channels, one_by_one_output_channels):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.five_by_five = Conv2dSamePadding(five_by_five_input_channels, five_by_five_output_channels , kernel_size=5,stride=1,bias=True)
        # self.fbf_bn = nn.BatchNorm2d(five_by_five_output_channels)
        self.one_by_one = Conv2dSamePadding(five_by_five_output_channels, one_by_one_output_channels , kernel_size=1,stride=1,bias=True)
        # self.obo_bn = nn.BatchNorm2d(one_by_one_output_channels)

    def forward(self, feature_maps):
        fbf = self.five_by_five(feature_maps)
        # fbf = self.fbf_bn(fbf)
        obo = self.one_by_one(fbf)
        # obo = self.obo_bn(obo)
        # Swish activation:
        w = self.sigmoid(obo)
        out = obo * w 
        return out


class InterChannelCrissCrossAttention(nn.Module):
    """ Inter-Channel Criss-Cross Attention Module """
    def __init__(self, in_dim):
        super(InterChannelCrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = torch.nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

        # # Layer Normalization
        # self.ln_query = nn.LayerNorm([in_dim // 2, h, w])  # Adjust the size to match feature map dimensions
        # self.ln_key = nn.LayerNorm([in_dim // 2, h, w])
        # self.ln_value = nn.LayerNorm([in_dim, h, w])

        # # Batch Normalization
        # self.bn_query = nn.BatchNorm2d(in_dim // 2)
        # self.bn_key = nn.BatchNorm2d(in_dim // 2)
        # self.bn_value = nn.BatchNorm2d(in_dim)


    def forward(self, x1, x2):
        m_batchsize, _, height, width = x1.size()
        proj_query = self.query_conv(x1)
        # proj_query = F.normalize(proj_query, p=1.0, dim=-2)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)
        
        proj_key = self.key_conv(x2)
        # proj_key = F.normalize(proj_key, p=1.0, dim=-2)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        
        proj_value = self.value_conv(x2)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        
        # Compute energies and concatenate
        energy_H = torch.bmm(proj_query_H, proj_key_H).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        # concate = torch.cat([energy_H, energy_W], 3)

        # Apply attention to the value
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height+width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)

        return self.gamma * (out_H + out_W) + x2  


class CustomAttention(nn.Module):
    def __init__(self, in_channels):
        super(CustomAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.fc = nn.Linear(15*32, 512, bias=False)

    def forward(self, feature_map, attention_map):
        batch_size, C, height, width = feature_map.size()

        # Generate query, key, and value matrices
        Q = self.query_conv(feature_map).view(batch_size, -1, height * width).permute(0, 2, 1)  # [B, H*W, C']
        K = self.key_conv(feature_map).view(batch_size, -1, height * width)  # [B, C', H*W]
        V = self.value_conv(feature_map).view(batch_size, -1, height * width)  # [B, C, H*W]

        # Compute attention scores
        attention_scores = torch.bmm(Q, K)  # [B, H*W, H*W]
        attention_scores = F.softmax(attention_scores, dim=-1)

        # Apply spatial attention
        attention_map = F.interpolate(attention_map, size=(height, width), mode='bilinear', align_corners=False)
        attention_map = attention_map.view(batch_size, 1, height * width).squeeze()
        out = self.fc(attention_map)
        # attention_map = attention_map.expand_as(attention_scores)
        # attention_scores = attention_scores * attention_map

        # # Apply attention to the value
        # out = torch.bmm(V, attention_scores.transpose(1, 2))
        # out = out.view(batch_size, C, height, width)

        return out


class CoordinateAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CoordinateAttention, self).__init__()
        self.channels = channels
        self.reduction = reduction

        # Attention blocks for horizontal and vertical
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # Horizontal pooling
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # Vertical pooling
        
        # Shared transformation for reduced dimensions
        self.conv1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels // reduction)
        self.relu = nn.ReLU()

        # Separate attention outputs for height and width
        self.conv_h = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, height, width = x.size()

        # Horizontal pooling branch
        h_pool = self.pool_h(x)  # (B, C, H, 1)
        h_transform = self.relu(self.bn(self.conv1(h_pool)))
        h_attn = self.sigmoid(self.conv_h(h_transform))  # (B, C, H, 1)
        h_attn = h_attn.expand(-1, -1, height, width)  # Broadcast across width

        # Vertical pooling branch
        w_pool = self.pool_w(x)  # (B, C, 1, W)
        w_transform = self.relu(self.bn(self.conv1(w_pool)))
        w_attn = self.sigmoid(self.conv_w(w_transform))  # (B, C, 1, W)
        w_attn = w_attn.expand(-1, -1, height, width)  # Broadcast across height

        # Fusion
        out = x * h_attn * w_attn
        return out



class CurvatureLayerCA(nn.Module):
    def __init__(self, channels=64, device='cuda'):
        super(CurvatureLayerCA, self).__init__()
        self.curvature_layer = CurvatureLayerRW(channels=1, device=device)  # Single-channel curvature map
        self.refine = nn.Conv2d(1, channels, kernel_size=3, padding=1)  # Expand to 64 channels
        self.coord_attention = CoordinateAttention(channels)

    def forward(self, input):
        # Compute curvature features
        curvature = self.curvature_layer(input)

        # Expand single-channel curvature into 64 channels
        refined_curvature = self.refine(curvature)

        # Apply coordinate attention
        enhanced_curvature = self.coord_attention(refined_curvature)

        return enhanced_curvature
