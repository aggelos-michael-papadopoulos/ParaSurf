import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import time
from torchviz import make_dot
from torchinfo  import summary

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # Initialize p as a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, self.p, self.eps)

    def gem(self, x, p, eps):
        # Clamp all elements in x to a minimum of eps and then raise them to the power of p
        # Apply avg_pool3d with kernel size being the spatial dimension of the feature map (entire depth, height, width)
        # Finally, take the power of 1/p to invert the earlier power of p operation
        return F.avg_pool3d(x.clamp(min=eps).pow(p), (x.size(2), x.size(3), x.size(4))).pow(1. / p)

    def __repr__(self):
        # This helps in identifying the layer characteristics when printing the model or layer
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', eps=' + str(
            self.eps) + ')'


# Define a custom Bottleneck module with optional dilation
class DilatedBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dilation=1, dropout_prob=0.25):
        super(DilatedBottleneck, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.dropout1 = nn.Dropout3d(dropout_prob)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.dropout2 = nn.Dropout3d(dropout_prob)
        self.conv3 = nn.Conv3d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        out = self.dropout2(F.relu(self.bn2(self.conv2(out))))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Define the Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, feature_size, nhead, num_layers):
        super(TransformerBlock, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead),
            num_layers=num_layers
        )

    def forward(self, x):
        orig_shape = x.shape  # Save original shape
        x = x.flatten(2)  # Flatten spatial dimensions
        x = x.permute(2, 0, 1)  # Reshape for the transformer (Seq, Batch, Features)
        x = self.transformer(x)
        x = x.permute(1, 2, 0).view(*orig_shape)  # Restore original shape
        return x

# Define a Compression Layer
class CompressionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CompressionLayer, self).__init__()
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# Define the Enhanced ResNet with hybrid architecture
class ResNet3D_Transformer(nn.Module):
    def __init__(self, in_channels, block, num_blocks, num_classes=1, dropout_prob=0.1):
        super(ResNet3D_Transformer, self).__init__()
        self.in_planes = 64

        self.initial_layers = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dilation=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.compression = CompressionLayer(512 * block.expansion, 256)
        self.transformer_block = TransformerBlock(feature_size=256, nhead=8, num_layers=1)          # change to 4
        # self.gem_pooling = GeM(p=3.0, eps=1e-6)
        self.dropout = nn.Dropout(dropout_prob)


        self.classifier = nn.Linear(256, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, dilation=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, dilation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 4, 3, 2, 1)
        x = self.initial_layers(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Compress and transform
        x = self.compression(x)
        x = self.transformer_block(x)
        # Global average pooling
        x = torch.mean(x, dim=[2, 3, 4])

        # Classify
        x = self.dropout(x)  # Apply dropout before classification
        x = self.classifier(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    num_classes = 1
    num_input_channels = 20  # Number of input channels
    model = ResNet3D_Transformer(num_input_channels, DilatedBottleneck, [3, 4, 6, 3], num_classes=num_classes).to(device)
    grid_size = 41  # Assuming the input grid size (for example, 41x41x41x19)

    start = time.time()
    num_params = count_parameters(model)
    print(f"Number of parameters in the model: {num_params}")
    print(model)

    dummy_input = torch.randn(64, grid_size, grid_size, grid_size, num_input_channels).to(device)
    dummy_input = dummy_input.float().to(device)


    output = model(dummy_input)

    print("Output shape:", output.shape)
    print(output)
    print(f'total time: {(time.time() - start)/60} mins')

