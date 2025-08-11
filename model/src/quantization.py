import torch
import copy
import torch.nn as nn
from torchvision.ops.misc import SqueezeExcitation

class QuantizedModel(torch.nn.Module):
  def __init__(self, model_fp32):
    super(QuantizedModel, self).__init__()
    self.quant = torch.quantization.QuantStub()
    self.dequant = torch.quantization.DeQuantStub()
    self.model_fp32 = model_fp32

  def forward(self, X):
    X = self.quant(X)
    X = self.model_fp32(X)
    X = self.dequant(X)
    return X

class QuantizableMBConv(nn.Module):
    def __init__(self, orig_block):
        super().__init__()
        self.block = orig_block.block
        self.use_res_connect = orig_block.use_res_connect
        self.skip_add = nn.quantized.FloatFunctional()
        self.stochastic_depth = orig_block.stochastic_depth

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result = self.skip_add.add(result, x)
            result = self.stochastic_depth(result)
        return result

class QuantizableSqueezeExcitation(nn.Module):
    def __init__(self, orig_se):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = orig_se.fc1
        self.fc2 = orig_se.fc2
        self.skip_mul = nn.quantized.FloatFunctional()

    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = nn.ReLU()(scale)
        scale = self.fc2(scale)
        scale = torch.sigmoid(scale)
        return self.skip_mul.mul(x, scale)

def make_quantizable(model):
    for name, module in model.named_children():
        if isinstance(module, nn.SiLU):
            setattr(model, name, nn.ReLU(inplace=True))
        else:
            make_quantizable(module)

    for name, module in model.named_children():
        if isinstance(module, SqueezeExcitation):
            setattr(model, name, QuantizableSqueezeExcitation(module))
        elif hasattr(module, 'block') and hasattr(module, 'use_res_connect'):
            setattr(model, name, QuantizableMBConv(module))
        else:
            make_quantizable(module)
    return model

def fuse_model(init_model, model_name):
    model = copy.deepcopy(init_model)
    model = make_quantizable(model)
    model.eval()

    def fuse_conv_bn_relu(module):
        if isinstance(module, nn.Sequential) and len(module) >= 2:
            if isinstance(module[0], nn.Conv2d) and isinstance(module[1], nn.BatchNorm2d):
                if len(module) >= 3 and isinstance(module[2], nn.ReLU):
                    torch.quantization.fuse_modules(module, [["0", "1", "2"]], inplace=True)
                else:
                    torch.quantization.fuse_modules(module, [["0", "1"]], inplace=True)

    if model_name == 'efficientnet_b0':
        fuse_conv_bn_relu(model.features[0])

        for block in model.features:
            if hasattr(block, "block") and hasattr(block, "use_res_connect"):
                for component in block.block:
                    if isinstance(component, nn.Sequential):
                        fuse_conv_bn_relu(component)

    elif model_name == 'mobilenet_v3_small':
        for feature in model.features:
            if isinstance(feature, nn.Sequential):
                fuse_conv_bn_relu(feature)

                for submodule in feature:
                    if hasattr(submodule, 'fc1') and isinstance(submodule.fc1, nn.Sequential):
                        if len(submodule.fc1) >= 2 and isinstance(submodule.fc1[0], nn.Linear) and isinstance(submodule.fc1[1], nn.BatchNorm1d):
                            torch.quantization.fuse_modules(submodule.fc1, [["0", "1"]], inplace=True)

    return model