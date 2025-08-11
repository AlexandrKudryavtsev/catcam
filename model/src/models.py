from torchvision.models import efficientnet_b0, mobilenet_v3_small
import torch

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
  
def setup_model(model_name, unfreeze_layers=5):
  if (model_name == "efficientnet_b0"):
    model = efficientnet_b0(pretrained=True)

    n_in = model.classifier[-1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(n_in, 256, bias=True),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, 1, bias=True),
    )

    for param in model.parameters():
      param.requires_grad = False

  elif (model_name == "mobilenet_v3_small"):
    model = mobilenet_v3_small(pretrained=True)

    n_in = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(n_in, 1, bias=True)

  total_layers = len(model.features)
  for i in range(total_layers - unfreeze_layers, total_layers):
    for param in model.features[i].parameters():
        param.requires_grad = True

  for param in model.classifier.parameters():
    param.requires_grad = True

  return model