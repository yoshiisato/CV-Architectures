import torch
from torch import nn

vgg_config = {
  "vgg16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
  "vgg19": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGGNet(nn.Module):
  def __init__(self, config):
    super().__init__()

    layers = []
    in_channels = 3

    for layer in config:
      if layer == 'M':
        layers.append(nn.MaxPool2d(2,2))
      else:
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=layer, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = layer

    self.net = nn.Sequential(*layers)
    self.classifier = nn.Sequential(
        nn.Linear(7*7*512, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096,1000),
    )

  def forward(self, x):
    x = self.net(x)
    x = torch.flatten(x,1)
    x = self.classifier(x)
    return x
  

def vgg16():
  return VGGNet(vgg_config['vgg16'])

def vgg19():
  return VGGNet(vgg_config['vgg19'])

def main():
  model = vgg19()
  print(model)

if __name__ == "__main__":
  main()