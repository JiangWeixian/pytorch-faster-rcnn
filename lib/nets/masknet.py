import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.autograd import Variable

class NetMask(nn.Module):
  def __init__(self, input_nc = 3, output_nc = 1):
    super(NetMask, self).__init__()
    self.input_nc = input_nc
    self.output_nc = output_nc

    self.deconvmodel = nn.Sequential(
      # *2
      nn.Conv2d(1024, 512, 1, 1),
      nn.BatchNorm2d(512),
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(512, 256, 4, 2, 1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(256, 128, 4, 2, 1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(128, 64, 4, 2, 1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 1, 7, 1),
      nn.Tanh(),
    )

  def _up_sample(self, fm, w, h):
    return F.upsample(fm, size=(int(w), int(h)), mode='bilinear')

  
  def forward(self, input):
    x = self.deconvmodel(input)
    return x


default_layers = [[('CONV', 64, 4, 2, 1), 'LR', ('CONV', 64*2, 4, 2, 1), 'B','LR'], [('CONV', 64*4, 4, 2, 1), 'B','LR'], [('CONV', 64*8, 4, 2, 1), 'B','LR'], [('CONV', 1, 4, 1, 0), 'S']]
default_dim = 1
def _create_layers(layer = default_layers, x_dim = default_dim):
  '''create gans network grah

  @Return:
  a array, stored the network grah
  '''
  layers = []
  i_dim = x_dim
  for v in layer:
    if v == 'R':
      layers += [nn.ReLU(True)]
    elif v == 'LR':
      layers += [nn.LeakyReLU(0.2, inplace=True)]
    elif v == 'S':
      layers += [nn.Sigmoid()]
    elif v == 'TH':
      layers += [nn.Tanh()]
    elif v == 'B':
      layers += [nn.BatchNorm2d(i_dim)]
    elif v == 'P':
      layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]
    elif type(v) == tuple:
      layer_type, o_dim, k, s, p = v
      if layer_type == 'DCONV':
        layers += [nn.ConvTranspose2d(i_dim, o_dim, kernel_size=k, stride=s, padding=p, bias=False)]
        i_dim = o_dim
      elif layer_type == 'CONV':
        layers += [nn.Conv2d(i_dim, o_dim, kernel_size=k, stride=s, padding=p, bias=False)]
        i_dim = o_dim
    elif type(v) == list:
      layer, o_dim = _create_layers(v, i_dim)
      i_dim = o_dim
      layers.append(layer)
  return layers, i_dim

class NetD(nn.Module):
  '''The netD class
  '''
  def __init__(self):
    super(NetD, self).__init__()
    layers = _create_layers()[0]
    print(layers)
    self.length = len(layers)
    network_parts = []
    for i in range(self.length):
      part = nn.Sequential(*layers[i])
      network_parts.append(part)
    self.main = nn.ModuleList(network_parts)

  def forward(self, input):
    x = input
    outputs = []
    for index in range(self.length):
      x = self.main[index](x)
      outputs.append(x)
    return outputs[-1], outputs[:-1]


if __name__ == '__main__':
  netG = NetMask()
  input = Variable(torch.randn(1, 512, 38, 50))
  res = netG.forward(input)
  print(res.size())
  netD = NetD()
  print(netD)