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

  def _up_sample(self, fm):
    return F.upsample(fm, size=(375, 500), mode='bilinear')

  
  def forward(self, input):
    x = self.deconvmodel(input)
    x = self._up_sample(x)
    return x

if __name__ == '__main__':
  netG = NetMask()
  input = Variable(torch.randn(1, 512, 38, 50))
  res = netG.forward(input)
  print(res.size())