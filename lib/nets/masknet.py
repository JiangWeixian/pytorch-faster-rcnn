import torch.nn as nn
import torch

from torch.autograd import Variable

class NetMask2(nn.Module):
  def __init__(self, input_nc = 3, output_nc = 1):
    super(NetMask2, self).__init__()
    self.input_nc = input_nc
    self.output_nc = output_nc

    self.deconvmodel = nn.Sequential(
      # *2
      nn.ConvTranspose2d(512, 256, 4, 2, 1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(256, 128, 3, 2, 1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(128, 64, 4, 2, 1, 1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 1, 7, 1),
      nn.Tanh()
    )

  
  def forward(self, input):
    x = self.deconvmodel(input)
    return x

if __name__ == '__main__':
  netG = NetMask2()
  input = Variable(torch.randn(1, 512, 41, 41))
  res = netG.forward(input)
  print(res.size())