from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths

from torch.autograd import Variable
from nets.masknet import NetMask

import torch.nn as nn
import torch.nn.functional as F
import torch

if __name__ == '__main__':
  netG = NetMask()
  input = Variable(torch.randn(1, 512, 38, 50))
  res = netG.forward(input)
  print(res.size())