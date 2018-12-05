from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths

from torch.autograd import Variable
from nets.masknet import NetMask
from nets.resnet_v1 import resnetv1

import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse
import sys

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a netG network')
  parser.add_argument('--saved_model', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)
  parser.add_argument('--saved_netG_model', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = parse_args()
  resnet = resnetv1(num_layers=101)
  netG = NetMask()
  if args.saved_model:
    print('load saved resnet model')
    resnet.load_state_dict(torch.load(args.saved_model))
  
  if args.saved_netG_model:
    print('load saved netG model')
    netG.load_state_dict(torch.load(args.saved_netG_model))
  
  
  print(res.size())