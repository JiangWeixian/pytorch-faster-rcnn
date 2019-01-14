import torch
import torch.nn as nn
from torch.autograd import Variable

class GANLoss(nn.Module):
  '''Wrap for BCEloss(now), the netD output can not be [batch_size, 1]
  ...So, this class is for more complex dim output like [batch_size, x, x]

  @Attributes:
  - real_label/fake_label: true/false
  - real_label_var/fake_label_var: true label/fake label, variable type 
  - loss: defalut(now) is BCELOSS
  '''
  def __init__(self, target_real_label=1.0, target_fake_label=0.0,
              tensor=torch.FloatTensor, type='BCE', cuda=True):
    super(GANLoss, self).__init__()
    self.real_label = target_real_label
    self.fake_label = target_fake_label
    self.real_label_var = None
    self.fake_label_var = None
    self.cuda = cuda
    self.Tensor = tensor
    if self.cuda:
        self.loss = nn.BCELoss().cuda()
    else:
        self.loss = nn.BCELoss()
          

  def get_target_tensor(self, input, target_is_real):
    '''if target_is_real, compute true data's loss, else compute fake data's loss
    '''
    target_tensor = None
    if target_is_real:
        create_label = ((self.real_label_var is None) or
                        (self.real_label_var.numel() != input.numel()))
        if create_label:
            real_tensor = self.Tensor(input.size()).fill_(self.real_label)
            if self.cuda:
                real_tensor = real_tensor.cuda()
            self.real_label_var = Variable(real_tensor, requires_grad=False)
        target_tensor = self.real_label_var
    else:
        create_label = ((self.fake_label_var is None) or
                        (self.fake_label_var.numel() != input.numel()))
        if create_label:
            fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
            if self.cuda:
                fake_tensor = fake_tensor.cuda()
            self.fake_label_var = Variable(fake_tensor, requires_grad=False)
        target_tensor = self.fake_label_var
    return target_tensor

  def __call__(self, input, target_is_real):
    target_tensor = self.get_target_tensor(input, target_is_real)
    return self.loss(input, target_tensor)

def criterion_fmloss(real_feats, fake_feats, criterion='HingeEmbeddingLoss', cuda=False):
  '''Compute distance bwtween real_feats and fake_feats, instead of l1loss

  - Params:
  @real_feats: real img's features, **not the last output of netD, and is hidden-layers's output**
  @fake_feats: same as upone, but just from fake imgs
  @criterion: criterion type, defalyt is `HingeEmbeddingLoss`
  '''
  if criterion == 'HingeEmbeddingLoss':
      criterion = nn.HingeEmbeddingLoss()
  losses = 0
  for real_feat, fake_feat in zip(real_feats, fake_feats):
      l2 = (real_feat.mean(0) - fake_feat.mean(0)) * (real_feat.mean(0) - fake_feat.mean(0))
      loss = criterion(l2, Variable(torch.ones(l2.size())).cuda())
      losses += loss
  return losses