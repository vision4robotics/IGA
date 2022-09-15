
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from config import cfg

from utile import TFGenerater
import numpy as np

test = TFGenerater(cfg).cuda()
x = np.ones((11, 192, 19,19))
x = np.array(x)

print(t.cuda.is_available)

k = t.Tensor(x)


y = test(k.cuda())

test.train()

test.eval()
