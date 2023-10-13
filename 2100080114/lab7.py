import torch as tc
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

S = [
    list("++++"),
    list("+   "),
    list("++++"),
    list("   +"),
    list("++++")
]

C = [
    list("++++"),
    list("+   "),
    list("+   "),
    list("+   "),
    list("++++")
]
S= np.array(S)
S = tc.tensor(np.where(S=="+", 1,-1), dtype=tc.float32)


C= np.array(C)
C = tc.tensor(np.where(C=="+", 1,-1), dtype=tc.float32)
target1 = tc.tensor(1.0)
target2 = tc.tensor(0.0)

class Hebb_Net(nn.Module):
  def _init_(self, arr_size=(3, 3)):
    super()._init_()
    self.len = np.prod(arr_size)
    self.weights = tc.zeros(self.len)
    self.bias = tc.zeros(1)

  def forward(self, data, target):
    self.weights.data += data*target
    self.bias.data += target

    return (data*self.weights)

model = Hebb_Net(arr_size=S.shape)

model(S.flatten(), 1)
model(C.flatten(), -1)

model.weights, model.bias
