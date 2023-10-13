import torch as tc
import torch.nn as nn
import numpy as np
import torch


class MP_Neuron(nn.Module):
  def _init_(self, ):
    super()._init_()
    self.weights=tc.tensor([1, -1])
    self.bias = 0
    self.threshold = 1
  def forward(self, x):
    net_input = tc.sum((self.weights*x)) + self.bias
    return tc.where(net_input>=self.threshold, tc.tensor(1.0), tc.tensor(0.0))


data=[
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

mp = MP_Neuron()
for i in tc.tensor(data):
  print(i,mp(i))


class XOR_N(nn.Module):
  def _init_(self, ):
    super()._init_()
    self.lin1 = nn.Linear(2, 2)
    self.sig = nn.Sigmoid()
    self.lin2 = nn.Linear(2, 1)
  def forward(self, x):
    x = self.lin1(x)
    x = self.sig(x)
    x = self.lin2(x)
    return x


model = XOR_N()
epochs = 1000
mseloss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.03)
data2=[0,1,1,0]

X = tc.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

Y = tc.tensor([0.0,1.0,1.0,0.0]).view(-1,1)
epochs = 1000

for epoch in range(epochs):
  for i in range(4):
    optimizer.zero_grad()
    yhat = model(X[i])
    loss = mseloss(yhat, Y[i])
    loss.backward()
    optimizer.step()
    if epoch%100==0:
      print(f"loss = {loss}")


print("{:.10f}".format(float(model(tc.tensor([0.0,1.0]))[0].detach().numpy())))
# print("{:.10f}".format(1.0e-05))
# float(tc.tensor(0).numpy())
