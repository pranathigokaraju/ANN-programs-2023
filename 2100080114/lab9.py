import torch
import torch.nn as nn

class XORGate(nn.Module):
    def __init__(self):
        super(XORGate, self).__init__()
        self.linear1 = nn.Linear(2, 10)
        self.linear2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        return x
model = XORGate()
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = torch.tensor([0, 1, 1, 0])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(1000):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
y_pred = model(X)
accuracy = torch.mean((y_pred == y).float())
print('Accuracy:', accuracy)
