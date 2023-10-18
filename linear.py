import torch
from torch import nn
from torch.utils import data

def generate_date(n, w, b):
    X = torch.normal(0,1, (n, len(w))) # 生成样本
    y = torch.matmul(X, w) + b         # 计算样本真实标签
    y += torch.normal(0, 0.01, y.shape) # 加上小扰动
    return (X, y)

true_w = torch.tensor([7.3, 2.8])
true_b = 39.2
N = 100
batch_size = 20
lr = 0.1
num_epoch = 10

def linear_model(w,b):
    def model(x):
        return torch.matmul(x,w) + b
    return model

if __name__ == '__main__':
    X, y = generate_date(N, true_w, true_b)
    dateset = data.TensorDataset(X, y)
    dateloader = data.DataLoader(dateset, batch_size, shuffle=True)

    w = torch.normal(0, 1, (2,), requires_grad=True)
    b = torch.normal(0, 1, (1,), requires_grad=True)
    model = linear_model(w, b)

    optimizer = torch.optim.SGD([w, b], lr)
    loss = nn.MSELoss()

    for epoch in range(1, num_epoch + 1):
        for xx, yy in dateloader:
            optimizer.zero_grad()
            l = loss(model(xx), yy)
            l.backward()
            optimizer.step()

        print(f'epoch {epoch} loss: {loss(model(X), y)}')

    print(f'w: {w} b:{b}')

    attn = nn.MultiheadAttention(10,5)
