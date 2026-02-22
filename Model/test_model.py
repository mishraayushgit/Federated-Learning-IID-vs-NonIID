from cnn_model import CNNModel
import torch

model = CNNModel()
x = torch.randn(1, 1, 28, 28)
y = model(x)


print(y.shape)  
