import torch

# Initialize a tensor containing numbers from 0 to 11
x = torch.arange(12)

print(f"x is {x}")

# 打印变量x的形状信息
print(f'{x.shape}')

# Print the number of elements in the tensor x
print(f'{x.numel()}')

# Reshape the variable x into a two-dimensional array of shape 3x4
X = x.reshape(3, 4)

print(f'{X}')