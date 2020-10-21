import torch
# 构建一个5*3的未初始化的矩阵：
x = torch.Tensor(5, 3)
print(x)

# 构建一个随机初始化矩阵：
x = torch.rand(5, 3)
print(x)
# 获取矩阵的size：
print(x.size())
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print(result)
y.add_(x)  # 任何原地改变tensor的运算后边会后缀一个“_”,例如：x.copy_(y),x.t_(),会改变x的值。
print(y)
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)

# y=(x+w)∗(w+1),我们需要对w求导，画一个计算图，a = x+w ,b = w+1，y=a∗b，实际结果w`=2*w+x+1
w = torch.tensor([1.],requires_grad=True)
x = torch.tensor([2.],requires_grad=True)

a = torch.add(w,x)
b = torch.add(w,1)
y = torch.mul(a,b)

y.backward()
print(w.grad)  # 求y关于x的偏导
