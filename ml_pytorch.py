import numpy as np
import torch
from matplotlib import pyplot as plt
# 生成 输入数据x 及 目标数据y
np.random.seed(100)
x = np.linspace(-1,1,100).reshape(100,1)
y = 3*np.power(x,2)+2+0.2*np.random.rand(x.size).reshape(100,1)
x=torch.tensor(x)
y=torch.tensor(y)
# 查看x、y数据分布情况
plt.scatter(x,y)
plt.show()
# 初始化权重参数
w1 =torch.zeros(1,1,requires_grad=True)
b1 =torch.zeros(1,1,requires_grad=True)
# 训练模型
lr = 0.001 # 学习率
cost = []
for i in range(800): #梯度下降
    y_pred = w1*x**2 + b1
    loss = torch.sum((y_pred - y) ** 2)
    loss.backward()
    # 参数更新
    print(w1.grad.data.item(),b1.grad.data.item())
    w1.data = w1.data - lr*w1.grad.data  # 将学习率看作步长
    b1.data = b1.data - lr*b1.grad.data
    w1.grad.data.zero_()  #梯度清零
    b1.grad.data.zero_()

# 可视化结果
plt.plot(x,y_pred.data,'r-',label='predict')
plt.scatter(x,y,color='blue',marker='o',label='true') # true data
plt.xlim(-1,1)
plt.ylim(2,6)
plt.legend()
plt.show()
print(w1.data,b1.data)
