import numpy as np
from matplotlib import pyplot as plt
# 生成 输入数据x 及 目标数据y
np.random.seed(100)
x = np.linspace(-1,1,100).reshape(100,1)
y = 3*np.power(x,2)+2+0.2*np.random.rand(x.size).reshape(100,1)
# 查看x、y数据分布情况
plt.scatter(x,y,label='data')
plt.legend()
plt.show()
# 初始化权重参数
w1 = np.random.rand(1,1)
b1 = np.random.rand(1,1)
# 训练模型
lr = 0.001 # 学习率
for i in range(800): #梯度下降
    y_pred = np.power(x,2)*w1+b1

    loss = 0.5*(y_pred - y)**2
    loss = loss.sum()  # 方差

    grad_w = np.sum((y_pred - y)*np.power(x,2))
    grad_b = np.sum((y_pred - y))
    w1 -= lr*grad_w  # 将学习率看作步长
    b1 -= lr*grad_b
# 可视化结果
plt.plot(x,y_pred,'r-',label='predict')
plt.scatter(x,y,color='blue',marker='o',label='true') # true data
plt.xlim(-1,1)
plt.ylim(2,6)
plt.legend()
plt.show()
print(w1,b1)
