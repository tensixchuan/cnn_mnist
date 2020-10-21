import numpy as np
print('----------numpy的定义-----------------')
x = np.array([1.0, 2.0, 3.0])
print("x=",x)          # [1. 2. 3.]
print("x类型",type(x))   # type(x)表示输出x类型<class 'numpy.ndarray'>

a=[2.0,4.0,6.0]
y=np.array(a)  #将list转换为numpy数组
print("y=",y)
print("x+y=",x+y)
A=np.array([x,y])     # 生成一个二维矩阵A
print("\nA=",A,"\nA.shape:",A.shape)  # .shape表示矩阵形状    (2, 3)   （行，列）
print("A.dtype:",A.dtype)  # 元素数据类型

print('----------numpy的运算-----------------')
B=np.array([[3,8,2],[1,5,9]])  #生成一个二维矩阵B
print("A=",A,"\nB=",B,"\nA*B=",A*B)     # 矩阵对应元素乘（Hadamard乘积）
B=np.array([1,2,3])
print("A=",A,"\nB=",B,"\nA*B=",A*B)# 广播
print("A*10=",A*10)

print('------------numpy的数据元素---------------')
print('A=',A,'\nA的每行：')
# for i in range(2):
#     print(A[i])
for i in A:
    i=i.flatten()
    print(i)            # [1. 2. 3.] 换行 [2. 4. 6.]
print('\nA的元素展开：')
for i in A:
    for j in i:
        print(j)

print('--------------numpy与数组转换-----------------')
print(x)         # [1. 2. 3.]
x=x.flatten()  # flatten()将x转换为一维数组
print(x)       # [1. 2. 3.]

print('-------------条件筛选------------------')
print("y=",y)
print("选择y中大于3的元素:",y[y>3])     # 条件筛选   [4. 6.]
print("判断y中元素是否大于3:",y>3)       # [False  True  True]