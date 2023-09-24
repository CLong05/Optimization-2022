import numpy as np
import random
import matplotlib.pyplot as plt

# # 使Matplotlib输出矢量图
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'  # 通过设置矢量图的方式来提高图片显示质量

""" 
定义相关常量 
"""
node_num = 20 # 节点总数
A_dim = (10, 300)  # A：10*300
A_all_dim = (node_num, 10, 300)
x_dim = 300  # x: 300*1
x_sparsity = 5  # 稀疏度为5
e_dim = 10  # noise
b_all_dim = (node_num, 10)

""" 
生成数据 
"""
# 生成x的真值
x_nonzero_index = random.sample(range(x_dim), x_sparsity)  # 非0元素的下标
x_nonzero_element = np.random.normal(0, 1, x_sparsity)
x_real = np.zeros(x_dim)
x_real[x_nonzero_index] = x_nonzero_element
# print(f'x_real:{x_real}')

A = np.zeros(A_all_dim)
e = np.zeros(b_all_dim)
b = np.zeros(b_all_dim)
for i in range(node_num):
    # 生成测量矩阵A
    A[i] = np.random.normal(0, 1, A_dim)
    # 生成测量噪声e
    e[i] = np.random.normal(0, 0.2, e_dim)
    # 计算带噪声的b
    b[i] = A[i] @ x_real + e[i]

""" 
用邻近点梯度法进行求解 
"""
alpha = 0.001   # 固定步长
p = 0.01        # 正则化参数
epsilon = 1e-5  # 误差精度

x_k = np.zeros(x_dim)  # x_k
x_k_pre = x_k.copy()  # x_{k-1}

iteration_result = []  # 每步计算结果
k = 1  # 迭代次数
max_iteration_times = 1e6  # 最大迭代次数，防止死循环

while k < max_iteration_times:
    '''Step1: master节点发送 x ^ k 给其他节点
            在此不进行模拟'''

    '''Step2:所有worker节点计算各自值，并传送给master节点
            这里在模拟时计算后直接加入x ^ {k + 1 / 2}中'''
    x_k_temp = x_k.copy()  # master节点数据：记录 x^{k+1/2}
    for i in range(node_num):
        x_k_temp -= alpha * A[i].T @ (A[i] @ x_k - b[i])

    '''Step3:master节点根据公式，计算出新的 x^k
            模拟时以下工作都由master节点实现'''
    # 临近点投影:使用软门限进行求值
    for i in range(x_dim):
        if x_k_temp[i] < -alpha * p:
            x_k[i] = x_k_temp[i] + alpha * p
        elif x_k_temp[i] > alpha * p:
            x_k[i] = x_k_temp[i] - alpha * p
        else:
            x_k[i] = 0

    iteration_result.append(x_k.copy())  # 记录每步计算结果

    '''Step4:重复（1）~（3）直到算法收敛或到达最大迭代次数'''
    if np.linalg.norm(x_k - x_k_pre) < epsilon:  # 如果误差小于精度，则视为收敛，退出循环
        break
    else:  # 如果误差大于精度，则视为未收敛，开始新一轮迭代
        x_k_pre = x_k.copy()  # 深拷贝
        k += 1

x_opt = x_k[:]  # 最优解

# 计算每步计算结果与真值的距离以及每步计算结果与最优解的距离
dist_real = []  # 每步结果与真值的距离
dist_opt = []  # 每步结果与最优解的距离
for x_i in iteration_result:
    # 取 ln 值：取log是为了便于观察线性收敛；+1是为了防止取log后最终结果接近负无穷导致溢出
    dist_real.append(np.log(np.absolute(np.linalg.norm(x_i - x_real) + 1)))
    dist_opt.append(np.log(np.absolute(np.linalg.norm(x_i - x_opt) + 1)))

# 作图
plt.title('Proximal Gradient')
plt.xlabel('Iteration times')
plt.ylabel('Distance (ln)')
plt.plot(dist_real, 'r', label='Distance from the true value')
plt.plot(dist_opt, 'g', label='Distance from the optimal solution')
plt.grid()
plt.legend()
plt.show()

print(f"最优解的稀疏度：{np.count_nonzero(x_opt)}")