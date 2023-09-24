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
用交替方向乘子法进行求解 
"""
c = 0.005
alpha = 1/c
p = 0.01       # 正则化参数
epsilon = 1e-5  # 最大允许误差
max_iteration_times = 1e6

x_k = np.zeros(x_dim)
y_k = np.zeros(x_dim)
v_k = np.zeros(x_dim)
x_k_pre = np.zeros(x_dim)

iteration_result = []  # 每步计算结果
k = 1  # 迭代次数

while k < max_iteration_times:
    '''Step1: master节点发送 x ^ k 给其他节点
                在此不进行模拟'''

    '''Step2:所有worker节点计算各自值，并传送给master节点
                这里在模拟时计算后直接加入temp_numerator和temp_denominator中'''
    temp_numerator = np.zeros(x_dim)
    temp_denominator = np.zeros((x_dim, x_dim))
    for i in range(node_num):
        temp_numerator += A[i].T @ b[i]
        temp_denominator += A[i].T @ A[i]

    '''Step3:master节点根据公式，计算出新的 x y v 
                模拟时以下工作都由master节点实现'''
    # 更新x：令导数为0推导公式，从而进行求解
    x_k = np.linalg.inv(temp_denominator + c * np.eye(x_dim, x_dim)) @ (temp_numerator + c * y_k - v_k)
    # 更新y：通过软门限求解
    for i in range(x_dim):
        if x_k[i] + v_k[i] / c < -p / c:
            y_k[i] = x_k[i] + v_k[i] / c + p / c
        elif x_k[i] + v_k[i] / c > p / c:
            y_k[i] = x_k[i] + v_k[i] / c - p / c
        else:
            y_k[i] = 0
    # 更新v：直接代入公式求解
    v_k = v_k + c * (x_k - y_k)

    iteration_result.append(x_k.copy())  # 记录每步计算结果

    '''Step4:重复（1）~（3）直到算法收敛或到达最大迭代次数'''
    if np.linalg.norm(x_k - x_k_pre) < epsilon:
        break
    else:
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
plt.title('Alternating Direction Multiplier Method')
plt.xlabel('Iteration times')
plt.ylabel('Distance (ln)')
plt.plot(dist_real, 'r', label='Distance from the true value')
plt.plot(dist_opt, 'g', label='Distance from the optimal solution')
plt.grid()
plt.legend()
plt.show()

print(f"最优解的稀疏度：{np.count_nonzero(x_opt)}")

