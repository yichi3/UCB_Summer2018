import numpy as np
import matplotlib.pyplot as plt
import nn


# h = 200

# x = np.linspace(-2 * np.pi, 2 * np.pi, num=16)[:, np.newaxis]
# batch_size, junk = x.shape
# W1 = nn.Variable(1, h)
# W2 = nn.Variable(h, 1)
# # b1 = nn.Variable(batch_size, 1)
# b1 = nn.Variable(batch_size, 1)
# b2 = nn.Variable(batch_size, 1)

# graph = nn.Graph([W1, W2, b1, b2])
# input_x = nn.Input(graph, x)
# mul_1 = nn.MatrixMultiply(graph, input_x, W1)
# add_1 = nn.MatrixVectorAdd(graph, mul_1, b1)
# relu_1 = nn.ReLU(graph, add_1)
# mul_2 = nn.MatrixMultiply(graph, relu_1, W2)
# add_2 = nn.Add(graph, mul_2, b2)
# value_mul_1 = graph.get_output(mul_1)
# value_add_1 = graph.get_output(add_1)
# # value_relu_1 = graph.get_output(relu_1)
# value_mul_2 = graph.get_output(mul_2)
# value_add_2 = graph.get_output(add_2)


# negation = np.array([[-1.]])
# n_x = np.dot(x, negation)
# input_n_x = nn.Input(graph, n_x)
# n_mul_1 = nn.MatrixMultiply(graph, input_n_x, W1)
# n_add_1 = nn.Add(graph, n_mul_1, b1)
# n_relu_1 = nn.ReLU(graph, n_add_1)
# n_mul_2 = nn.MatrixMultiply(graph, n_relu_1, W2)
# n_add_2 = nn.Add(graph, n_mul_2, b2)
# value_n_mul_1 = graph.get_output(n_mul_1)
# value_n_add_1 = graph.get_output(n_add_1)
# value_n_mul_2 = graph.get_output(n_mul_2)
# value_n_add_2 = graph.get_output(n_add_2)

# even = value_add_2 + value_n_add_2
# odd = value_add_2 - value_n_add_2
# print odd
# plt.plot(x, even, 'r--', x, odd, 'b--')
# plt.show()






A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
C = np.copy(A)
D = np.copy(A)
E = np.copy(A)
l = [A, B, C, D, E]
F = np.hstack(l)
print F

# def ReLU(x):
# 	return max(x, 0)

# def H(w1, w2, x1, x2):
# 	return ReLU(w1*x1 + w2*x2)

# def Y(w1, w2, x1, x2):
# 	return w1*x1 + w2*x2


# X1 = [1]
# X2 = [-1]
# W = [3, 5, -4, -6, 1, -2]

# # for x1 in X1:
# # 	for x2 in X2:
# # 		h1 = H(W[0], W[1], x1, x2)
# # 		h2 = H(W[2], W[3], x1, x2)
# # 		y = Y(W[4], W[5], h1, h2)
# # 		print x1, x2, h1, h2, y


# a = np.array([[7.],[8.]])
# b = a + 3
# print a
# print b