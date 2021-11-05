from keras.datasets import mnist
import numpy as np

from mlpv2 import MLPv2

(train_X, train_y), (test_X, test_y) = mnist.load_data()
a = []
for i in range (30):
    a.append(np.ravel(train_X[i]))
a = np.array(a)
print(a.shape)
print(train_y.shape)

mlp =MLPv2(784, [800, 500])

mlp.train(a, train_y)