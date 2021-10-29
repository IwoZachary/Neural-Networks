#from keras.datasets import mnist
#import numpy as np

from mlp import MLP

#(train_X, train_y), (test_X, test_y) = mnist.load_data()
mlp =MLP([[1, 2, 3, 4]], [], [3,2,4])


print(mlp.forward_propagate([1, 2, 3, 4]))