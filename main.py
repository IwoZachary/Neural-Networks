#from keras.datasets import mnist
#import numpy as np

from mlpv2 import MLPv2
from mnist import MNIST
mndata = MNIST('./train')
images, labels = mndata.load_training()
img = []
labl = []
for i in range(0, 20):
    img.append(images[i])
    labl.append(labels[i])
#print(images[0])
#(train_X, train_y), (test_X, test_y) = mnist.load_data()
#a = []
#for i in range (30):
#    a.append(np.ravel(train_X[i]))
#a = np.array(a)
#print(a.shape)
#print(train_y.shape)

mlp =MLPv2(784, [500, 300])

mlp.train(img, labl)