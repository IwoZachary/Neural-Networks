import random as rnd
import numpy as np
class Adeline:
    def __init__(self, learning_rate, dataset, classes):
        self.__learning_rate = learning_rate
        self.__epochs = 0
        weights = []
        for _ in range(0, 3):
           weights.append(rnd.uniform(0, 0.5).__round__(2))
        self.__weights = np.array(weights)
        self.__dataset = dataset
        self.__classes = classes
        for d in self.__dataset:
            d.append(1)

    def fit(self):
        sum_error_sqr = 10
        while sum_error_sqr > 0.0:
            self.__epochs += 1
            sum_error_sqr = 0
            for index, x in enumerate(self.__dataset):
                xv = np.array(x)
                z = np.dot(xv, self.__weights)
                #z = self.sum(x)
                y = self.__bipolar(z)
                dk = self.__classes[index]
                error_sqr = (dk - y) ** 2
                sum_error_sqr += error_sqr
                temp_weight = []
                for i, _ in enumerate(self.__weights):
                    self.__weights[i] += (dk - z) * x[i] * self.__learning_rate
                    #for weight, xk in zip(self.__weights, x):
                    #    new_weight = weight + 2 * self.__learning_rate * (dk - weight * xk) * xk
                    #    temp_weight.append(new_weight)
                    #self.__weights = temp_weight
            print(self.__weights)

            sum_error_sqr /= len(self.__dataset)
            print(sum_error_sqr)





    def sum(self, x):
        z = 0
        for xi, wi in zip(x, self.__weights):
            z += xi * wi
        return z

    def predict(self, x):
        return self.__bipolar(self.sum(x))

    def __unipolar(self, z):
        if z > 0:
            return 1
        else:
            return 0

    def __bipolar(self, z):
        if z > 0:
            return 1
        else:
            return -1
