import random as rnd


class Perceptron:

    def __init__(self, learning_rate, dataset, classes):
        self.__learning_rate = learning_rate
        self.__weights = []
        self.__epochs = 0
        for _ in range(0, 3):
            self.__weights.append(rnd.uniform(0, 0.5).__round__(2))
        self.__dataset = dataset
        self.__classes = classes
        for d in self.__dataset:
            d.append(1)
        #print(dataset)
        #print(self.__weights)

    def fit(self):
        error_found = 10
        while error_found > 0:
            self.__epochs += 1
            error_found = 0
            for index, x in enumerate(self.__dataset):
                y = self.predict(x)
                error = self.__classes[index] - y
                if error != 0:
                    error_found += 1
                temp_weight = []
                for weight, xk in zip(self.__weights, x):
                    temp_weight.append(weight + self.__learning_rate * error * xk)
                self.__weights = temp_weight
            print(error_found)
            print(self.__weights)

    def predict(self, x):
        z = 0
        for xi, wi in zip(x, self.__weights):
            z += xi * wi
        return self.__bipolar(z)

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

