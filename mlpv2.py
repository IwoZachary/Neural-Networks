import numpy as np
import copy

NUMBER_OF_OUTPUTS = 10


class MLPv2:
    def __init__(self, num_inputs, hidden_layers):

        self.num_inputs = num_inputs  # ilość danych wejściowych
        self.layers = [self.num_inputs] + hidden_layers + [NUMBER_OF_OUTPUTS]  # warstwy z ilością neuronów w warstwie
        self.num_layers = len(self.layers)  # liczba warstw
        weights = []
        derivatives = []
        bias = []
        for i in range(self.num_layers - 1):
            weights.append(np.random.rand(self.layers[i + 1], self.layers[i]))
            derivatives.append(np.zeros((self.layers[i + 1], self.layers[i])))
            bias.append(np.random.rand(self.layers[i + 1], 1))
        self.weights = weights  # wygenerowane początkowe wagi z uwzględnieniem bias
        self.derivatives = derivatives  # wygenerowanie tablic na pochodne potrzebne do backpropagation
        self.bias = bias
        activation = []  # zapisanie starych funkcji aktywacji, inicjacja
        for i in range(len(self.layers)):
            activation.append(np.zeros(self.layers[i]))
        self.activation = activation

    def forward_propagate(self, input_x):
        #print(input_x)
        activation = input_x  # warstwa wejściowa plus bias
        self.activation[0] = np.array(activation)
        for i, weight in enumerate(self.weights):
            net_weight = []
            for w, b in zip(weight, self.bias[i]):
                net_weight.append(float(np.dot(activation, w) + b))  # wyliczanie iloczynu wag i wektora aktywacji
            net_weight = list(map(self.sigma_activation, net_weight))  # przetworzenie danych przez funkcję aktywacji
            activation = net_weight  # przekazanie danych do kolejnej warstwy
            self.activation[i + 1] = np.array(activation)  # zapisanie wartości aktywacji do propagacji wstecznej
        return self.softmax(activation)

    def backword_propagate(self, error):
        for i in reversed(range(len(self.weights))):
            act = self.activation[i + 1]
            delta = error * np.array(list(map(self.sigma_derivativ, act)))
            delta_re = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activation[i]
            current_activations = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.matmul(current_activations, delta_re)
            error = np.dot(delta, self.weights[i])


    def train(self, input, target):
        for i in range(0, 10):
            sum_errors = 0
            for _inp, _targ in zip(input, target):
                output = self.forward_propagate(_inp)
                error = -(_targ - output)
                self.backword_propagate(error)
                self.gradient_descent()
                if error != 0:
                    sum_errors += 1
            print("Error: {} at epoch {}".format(sum_errors / len(input), i + 1))

    def gradient_descent(self, learningRate=0.001):
        #print("suma 1 {}".format(sum(self.weights[1][0])))
        for i in range(len(self.weights)):
            self.weights[i] += learningRate*self.derivatives[i].T
            self.bias[i] += learningRate*self.bias[i]
            #print(weights[i])
            #print(bias[i])
        #print("suma 2 {}".format(sum(self.weights[1])[0]))




    @staticmethod
    def sigma_activation(x):
        y = 1.0 / (1 + np.exp(-x))
        return y

    @staticmethod
    def sigma_derivativ(x):
        y = np.exp(-x) / (1 + np.exp(-x)) ** 2
        return y

    @staticmethod
    def softmax(f_activation):
        sum_e = 0
        for x in f_activation:
            sum_e += np.exp(x)
        f_activation = list(map(lambda el: el / sum_e, f_activation))
        max_val = max(f_activation)
        max_index = f_activation.index(max_val)
        return max_index
