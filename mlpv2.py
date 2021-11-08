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
        bias_derivative = []
        for i in range(self.num_layers - 1):
            weights.append(np.random.rand(self.layers[i + 1], self.layers[i]))
            derivatives.append(np.zeros((self.layers[i + 1], self.layers[i])))
            bias.append(np.random.rand(self.layers[i + 1], 1))
            bias_derivative.append(np.zeros((self.layers[i + 1], 1)))
        self.weights = weights  # wygenerowane początkowe wagi z uwzględnieniem bias
        self.derivatives = derivatives  # wygenerowanie tablic na pochodne potrzebne do backpropagation
        self.bias = bias
        self.bias_derivative = bias_derivative
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
                net_weight.append(float(np.dot(activation, w)+b))
                  # wyliczanie iloczynu wag i wektora aktywacji
            net_weight = list(map(self.sigma_activation, net_weight))  # przetworzenie danych przez funkcję aktywacji
            activation = net_weight  # przekazanie danych do kolejnej warstwy
            self.activation[i + 1] = np.array(activation)  # zapisanie wartości aktywacji do propagacji wstecznej
        return self.softmax(activation)

    def backword_propagate(self, error):
        for i in reversed(range(len(self.weights))):
            act = self.activation[i + 1]
            error_re = np.array([error])
            #print(error_re)
            #error_re = error_re.reshape(error_re.shape[0], -1)
            delta = error * np.array(list(map(self.sigma_derivativ, act)))
            d2 = list(delta)
            #d2.append(1)
            d2 = np.array(d2)
            d2 = d2.reshape(d2.shape[0], -1)
            self.bias_derivative[i] = np.array(d2)
            #print("biasder od {} {}".format(i, sum(self.bias_derivative[i].shape)))
            delta_re = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activation[i]
            current_activations = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.matmul(current_activations, delta_re)
            #self.bias_derivative[i] = np.dot(self.bias[i], error_re)
            #print(self.bias_derivative[i])
            error = np.dot(delta, self.weights[i])


    def train(self, input, target):
        ammount = len(input)
        for i in range(0, 1000):
            sum_errors = 0
            for _inp, _targ in zip(input, target):
                output = self.forward_propagate(_inp)
                error =  (output - _targ)**2
                self.backword_propagate(error)
                self.gradient_descent(ammount)
                if error != 0:
                    sum_errors += error
                #print('weight od  0: {}'.format(self.weights[0]))
            print("Error: {} at epoch {}".format(sum_errors / len(input), i + 1))

    def gradient_descent(self,ammount, learningRate=0.1):
        
        for i in range(len(self.weights)):
            
            self.weights[i] -= learningRate/ammount*self.derivatives[i].T
            self.bias[i] -= learningRate/ammount*self.bias_derivative[i]
            #print(weights[i])
            #print(bias[i])
        #print("suma 2 {}".format(sum(self.weights[1])[0]))
        #print("wagi od 0  {}".format( sum(self.weights[0])))
        #print("wagi od 1 {}".format(i, self.weights[2]))



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
