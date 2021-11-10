import numpy as np
NUMBER_OF_OUTPUTS = 10

class MLPv2:
    def __init__(self, num_inputs, hidden_layers,learnig_rate, epoches, weight_loc, weight_scale, bias_loc, bias_scale):
        '''

        :param num_inputs: liczba parametrów wejściowych we wzorcu
        :param hidden_layers: lista zawierająca liczbę neuronów w warstwie ukrytej
        :param weight_loc: wartość reprezentująca środek rozkładu wag
        :param weight_scale: odchylenie standardowe wagi
        :param bias_loc: wartość reprezentująca środek rozkładu biasów
        :param bias_scale: odchylenie standardowe biasów
        '''
        self.learning_rate = learnig_rate
        self.num_inputs = num_inputs  # ilość danych wejściowych
        self.layers = [self.num_inputs] + hidden_layers + [NUMBER_OF_OUTPUTS]  # warstwy z ilością neuronów w warstwie
        self.num_layers = len(self.layers)  # liczba warstw
        self.activation = [] #wartość funkcji aktywacji
        self.boost = [] #wartość funkcji pobudzenia
        self.derivatives = [] #pochodne wag
        self.epoches = epoches
        weights = []
        bias = []
        for i in range(self.num_layers - 1):
            weights.append(
                np.random.normal(loc=weight_loc, scale=weight_scale, size=(self.layers[i], self.layers[i+1])))
            bias.append(
                np.random.normal(loc=bias_loc, scale=bias_scale, size=( self.layers[i + 1])))
        self.weights = weights #wagi
        self.bias = bias #biasy

    def forward_propagate(self, input_x):
        act = input_x  # wartość pobudzenia w warstwie pierwszej
        self.activation.append(act)
        for i, weight in enumerate(self.weights):
            booster = np.dot(act, weight)
            act = self.tanh(booster)
            self.activation.append(act)
        return self.softmax(act)

    def backward_propagate(self, error):
        for i in reversed(range(len(self.weights))):
            activation_upper = self.activation[i+1]
            derivative = error * self.tanh_derivative(activation_upper)
            derivative_reshape = derivative.reshape(derivative.shape[0], -1).T
            activation_current = self.activation[i]
            activation_current = activation_current.reshape(activation_current.shape[0], -1)
            self.derivatives.append(np.dot(activation_current, derivative_reshape))
            error = np.dot(derivative, self.weights[i].T)
        self.derivatives.reverse()

    def train(self, input, target, batch):
        start = True
        ammount = len(input)
        input = np.array(input)
        target = np.array(target)
        b_input = np.split(input, batch)
        b_target = np.split(target, batch)
        for b_i, b_t in zip(b_input, b_target):
            for i in range(0, self.epoches):
                sum_errors = 0
                for _inp, _targ in zip(b_i, b_t):
                    self.activation.clear()
                    self.boost.clear()
                    self.derivatives.clear()
                    output = self.forward_propagate(_inp)
                    t = np.zeros(NUMBER_OF_OUTPUTS)
                    t[_targ] = 1
                    error = t - output
                    self.backward_propagate(error)
                    self.gradient_descent(ammount)
                    sum_errors += error.mean() ** 2
                if start:
                    startError = sum_errors / ammount
                    start = False
        endError = sum_errors / ammount
        avverageStep = (startError - endError)/self.epoches
        print(" {} {} {} ".format(startError, endError, avverageStep),end= "")
                #print("Error: {} at epoch {}".format(sum_errors / ammount, i + 1))

    def gradient_descent(self, ammount):
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate/ammount*self.derivatives[i]
            self.bias[i] += self.learning_rate/ammount*self.derivatives[i].T.sum(axis =1)

    def validate(self, input_data, target):
        correct = 0
        for _input, _target in zip(input_data, target):
            res = self.forward_propagate(_input)
            _value = max(res)
            _value = res.index(_value)
            if _value == _target:
                correct += 1
        return correct/len(input_data)



    @staticmethod
    def sigma_activation(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def sigma_derivative(x):
        return np.exp(-x) / (1 + np.exp(-x)) ** 2

    @staticmethod
    def relu(x):
        return np.where(x > 0, x, 0)

    @staticmethod
    def softplus(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def softmax(f_activation):
        sum_e = 0
        for x in f_activation:
            sum_e += np.exp(x)
        f_activation = list(map(lambda el: el / sum_e, f_activation))
        return f_activation

    @staticmethod
    def tanh(x):
        return 2 / (1 + np.exp(-2 * x)) - 1

    def tanh_derivative(self, x):
        return 1 - self.tanh(x) ** 2