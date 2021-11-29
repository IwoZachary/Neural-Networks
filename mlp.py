import numpy as np

NUMBER_OF_OUTPUTS = 10
EPSILON = 0.00000001


class MLPv2:
    def __init__(self, num_inputs, hidden_layers, learnig_rate, epoches, weight_loc, weight_scale):
        '''

        :param num_inputs: liczba parametrów wejściowych we wzorcu
        :param hidden_layers: lista zawierająca liczbę neuronów w warstwie ukrytej
        :param weight_loc: wartość reprezentująca środek rozkładu wag
        :param weight_scale: odchylenie standardowe wagi
        '''
        self.learning_rate = learnig_rate
        self.num_inputs = num_inputs  # ilość danych wejściowych
        self.layers = [self.num_inputs] + hidden_layers + [NUMBER_OF_OUTPUTS]  # warstwy z ilością neuronów w warstwie
        self.num_layers = len(self.layers)  # liczba warstw
        self.activation = []  # wartość funkcji aktywacji
        self.boost = []  # wartość funkcji pobudzenia
        self.derivatives = []  # pochodne wag
        self.epoches = epoches
       # weights = []
       # bias = []
        previousDerivatives = []
        sumGradient = []
        sumWeightDelta = []
        previousWeightDelta = []
        mt = []
        vt = []
        for i in range(self.num_layers - 1):
            previousDerivatives.append(np.zeros(shape=(self.layers[i], self.layers[i + 1])))
            sumGradient.append(np.zeros(shape=(self.layers[i], self.layers[i + 1])))
            sumWeightDelta.append(np.zeros(shape=(self.layers[i], self.layers[i + 1])))
            previousWeightDelta.append(np.zeros(shape=(self.layers[i], self.layers[i + 1])))
            mt.append(np.zeros(shape=(self.layers[i], self.layers[i + 1])))
            vt.append(np.zeros(shape=(self.layers[i], self.layers[i + 1])))
        (self.weights, self.bias) = self.xavier_weight_dist(weight_loc, weight_scale)
        self.previousDerivatives = previousDerivatives
        self.sumGradient = sumGradient
        self.sumWeightDelta = sumWeightDelta
        self.previousWeightDelta = previousWeightDelta
        self.mt = mt
        self.vt = vt

    def normal_weight_dist(self, weight_loc, weight_scale):
        weights = []
        bias = []
        for i in range(self.num_layers - 1):
            weights.append(
                np.random.normal(loc=weight_loc, scale=weight_scale, size=(self.layers[i], self.layers[i + 1])))
            bias.append(
                np.random.normal(loc=weight_loc, scale=weight_scale, size=(self.layers[i + 1])))
        return(weights, bias)

    def xavier_weight_dist(self, weight_loc, weight_scale):
        weights = []
        bias = []
        xavier_val = []
        for i in range(len(self.layers)-1):
         xavier_val.append(2/(self.layers[i]+self.layers[i+1]))

        for i in range(self.num_layers - 1):
            temp_w =  np.random.normal(loc=weight_loc, scale=weight_scale, size=(self.layers[i], self.layers[i + 1]))
            temp_w = temp_w * xavier_val[i]
            temp_b = np.random.normal(loc=weight_loc, scale=weight_scale, size=(self.layers[i + 1]))
            temp_b = temp_b * xavier_val[i]
            weights.append(temp_w)
            bias.append(temp_b)
        return (weights, bias)

    def he_weight_dist(self, weight_loc, weight_scale):
        weights = []
        bias = []
        he_val = []
        for i in range(len(self.layers) - 1):
            he_val.append(2 / (self.layers[i]))

        for i in range(self.num_layers - 1):
            temp_w = np.random.normal(loc=weight_loc, scale=weight_scale, size=(self.layers[i], self.layers[i + 1]))
            temp_w = temp_w * he_val[i]
            temp_b = np.random.normal(loc=weight_loc, scale=weight_scale, size=(self.layers[i + 1]))
            temp_b = temp_b * he_val[i]
            weights.append(temp_w)
            bias.append(temp_b)
        return (weights, bias)



    def forward_propagate(self, input_x):
        act = input_x  # wartość pobudzenia w warstwie pierwszej
        self.activation.append(act)
        for i, weight in enumerate(self.weights):
            booster = np.dot(act, weight)
            act = self.sigma_activation(booster)
            self.activation.append(act)
        return self.softmax(act)

    def backward_propagate(self, error):
        for i in reversed(range(len(self.weights))):
            activation_upper = self.activation[i + 1]
            derivative = error * self.sigma_derivative(activation_upper)
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
                    #self.gradient_descent(ammount)
                    #self.gradient_descent_with_momentum(0.7, ammount)
                    #self.gradient_descent_with_nestov(0.7, ammount)
                    #self.adaGrad(ammount)
                    self.adaDelta(0.9, ammount)
                    #self.adam(0.9, 0.999, ammount)
                    sum_errors += error.mean() ** 2
                if start:
                    startError = sum_errors / ammount
                    start = False
                #print("Error: {} at epoch {}".format(sum_errors / ammount, i + 1))
        endError = sum_errors / ammount
        avverageStep = (startError - endError) / self.epoches
        print(" {} {} {} ".format(startError, endError, avverageStep), end="")


    def gradient_descent(self, ammount):
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate / ammount * self.derivatives[i]
            self.bias[i] += self.learning_rate / ammount * self.derivatives[i].T.sum(axis=1)

    def gradient_descent_with_momentum(self, gamma, ammount):
        for i in range(len(self.weights)):
            delta_w = gamma * self.previousDerivatives[i] + self.learning_rate/ammount * self.derivatives[i]
            self.weights[i] += delta_w
            self.bias[i] += delta_w.T.sum(axis=1)
            self.previousDerivatives[i] = delta_w

    def gradient_descent_with_nestov(self, gamma, ammount):
        for i in range(len(self.weights)):
            delta_w = gamma * self.previousDerivatives[i] + self.learning_rate/ammount * (self.weights[i] - gamma*self.previousDerivatives[i])
            self.weights[i] += delta_w
            self.bias[i] += delta_w.T.sum(axis=1)
            self.previousDerivatives[i] = delta_w

    def adaGrad(self, ammount):
        for i in range(len(self.weights)):
            self.sumGradient[i] += self.derivatives[i] * self.derivatives[i]
            delta_w = (self.learning_rate/ammount)/np.sqrt((self.sumGradient[i]+EPSILON))*self.derivatives[i]
            self.weights[i] += delta_w
            self.bias[i] += delta_w.T.sum(axis=1)

    def adaDelta(self, par, ammount):
        for i in range(len(self.weights)):
            self.sumGradient[i] = par*self.sumGradient[i] + (1-par)*(self.derivatives[i] * self.derivatives[i])
            self.sumWeightDelta[i] = par*self.sumWeightDelta[i]+(1-par)*(self.previousWeightDelta[i]*self.previousWeightDelta[i])
            delta_w = np.sqrt((self.sumWeightDelta[i]+EPSILON))/np.sqrt((self.sumGradient[i]+EPSILON))*self.derivatives[i]
            self.weights[i] += delta_w
            self.bias[i] += delta_w.T.sum(axis=1)
            self.previousWeightDelta[i] = delta_w

    def adam(self, b1, b2, ammount):
        for i in range(len(self.weights)):
            self.mt[i] = b1 * self.mt[i] + (1 - b1)*self.derivatives[i]
            self.vt[i] = b2 * self.vt[i] + (1 - b2)*self.derivatives[i]*self.derivatives[i]
            mt_prim = self.mt[i]/(1 - b1)
            vt_prim = self.vt[i]/(1 - b2)
            delta_w = (self.learning_rate/ammount)/(np.sqrt(vt_prim)+EPSILON)*mt_prim
            self.weights[i] += delta_w
            self.bias[i] += delta_w.T.sum(axis=1)


    def validate(self, input_data, target):
        correct = 0
        for _input, _target in zip(input_data, target):
            res = self.forward_propagate(_input)
            _value = max(res)
            _value = res.index(_value)
            if _value == _target:
                correct += 1
        return correct / len(input_data)

    @staticmethod
    def sigma_activation(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigma_derivative(x):
        return np.exp(-x) / (1 + np.exp(-x)) ** 2

    @staticmethod
    def relu(x):
        return np.where(x >= 0, x, 0)

    @staticmethod
    def softplus(x):
        return np.log(1 + np.exp(x))
    @staticmethod
    def softplus_derivative(x):
        return 1 / (1 + np.exp(-x))

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