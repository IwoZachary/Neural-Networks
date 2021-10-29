import numpy as np

NUMBER_OF_OUTPUTS = 10


class MLP:
    def __init__(self, inputs, targets, hidden_layers):

        self.inputs = np.array(inputs)  # stworzenie macierzy wejściowej płaskiej
        self.num_inputs = len(self.inputs[0])  # ilość danych wejściowych
        self.layers = [self.num_inputs] + hidden_layers + [NUMBER_OF_OUTPUTS]  # warstwy z ilością neuronów w warstwie
        self.num_layers = len(self.layers)  # liczba warstw

        weights = []
        for i in range(self.num_layers - 1):
            weights.append(np.random.rand(self.layers[i + 1], self.layers[i] + 1))
        self.weights = weights  # wygenerowane początkowe wagi z uwzględnieniem bias

        activation = [] #zapisanie starych funkcji aktywacji, inicjacja
        for i in range(len(self.layers)):
            if i < len(self.layers)-1:
                activation.append(np.zeros(self.layers[i]+1))
            else:
                activation.append(np.zeros(self.layers[i]))
        self.activation = activation
        print(self.activation)

    def forward_propagate(self, input_x):
        activation = input_x + [1]  # warstwa wejściowa plus bias
        self.activation[0] = activation
        for i, weight in enumerate(self.weights):
            net_weight = []
            for w in weight:
                net_weight.append(np.dot(activation, w))  # wyliczanie iloczynu wag i wektora aktywacji
            net_weight = list(map(self.sigma_activation, net_weight))  # przetworzenie danych przez funkcję aktywacji
            if i < self.num_layers - 2:  # nie dodawanie bias do warstwy wyjściowej
                net_weight.append(1)
            activation = net_weight  # przekazanie danych do kolejnej warstwy
            self.activation[i+1] = activation # zapisanie wartości aktywacji do propagacji wstecznej
            #print(activation)
        print(self.activation)
        return self.softmax(activation)

    def backword_propagate(self):
        print(1)

    @staticmethod
    def sigma_activation(x):
        y = 1.0 / (1 + np.exp(-x))
        return y

    @staticmethod
    def softmax(f_activation):
        sum_e = 0
        for x in f_activation:
            sum_e += np.exp(x)
        f_activation = list(map(lambda el: el/sum_e, f_activation))
        max_val = max(f_activation)
        max_index = f_activation.index(max_val)
        return max_index
