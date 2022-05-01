import random as rd
import math
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
rd.seed(7)

class Neuron:
    def __init__(self, n_inputs):
        self.weights = [rd.random() for i in range(n_inputs)]
        self.bias = rd.random()
        self.output = None
        self.delta = None

    def get_weighted_sum(self, inputs):
        summation = self.bias
        for weight, inp in zip(self.weights, inputs):
            summation += weight * inp
        return summation 

    def __repr__(self):
        return str({"weights": self.weights, "bias": self.bias, "output": self.output, "delta": self.delta})

class NeuralNetwork:
    def __init__(self, training_data, hidden_layers, n_hidden):
        self.train_inp = [row[ : -1] for row in training_data]
        self.expected = [row[-1] for row in training_data]
        self.network = []
        n_inputs = len(self.train_inp[0])
        n_outputs = len(set(self.expected))
        for i in range(hidden_layers):
            tot_inputs = n_inputs if i == 0 else n_hidden
            self.network.append([Neuron(tot_inputs) for i in range(n_hidden)])      # hidden layers
        self.network.append([Neuron(n_hidden) for i in range(n_outputs)])           # output layer

    def activate(self, weighted_sum):
        return self.sigmoid(weighted_sum)

    def feed_forward(self, inputs):
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                neuron.output = self.activate(neuron.get_weighted_sum(inputs))
                new_inputs.append(neuron.output)
            inputs = new_inputs
        return inputs               # which now contains output of the output layer

    def prop_back(self, exp_netw_op):
        for neuron, expec_val in zip(self.network[-1], exp_netw_op):      # calc err for output layer
            neuron.delta = (neuron.output - expec_val) * self.sig_derv(neuron.output)
        for i in reversed(range(len(self.network) - 1)):           # calc err for hidden layers
            layer = self.network[i]
            for j, neuron in enumerate(layer):
                error = 0.0
                for nxt_lyr_neuron in self.network[i + 1]:
                    error += nxt_lyr_neuron.weights[j] * nxt_lyr_neuron.delta
                neuron.delta = error * self.sig_derv(neuron.output)

    def update_weights(self, inputs, lr_rate):
        for layer in self.network:
            for neuron in layer:
                for i, inp in enumerate(inputs):
                     neuron.weights[i] -= lr_rate * neuron.delta * inp
                neuron.bias -= lr_rate * neuron.delta
            inputs = [neuron.output for neuron in layer]

    def train(self, n_epochs = 5, lr_rate = 0.1):
        num=0
        arr=[]
        for epoch in range(n_epochs):
            sum_err = 0.0
            for r, row in enumerate(self.train_inp):
                outputs = self.feed_forward(row)
                exp_ntw_op = [0 for i in range(len(outputs))]
                # if there are 3 labels (0, 1, 2) and expected o/p is for eg. 2 then exp_ntw_op should be [0,0,1]
                # i.e. the 3rd output neuron should be fired
                exp_ntw_op[self.expected[r]] = 1
                sum_err += sum([(expc_op - op) ** 2 for expc_op, op in zip(exp_ntw_op, outputs)])
                self.prop_back(exp_ntw_op)
                self.update_weights(row, lr_rate)
            arr.append(round(sum_err,4))
            if(epoch>=1):
                if(arr[epoch]==arr[epoch-1]):
                    num+=1
                else:
                    num=0
            print(">epoch = %d, lrate = %.3f, error = %.4f" % (epoch, lr_rate, sum_err))
            if(num==15):
                break

    def predict(self, inputs):
        outputs = self.feed_forward(inputs)
        return outputs.index(max(outputs))

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sig_derv(self, sig_op):              # arg is already an output of a sigmoid func
        return sig_op * (1 - sig_op)

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=73)
training_data = np.c_[X_train, y_train].tolist()

for data in training_data:
    data[-1] = int(data[-1])

neural_net = NeuralNetwork(training_data, hidden_layers = 2, n_hidden = 4)
neural_net.train(n_epochs = 200, lr_rate = 0.1)

print()
for layer in neural_net.network:
    for neuron in layer:
        print(neuron)
    print()
correct_predictions = 0
for test_data, actual_class in zip(X_test, y_test):
    pred_class = neural_net.predict(test_data)
    if pred_class == actual_class:
        correct_predictions += 1

print("Accuracy: {}%".format(correct_predictions * 100 / len(y_test)))