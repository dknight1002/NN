import random as rd
import math
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
            self.network.append([Neuron(tot_inputs) for i in range(n_hidden)])
        self.network.append([Neuron(n_hidden) for i in range(n_outputs)])

    def activate(self, weighted_sum):
        return self.sigmoid(weighted_sum)

    def feed_forward(self, inputs):
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                neuron.output = self.activate(neuron.get_weighted_sum(inputs))
                new_inputs.append(neuron.output)
            inputs = new_inputs
        return inputs

    def prop_back(self, exp_netw_op):
        for neuron, expec_val in zip(self.network[-1], exp_netw_op):
            neuron.delta = (neuron.output - expec_val) * self.sig_derv(neuron.output)
        for i in reversed(range(len(self.network) - 1)):
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

    def train(self, n_epochs, lr_rate):
        num=0
        arr=[]
        for epoch in range(n_epochs):
            sum_err = 0.0
            for r, row in enumerate(self.train_inp):
                outputs = self.feed_forward(row)
                exp_ntw_op = [0 for i in range(len(outputs))]
                exp_ntw_op[self.expected[r]] = 1                   
                sum_err += sum([(expc_op - op) ** 2 for expc_op, op in zip(exp_ntw_op, outputs)])
                self.prop_back(exp_ntw_op)
                self.update_weights(row, lr_rate)
            arr.append(round(sum_err,4))
            #early stopping
            if(epoch>0):
                if(arr[epoch-1]==arr[epoch]):
                    num+=1
                else:
                    num=0
            print(">epoch = %d, lrate = %.3f, error = %.3f" % (epoch, lr_rate, sum_err))
            if(num>=10):
                break

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def sig_derv(self, sig_op):
        return sig_op * (1 - sig_op)

dataset = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
] 

neural_net = NeuralNetwork(dataset, hidden_layers = 1, n_hidden = 2)
neural_net.train(n_epochs = 50000, lr_rate = 0.1)

print()
for layer in neural_net.network:
    for neuron in layer:
        print(neuron)
    print()