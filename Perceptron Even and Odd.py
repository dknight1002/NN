import numpy as np
class Perceptron(object):
    def __init__(self, input_size, lr=0.1, epochs=10000):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr
    def activation_fn(self, x):
        return 1 if x>=0 else 0
    def predict(self, x):
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a
    def fit(self, X, d):
        for _ in range(self.epochs):
            sum_error=0
            print("Current Weights:",self.W)
            for i in range(len(d)):
                x = np.insert(X[i], 0, 1)
                y = self.predict(x)
                print("Expected Value:",d[i],"Predicted Value:",y)
                e = d[i] - y
                sum_error+=e**2
                self.W = self.W + self.lr * e * x
            print("epoch:",_," Learning Rate:",self.lr," SUM ERROR:",sum_error)
            print("Updated Weights:",self.W)
            print("\n")
            if(sum_error==0):
                break

X=np.array([[0,0,0,1],[1,1,1,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[1,0,1,0]]) #converting the numbers to binary
d=np.array([1,1,0,1,0,1,0,1,0,1,0]) #1 is odd, 0 is even

perceptron = Perceptron(input_size=4)
perceptron.fit(X, d)
print("Final Weights:",perceptron.W)