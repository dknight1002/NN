import pandas as pd
from keras.layers import Layer,Dense,Flatten
from keras import backend as K
from keras.models import Sequential
from keras.losses import mse

xor = pd.DataFrame([[0, 0, 0],[0, 1, 1],[1, 0, 1],[1, 1, 0]])
x = xor.iloc[:, :2]
y = xor.iloc[:, -1]
print(x,y)

class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',shape=(int(input_shape[1]), self.units),initializer='uniform',trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

model = Sequential()
model.add(Flatten(input_shape=(2, 1)))
model.add(RBFLayer(50, 0.5))
model.add(Dense(1, activation='ReLU'))

model.compile(optimizer='sgd', loss=mse)

model.fit(x, y, epochs=500)
print(model.predict([[0, 0]]))
print(model.predict([[0, 1]]))
print(model.predict([[1, 0]]))
print(model.predict([[1, 1]]))