a = 5
b = 7
c = a + b
print(c)

import numpy as np
x=np.array([1,2,3])
x.__class__
x.shape
x.ndim

w = np.array([[1,2,3],[4,5,6]])
w.shape
w.ndim

w = np.array([[1,2,3],[4,5,6]])
x = np.array([[0,1,2],[3,4,5]])
w+x
w*x

A= np.array([[1,2],[3,4]])
A*10

A= np.array([[1,2],[3,4]])
b= np.array([10,20])
A*b

a = np.array([1,2,3])
b = np.array([4,5,6])
np.dot(a,b)

A= np.array([[1,2],[3,4]])
B= np.array([[5,6],[7,8]])
np.matmul(A,B)

import numpy as np
w1 = np.random.randn(2,4)
b1 = np.random.randn(4)
x = np.random.randn(10,2)
h = np.matmul(x,w1) + b1

def sigmoid(x):
    return 1/(1 + np.exp(-x))
a = sigmoid(h)

import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.random.randn(10,2)
w1 = np.random.randn(2,4)
b1 = np.random.randn(4)
w2 = np.random.randn(4,3)
b2 = np.random.randn(3)

h = np.matmul(x,w1) + b1
a = sigmoid(h)
s = np.matmul(a,w2) + b2

import numpy as np

class sigmoid :
    def __int__(self):
        self.params = []

    def forward(self):
        return 1/(1+np.exp(-x))

class affine :
    def __int__(self):
        self.params = [w,b]

    def forward(self,x):
        w, b = self.params
        out = np.matmul(x,w)+b
        return out

class twolayernet :
    def __int__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        w1 = np.random.randn(I,H)
        b1 = np.random.randn(H)
        w2 = np.random.randn(H,O)
        b2 = np.random.randn(O)

        self.layers = [
            affine(w1,b1),
            sigmoid(),
            affine(w2,b2)
        ]

        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
