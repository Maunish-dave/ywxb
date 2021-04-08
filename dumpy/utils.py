import numpy as np

def get_activation_function(activation='relu',backward=False):

  def sigmoid(Z):
      return 1. /(1. + np.exp(-Z)) 

  def relu(Z):
      return np.maximum(0,Z)

  def softmax(z):
    exps = np.exp(z - z.max())
    return exps / np.sum(exps, axis=0)

  def tanh(z):
    return np.tanh(z)

  def sigmoid_backward(dA, Z):
      sig = sigmoid(Z)
      return dA * sig * (1. - sig)

  def relu_backward(dA, Z):
      dZ = np.array(dA, copy = True)
      dZ[Z <= 0.0] = 0.0
      return dZ

  def softmax_backward(dA,z):
    return dA

  def tanh_backward(dA,z):
    a = np.tanh(z)
    return dA * (1.0 - a**2)
  
  if activation == 'relu' and not backward:
    return relu
  elif activation == 'relu' and backward:
    return relu_backward

  elif activation == 'sigmoid' and not backward:
    return sigmoid
  elif activation == 'sigmoid' and backward:
    return sigmoid_backward

  elif activation == 'softmax' and not backward:
    return softmax
  elif activation == 'softmax' and backward:
    return softmax_backward 

  elif activation == 'tanh' and not backward:
    return tanh
  elif activation == 'tanh' and backward:
    return tanh_backward 
  else:
    raise Exception(f'{activation} is not a activation function; options are "relu", "sigmoid", "softmax", "tanh" ')