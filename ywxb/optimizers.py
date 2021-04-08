import numpy as np
from dumpy.utils import get_activation_function

class SGD():
  def __init__(self,lr=0.01,wd=0.01,momentum= 0.9):
    self.lr = lr
    self.momentum = momentum
    self.wd = wd
    self.model = None
    self.grads_values = {}
    self.grads_values['w_sum'] = 0.0
  
  def backward(self,model,targets,outputs):
    self.model = model
    nn_architecture = self.model.nn_architecture
    parameters = self.model.parameters
    memory = self.model.memory

    dA_prev = 2*(outputs - targets)/outputs.shape[1] + self.wd * self.grads_values['w_sum']

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        dA_curr = dA_prev

        self.grads_values['w_sum'] += np.sum(parameters[f"W{layer_idx_curr}"]**2)

        self.grads_values[f"dW_prev{layer_idx_curr}"] = self.grads_values[f"dW{layer_idx_curr}"] if (type(self.grads_values.get(f"dW{layer_idx_curr}")) == np.ndarray) else 0.0
        self.grads_values[f"db_prev{layer_idx_curr}"] = self.grads_values[f"db{layer_idx_curr}"] if (type(self.grads_values.get(f"db{layer_idx_curr}")) == np.ndarray) else 0.0

        A_prev = memory[f"A{layer_idx_prev}"]
        Z_curr = memory[f"Z{layer_idx_curr}"]
        W_curr = parameters[f"W{layer_idx_curr}"]
        b_curr = parameters[f"b{layer_idx_curr}"]

        m = A_prev.shape[1]
        backward_activation_func = get_activation_function(layer['activation'],backward=True)
        dZ_curr = backward_activation_func(dA_curr, Z_curr)
        dW_curr = (1. / m) * np.matmul(dZ_curr, A_prev.T)
        db_curr = (1. / m) * np.sum(dZ_curr, axis=1, keepdims=True)
        dA_prev = np.matmul(W_curr.T, dZ_curr)

        self.grads_values[f"dW{layer_idx_curr}"] = dW_curr 
        self.grads_values[f"db{layer_idx_curr}"] = db_curr

    self.grads_values['w_sum'] = 0.0
        
    for idx, layer in enumerate(nn_architecture):
      layer_idx = idx + 1
      self.grads_values[f"dW{layer_idx}"] = self.grads_values[f"dW_prev{layer_idx}"] * self.momentum + self.lr * self.grads_values[f"dW{layer_idx}"]
      self.grads_values[f"db{layer_idx}"] = self.grads_values[f"db_prev{layer_idx}"] * self.momentum + self.lr * self.grads_values[f"db{layer_idx}"]
      
      parameters[f"W{layer_idx}"] -= self.grads_values[f"dW{layer_idx}"]        
      parameters[f"b{layer_idx}"] -= self.grads_values[f"db{layer_idx}"]
    
    self.model.parameters = parameters

    return self.model


class NAG():
  def __init__(self,lr=0.01,wd=0.01,momentum= 0.9):
    self.lr = lr
    self.momentum = momentum
    self.model = None
    self.wd = wd
    self.grads_values = {}
    self.grads_values['w_sum'] = 0.0
  
  def backward(self,model,targets,outputs):
    self.model = model
    nn_architecture = self.model.nn_architecture
    parameters = self.model.parameters
    memory = self.model.memory

    dA_prev = 2*(outputs - targets)/outputs.shape[1] + self.wd * self.grads_values['w_sum']

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        dA_curr = dA_prev

        self.grads_values['w_sum'] += np.sum(parameters[f"W{layer_idx_curr}"]**2)

        self.grads_values[f"dW_prev{layer_idx_curr}"] = self.grads_values[f"dW{layer_idx_curr}"] if (type(self.grads_values.get(f"dW{layer_idx_curr}")) == np.ndarray) else 0.0
        self.grads_values[f"db_prev{layer_idx_curr}"] = self.grads_values[f"db{layer_idx_curr}"] if (type(self.grads_values.get(f"db{layer_idx_curr}")) == np.ndarray) else 0.0

        A_prev = memory[f"A{layer_idx_prev}"]
        Z_curr = memory[f"Z{layer_idx_curr}"]
        W_curr = parameters[f"W{layer_idx_curr}"] - self.momentum * self.grads_values[f"dW_prev{layer_idx_curr}"]
        b_curr = parameters[f"b{layer_idx_curr}"] - self.momentum * self.grads_values[f"db_prev{layer_idx_curr}"]

        m = A_prev.shape[1]
        backward_activation_func = get_activation_function(layer['activation'],backward=True)
        dZ_curr = backward_activation_func(dA_curr, Z_curr)
        dW_curr = (1. / m) * np.matmul(dZ_curr, A_prev.T)
        db_curr = (1. / m) * np.sum(dZ_curr, axis=1, keepdims=True)
        dA_prev = np.matmul(W_curr.T, dZ_curr)

        self.grads_values[f"dW{layer_idx_curr}"] = dW_curr
        self.grads_values[f"db{layer_idx_curr}"] = db_curr
    
    self.grads_values['w_sum'] = 0.0
    
    for idx, layer in enumerate(nn_architecture):
      layer_idx = idx + 1
      self.grads_values[f"dW{layer_idx}"] = self.grads_values[f"dW_prev{layer_idx}"] * self.momentum + self.lr * self.grads_values[f"dW{layer_idx}"]
      self.grads_values[f"db{layer_idx}"] = self.grads_values[f"db_prev{layer_idx}"] * self.momentum + self.lr * self.grads_values[f"db{layer_idx}"]
      
      parameters[f"W{layer_idx}"] -= self.grads_values[f"dW{layer_idx}"]        
      parameters[f"b{layer_idx}"] -= self.grads_values[f"db{layer_idx}"]
    
    self.model.parameters = parameters

    return self.model

class Adagrad():
  def __init__(self,lr=0.01,momentum= 0.9):
    self.lr = lr
    self.model = None
    self.grads_values = {}
  
  def backward(self,model,targets,outputs):
    self.model = model
    nn_architecture = self.model.nn_architecture
    parameters = self.model.parameters
    memory = self.model.memory

    dA_prev = 2*(outputs - targets)/outputs.shape[1]

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        dA_curr = dA_prev

        if type(self.grads_values.get(f"dW{layer_idx_curr}")) == np.ndarray:
          self.grads_values[f"dW_square{layer_idx_curr}"] += self.grads_values[f"dW{layer_idx_curr}"]**2
        else:
          self.grads_values[f"dW_square{layer_idx_curr}"] = 1.0

        
        if type(self.grads_values.get(f"db{layer_idx_curr}")) == np.ndarray:
          self.grads_values[f"db_square{layer_idx_curr}"] += self.grads_values[f"db{layer_idx_curr}"]**2
        else:
          self.grads_values[f"db_square{layer_idx_curr}"] = 1.0

        A_prev = memory[f"A{layer_idx_prev}"]
        Z_curr = memory[f"Z{layer_idx_curr}"]
        W_curr = parameters[f"W{layer_idx_curr}"]
        b_curr = parameters[f"b{layer_idx_curr}"]

        m = A_prev.shape[1]
        backward_activation_func = get_activation_function(layer['activation'],backward=True)
        dZ_curr = backward_activation_func(dA_curr, Z_curr)
        dW_curr = (1. / m) * np.matmul(dZ_curr, A_prev.T)
        db_curr = (1. / m) * np.sum(dZ_curr, axis=1, keepdims=True)
        dA_prev = np.matmul(W_curr.T, dZ_curr)

        self.grads_values[f"dW{layer_idx_curr}"] = dW_curr
        self.grads_values[f"db{layer_idx_curr}"] = db_curr
    
    for idx, layer in enumerate(nn_architecture):
      layer_idx = idx +1
      parameters[f"W{layer_idx}"] -= (self.lr/np.sqrt(self.grads_values[f"dW_square{layer_idx}"]+1e-8)) * self.grads_values[f"dW{layer_idx}"]     
      parameters[f"b{layer_idx}"] -= (self.lr/np.sqrt(self.grads_values[f"db_square{layer_idx}"]+1e-8)) * self.grads_values[f"db{layer_idx}"]
    
    self.model.parameters = parameters

    return self.model

class RMSprop():
  def __init__(self,lr=0.01,wd= 0.01,momentum=0.9):
    self.lr = lr
    self.wd = wd
    self.model = None
    self.momentum = momentum
    self.grads_values = {}
    self.grads_values['w_sum'] = 0.0

  def backward(self,model,targets,outputs):
    self.model = model
    nn_architecture = self.model.nn_architecture
    parameters = self.model.parameters
    memory = self.model.memory

    dA_prev = 2*(outputs - targets)/outputs.shape[1] + self.wd *self.grads_values['w_sum']

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        dA_curr = dA_prev

        self.grads_values['w_sum'] += np.sum(parameters[f"W{layer_idx_curr}"]**2)

        if type(self.grads_values.get(f"dW{layer_idx_curr}")) == np.ndarray:
          self.grads_values[f"dW_square{layer_idx_curr}"] = self.momentum * self.grads_values[f"dW_square{layer_idx_curr}"] +  (self.grads_values[f"dW{layer_idx_curr}"]**2)
        else:
          self.grads_values[f"dW_square{layer_idx_curr}"] = 1.0
        
        if type(self.grads_values.get(f"db{layer_idx_curr}")) == np.ndarray:
          self.grads_values[f"db_square{layer_idx_curr}"] = self.momentum * self.grads_values[f"db_square{layer_idx_curr}"] + (1.0 - self.momentum) * (self.grads_values[f"db{layer_idx_curr}"]**2)
        else:
          self.grads_values[f"db_square{layer_idx_curr}"] = 1.0

        A_prev = memory[f"A{layer_idx_prev}"]
        Z_curr = memory[f"Z{layer_idx_curr}"]
        W_curr = parameters[f"W{layer_idx_curr}"]
        b_curr = parameters[f"b{layer_idx_curr}"]

        m = A_prev.shape[1]
        backward_activation_func = get_activation_function(layer['activation'],backward=True)
        dZ_curr = backward_activation_func(dA_curr, Z_curr)
        dW_curr = (1. / m) * np.matmul(dZ_curr, A_prev.T)
        db_curr = (1. / m) * np.sum(dZ_curr, axis=1, keepdims=True)
        dA_prev = np.matmul(W_curr.T, dZ_curr)

        self.grads_values[f"dW{layer_idx_curr}"] = dW_curr
        self.grads_values[f"db{layer_idx_curr}"] = db_curr

    self.grads_values['w_sum'] = 0.0
    
    for idx, layer in enumerate(nn_architecture):
      layer_idx = idx +1
      self.grads_values[f'dW{layer_idx}'] = self.lr/np.sqrt(self.grads_values[f"dW_square{layer_idx}"]+1e-8) * self.grads_values[f"dW{layer_idx}"]
      self.grads_values[f'db{layer_idx}'] = self.lr/np.sqrt(self.grads_values[f"db_square{layer_idx}"]+1e-8) * self.grads_values[f"db{layer_idx}"]
      
      parameters[f"W{layer_idx}"] -= self.lr * self.grads_values[f"dW{layer_idx}"]        
      parameters[f"b{layer_idx}"] -= self.lr * self.grads_values[f"db{layer_idx}"]
    
    self.model.parameters = parameters

    return self.model

class Adam():
  def __init__(self,lr=0.01,wd=0.01,beta1=0.9,beta2=0.999):
    self.lr = lr
    self.wd = wd
    self.model = None
    self.grads_values = {}
    self.beta1 = beta1
    self.beta2 = beta2
    self.grads_values['w_sum'] = 0.0
  
  def backward(self,model,targets,outputs):
    self.model = model
    nn_architecture = self.model.nn_architecture
    parameters = self.model.parameters
    memory = self.model.memory

    dA_prev = 2*(outputs - targets)/outputs.shape[1] + self.wd *self.grads_values['w_sum']

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        dA_curr = dA_prev

        self.grads_values['w_sum'] += np.sum(parameters[f"W{layer_idx_curr}"]**2)

        if type(self.grads_values.get(f"dW{layer_idx_curr}")) == np.ndarray:
          self.grads_values[f"dW_sum{layer_idx_curr}"] += self.grads_values[f"dW{layer_idx_curr}"]
        else:
          self.grads_values[f"dW_sum{layer_idx_curr}"] = 0.0

        if type(self.grads_values.get(f"db{layer_idx_curr}")) == np.ndarray:
          self.grads_values[f"db_sum{layer_idx_curr}"] += self.grads_values[f"db{layer_idx_curr}"]
        else:
          self.grads_values[f"db_sum{layer_idx_curr}"] = 0.0

        if type(self.grads_values.get(f"dW{layer_idx_curr}")) == np.ndarray:
          vw_curr = self.beta2 * self.grads_values[f"dW_square{layer_idx_curr}"] + (1.0) * self.grads_values[f"dW{layer_idx_curr}"]**2
          self.grads_values[f"dW_square{layer_idx_curr}"] = vw_curr
        else:
          self.grads_values[f"dW_square{layer_idx_curr}"] = 1.0

        if type(self.grads_values.get(f"db{layer_idx_curr}")) == np.ndarray:
          vb_curr = self.beta2 * self.grads_values[f"db_square{layer_idx_curr}"] + (1.0) * self.grads_values[f"db{layer_idx_curr}"]**2
          self.grads_values[f"db_square{layer_idx_curr}"] = vb_curr
        else:
          self.grads_values[f"db_square{layer_idx_curr}"] = 1.0

        if self.grads_values.get("beta1"):
          self.grads_values["beta1"] *= self.beta1
        else:
          self.grads_values["beta1"] = self.beta1

        if self.grads_values.get("beta2"):
          self.grads_values["beta2"] *= self.beta2
        else:
          self.grads_values["beta2"] = self.beta2

        A_prev = memory[f"A{layer_idx_prev}"]
        Z_curr = memory[f"Z{layer_idx_curr}"]
        W_curr = parameters[f"W{layer_idx_curr}"]
        b_curr = parameters[f"b{layer_idx_curr}"]

        m = A_prev.shape[1]
        backward_activation_func = get_activation_function(layer['activation'],backward=True)
        dZ_curr = backward_activation_func(dA_curr, Z_curr)
        dW_curr = (1. / m) * np.matmul(dZ_curr, A_prev.T)
        db_curr = (1. / m) * np.sum(dZ_curr, axis=1, keepdims=True)
        dA_prev = np.matmul(W_curr.T, dZ_curr)

        mw_curr = self.beta1 * self.grads_values[f"dW_sum{layer_idx_curr}"] +  dW_curr
        mb_curr = self.beta1 * self.grads_values[f"db_sum{layer_idx_curr}"] +  db_curr

        self.grads_values[f"dW{layer_idx_curr}"] = dW_curr
        self.grads_values[f"db{layer_idx_curr}"] = db_curr

    self.grads_values['w_sum'] = 0.0
    
    for idx, layer in enumerate(nn_architecture):
      layer_idx = idx +1
      vtw_curr = self.grads_values[f"dW_square{layer_idx}"] / (1.0 - self.grads_values['beta2'])
      vtb_curr = self.grads_values[f"db_square{layer_idx}"] / (1.0 - self.grads_values['beta2'])

      mtw_curr = self.grads_values[f"dW{layer_idx}"] / (1.0 - self.grads_values['beta1'])
      mtb_curr = self.grads_values[f"db{layer_idx}"] / (1.0 - self.grads_values['beta1'])

      parameters[f"W{layer_idx}"] -= self.lr/(np.sqrt(vtw_curr) + 1e-8) * mtw_curr 
      parameters[f"b{layer_idx}"] -= self.lr/(np.sqrt(vtb_curr) + 1e-8) * mtb_curr
    
    self.model.parameters = parameters

    return self.model

class NAdam():
  def __init__(self,lr=0.01,wd=0.01,beta1=0.9,beta2=0.999):
    self.lr = lr
    self.wd = wd
    self.model = None
    self.grads_values = {}
    self.beta1 = beta1
    self.beta2 = beta2
    self.grads_values['w_sum'] = 0.0
  
  def backward(self,model,targets,outputs):
    self.model = model
    nn_architecture = self.model.nn_architecture
    parameters = self.model.parameters
    memory = self.model.memory

    dA_prev = 2*(outputs - targets)/outputs.shape[1] + self.wd *self.grads_values['w_sum']

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        dA_curr = dA_prev

        self.grads_values['w_sum'] += np.sum(parameters[f"W{layer_idx_curr}"]**2)

        if type(self.grads_values.get(f"dW{layer_idx_curr}")) == np.ndarray:
          self.grads_values[f"dW_sum{layer_idx_curr}"] += self.grads_values[f"dW{layer_idx_curr}"]
        else:
          self.grads_values[f"dW_sum{layer_idx_curr}"] = 0.0

        if type(self.grads_values.get(f"db{layer_idx_curr}")) == np.ndarray:
          self.grads_values[f"db_sum{layer_idx_curr}"] += self.grads_values[f"db{layer_idx_curr}"]
        else:
          self.grads_values[f"db_sum{layer_idx_curr}"] = 0.0

        if type(self.grads_values.get(f"dW{layer_idx_curr}")) == np.ndarray:
          vw_curr = self.beta2 * self.grads_values[f"dW_square{layer_idx_curr}"] + (1.0) * self.grads_values[f"dW{layer_idx_curr}"]**2
          self.grads_values[f"dW_square{layer_idx_curr}"] = vw_curr
        else:
          self.grads_values[f"dW_square{layer_idx_curr}"] = 1.0

        if type(self.grads_values.get(f"db{layer_idx_curr}")) == np.ndarray:
          vb_curr = self.beta2 * self.grads_values[f"db_square{layer_idx_curr}"] + (1.0) * self.grads_values[f"db{layer_idx_curr}"]**2
          self.grads_values[f"db_square{layer_idx_curr}"] = vb_curr
        else:
          self.grads_values[f"db_square{layer_idx_curr}"] = 1.0

        ## beta
        if self.grads_values.get("beta1"):
          self.grads_values["beta1"] *= self.beta1
        else:
          self.grads_values["beta1"] = self.beta1

        if self.grads_values.get("beta2"):
          self.grads_values["beta2"] *= self.beta2
        else:
          self.grads_values["beta2"] = self.beta2

        A_prev = memory[f"A{layer_idx_prev}"]
        Z_curr = memory[f"Z{layer_idx_curr}"]
        W_curr = parameters[f"W{layer_idx_curr}"]
        b_curr = parameters[f"b{layer_idx_curr}"]

        m = A_prev.shape[1]
        backward_activation_func = get_activation_function(layer['activation'],backward=True)
        dZ_curr = backward_activation_func(dA_curr, Z_curr)
        dW_curr = (1. / m) * np.matmul(dZ_curr, A_prev.T)
        db_curr = (1. / m) * np.sum(dZ_curr, axis=1, keepdims=True)
        dA_prev = np.matmul(W_curr.T, dZ_curr)

        mw_curr = self.beta1 * self.grads_values[f"dW_sum{layer_idx_curr}"] + (1.0 )* dW_curr
        mb_curr = self.beta1 * self.grads_values[f"db_sum{layer_idx_curr}"] + (1.0 )* db_curr

        self.grads_values[f"dW{layer_idx_curr}"] = dW_curr
        self.grads_values[f"db{layer_idx_curr}"] = db_curr

    self.grads_values['w_sum'] = 0.0
    
    for idx, layer in enumerate(nn_architecture):
      layer_idx = idx +1
      vtw_curr = self.grads_values[f"dW_square{layer_idx}"] / (1.0 - self.grads_values['beta2'])
      vtb_curr = self.grads_values[f"db_square{layer_idx}"] / (1.0 - self.grads_values['beta2'])

      mtw_curr = self.grads_values[f"dW{layer_idx}"] / (1.0 - self.grads_values['beta1'])
      mtb_curr = self.grads_values[f"db{layer_idx}"] / (1.0 - self.grads_values['beta1'])

      parameters[f"W{layer_idx}"] -= self.lr/(np.sqrt(vtw_curr) + 1e-8) * (self.beta1 * mtw_curr + ((1-self.beta1) *\
                                                                           self.grads_values[f"dW{layer_idx}"])/(1.0 - self.grads_values['beta1']))

      parameters[f"b{layer_idx}"] -= self.lr/(np.sqrt(vtb_curr) + 1e-8) * (self.beta1 * mtb_curr + ((1-self.beta1) *\
                                                                           self.grads_values[f"db{layer_idx}"])/(1.0 - self.grads_values['beta1']))

    
    self.model.parameters = parameters

    return self.model

