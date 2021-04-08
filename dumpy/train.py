import  time
import numpy as np
from dumpy.optimizers import *
from dumpy.utils import get_activation_function

class Model():
  def __init__(self,nn_architecture,initializer='random'):
    np.random.seed(42)
    self.nn_architecture = nn_architecture
    self.parameters = dict()
    self.memory = dict()

    for idx,layer in enumerate(self.nn_architecture):
      layer_idx = idx + 1
      if initializer == 'random':
        self.parameters[f"W{layer_idx}"] = np.random.randn(layer['output_dim'],layer['input_dim'])  * np.sqrt(1. / layer['input_dim'])
        self.parameters[f"b{layer_idx}"] = np.random.randn(layer['output_dim'],1) * np.sqrt(1. / layer['input_dim'])  

      elif initializer == 'xavier':
        l = np.sqrt(6.0 / (layer['output_dim'] +layer['input_dim']))
        self.parameters[f"W{layer_idx}"] = np.random.uniform(-l,l,size=(layer['output_dim'],layer['input_dim']))
        self.parameters[f"b{layer_idx}"] = np.random.uniform(-l,l,size=(layer['output_dim'],1))

      else:
        raise Exception(f"{initializer} is not a initializer options are 'random' and 'xavier' ")
    

  def forward(self,xb):
    self.memory = dict()
    A_curr = xb
    for idx, (layer) in enumerate(self.nn_architecture):
      layer_idx = idx + 1
      A_prev = A_curr

      W_curr = self.parameters[f"W{layer_idx}"]
      b_curr = self.parameters[f"b{layer_idx}"]

      Z_curr = np.matmul(W_curr,A_prev) + b_curr
      active_function = get_activation_function(layer['activation'])
      A_curr = active_function(Z_curr)

      self.memory[f"A{idx}"] = A_prev
      self.memory[f"Z{layer_idx}"] = Z_curr
    
    return A_curr

def train(X_train,y_train,X_test,y_test,nn_architecture,optim='sgd',initializer='random',lr=0.01,wd=0.01,momentum=0.9,beta1=0.9,beta2=0.99,epochs=500,batch_size=32,verbose=10,wandb_log=None):
  
  def get_data_loader(X,y,batch_size,shuffle=False):
    if shuffle:
      permutation = np.random.permutation(X.shape[1])
      X = X[:,permutation]
      X = y[:,permutation]

    for i in range(0,len(X),batch_size):
      xb = X[:,i:i+batch_size]
      yb = y[:,i:i+batch_size]
      yield xb, yb

  def compute_accuracy(outputs, targets):
    return np.mean(np.argmax(targets,0)==np.argmax(outputs,0))

  def compute_loss(targets,outputs):
    L_sum = np.sum(np.multiply(targets, np.log(outputs)))
    m = targets.shape[1]
    L = -(1/m) * L_sum
    return L

  def one_hot(Y):
      one_hot_Y = np.zeros((Y.size, Y.max() + 1))
      one_hot_Y[np.arange(Y.size), Y] = 1
      return one_hot_Y

  X_train = X_train.T
  X_test = X_test.T
  y_train = one_hot(y_train).T
  y_test = one_hot(y_test).T

  model = Model(nn_architecture,initializer=initializer)

  optimizers = {"sgd":SGD,
                "nag":NAG,
                "rmsprop":RMSprop,
                "adam":Adam,
                "nadam":NAdam}

  if optimizers.get(optim):
    if optim == 'adam' or optim == 'nadam':
      optimizer = optimizers.get(optim)(lr = lr,wd=wd,beta1=beta1,beta2=beta2)
    else:
      optimizer = optimizers.get(optim)(lr = lr,wd=wd,momentum=momentum)
  else:
    raise Exception(f"{optim} is not a optimizer: options are 'sgd', 'nag','adagrad', 'adam', 'nadam' ")

  best_acc = 0.0
  for e in range(epochs):
    train_dl = get_data_loader(X_train,y_train,batch_size=32)
    start = time.time()

    for i,(inputs,targets) in enumerate(train_dl):
      outputs = model.forward(inputs)
      model = optimizer.backward(model,targets,outputs)

    train_outputs = model.forward(X_train)
    train_acc = compute_accuracy(train_outputs,y_train)
    train_loss = compute_loss(y_train,train_outputs)

    test_outputs = model.forward(X_test)
    test_acc = compute_accuracy(test_outputs,y_test)
    test_loss = compute_loss(y_test,test_outputs)

    epoch_time = time.time() - start

    if wandb_log:
      wandb_log.log({"Train loss":train_loss,'Train acc':train_acc,"Test loss":test_loss,'Test acc':test_acc})

    if e % verbose == 0:
      print(f"Iteration {e} : Train Accuracy: {train_acc:.3f} | Train Loss: {train_loss:.3f} | Test Accuracy: {test_acc:.3f} | Test Loss: {test_loss:.3f} | Epoch Time: {epoch_time:.2f}s")
      
  return model