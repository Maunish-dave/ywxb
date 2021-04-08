# YWXB
This repository my implementation of deep neural networks using numpy.

It allows you to use fully connected Neural Netowrk with Many options for Optimizers and Activation Functions

Here is an example of how to use dumpy.

You only need to import train function from dumpy.

```python
from ywxb.train import  train
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X_train, y_test = make_classification(n_samples=10000,n_features=10)
X_train, X_test, y_train, y_test = train_test_split(X_train,y_test,test_size=0.2)

input_shape = X_train.shape[1]
output_shape = 2
```

you can create you own architecture using list of dict <br/>
Here you have to specify three thing input_dim, output_dim and activation<br/>
you can use 'relu', 'tanh', 'sigmoid' , 'softmax' for activation.

```python
nn_architecture = [
    {"input_dim": input_shape, "output_dim": 64, "activation": "relu"},
    {"input_dim": 64, "output_dim": 32, "activation": "relu"},
    {"input_dim": 32, "output_dim": output_shape, "activation": "sigmoid"},
]
```

In config set following things.<br/>
Optim: options are "sgd","nag","rmsprop", "adam", "nadam".<br/>
lr is learning rate, wd is weight decay.<br/>
momentum will be used if applied to optimizer.<br/>
verbose will print train and test accuracy.


```python
config = {'optim':'nadam',
          'lr':0.1,
          'wd':0.01,
          'epochs':500,
          'momentum':0.9,
          'batch_size':128,
          'verbose':50}

model = train(X_train,y_train,
              X_test,y_test,
              nn_architecture=nn_architecture,
              optim=config['optim'],
              epochs=config['epochs'],
              lr=config['lr'],
              momentum=config['momentum'],
              batch_size=config['batch_size'],
              verbose=config['verbose'])
 ```
                  
 Output:
 
    Iteration 0 : Train Accuracy: 0.569 | Train Loss: 0.884 | Test Accuracy: 0.562 | Test Loss: 0.873 | Epoch Time: 0.02s
    Iteration 50 : Train Accuracy: 0.643 | Train Loss: 0.827 | Test Accuracy: 0.646 | Test Loss: 0.816 | Epoch Time: 0.02s
    Iteration 100 : Train Accuracy: 0.726 | Train Loss: 0.746 | Test Accuracy: 0.739 | Test Loss: 0.736 | Epoch Time: 0.02s
    Iteration 150 : Train Accuracy: 0.780 | Train Loss: 0.663 | Test Accuracy: 0.792 | Test Loss: 0.653 | Epoch Time: 0.02s
    Iteration 200 : Train Accuracy: 0.810 | Train Loss: 0.586 | Test Accuracy: 0.815 | Test Loss: 0.577 | Epoch Time: 0.02s
    Iteration 250 : Train Accuracy: 0.823 | Train Loss: 0.519 | Test Accuracy: 0.830 | Test Loss: 0.511 | Epoch Time: 0.02s
    Iteration 300 : Train Accuracy: 0.839 | Train Loss: 0.458 | Test Accuracy: 0.848 | Test Loss: 0.451 | Epoch Time: 0.02s
    Iteration 350 : Train Accuracy: 0.852 | Train Loss: 0.406 | Test Accuracy: 0.855 | Test Loss: 0.401 | Epoch Time: 0.02s
    Iteration 400 : Train Accuracy: 0.861 | Train Loss: 0.366 | Test Accuracy: 0.865 | Test Loss: 0.362 | Epoch Time: 0.02s
    Iteration 450 : Train Accuracy: 0.865 | Train Loss: 0.338 | Test Accuracy: 0.867 | Test Loss: 0.336 | Epoch Time: 0.03s
 
 **This Library is under progress, new features will be added soon.**
                  
   
