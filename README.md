# dumpy
This repository my implementation of deep neural networks using numpy.

It allows you to use fully connected Neural Netowrk with Many options for Optimizers and Activation Functions

Here is an example of how to use dumpy.

You only need to import train function from dumpy.

`from dumpy.train import  train
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X_train, y_test = make_classification(n_samples=10000,n_features=10)
X_train, X_test, y_train, y_test = train_test_split(X_train,y_test,test_size=0.2)

input_shape = X_train.shape[1]
output_shape = 2`

you can create you own architecture using list of dict <br/>
Here you have to specify three thing input_dim, output_dim and activation<br/>
you can use 'relu', 'tanh', 'sigmoid' , 'softmax' for activation.

`
nn_architecture = [
    {"input_dim": input_shape, "output_dim": 64, "activation": "relu"},
    {"input_dim": 64, "output_dim": 32, "activation": "relu"},
    {"input_dim": 32, "output_dim": output_shape, "activation": "sigmoid"},
]
`

In config set following things.<br/>
Optim: options are "sgd","nag","rmsprop", "adam", "nadam".<br/>
lr is learning rate, wd is weight decay.<br/>
momentum will be used if applied to optimizer.<br/>
verbose will print train and test accuracy.


`
config = {'optim':'nadam',
          'lr':0.1,
          'wd':0.01,
          'epochs':500,
          'momentum':0.9,
          'batch_size':128,
          'verbose':50}
`
`      
model = train(X_train,y_train,
              X_test,y_test,
              nn_architecture=nn_architecture,
              optim=config['optim'],
              epochs=config['epochs'],
              lr=config['lr'],
              momentum=config['momentum'],
              batch_size=config['batch_size'],
              verbose=config['verbose'])`
