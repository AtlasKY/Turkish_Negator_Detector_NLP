import fasttext
import torch
from torch.nn import Linear, Sigmoid, ReLU, Dropout, Sequential, Module, BatchNorm1d, Dropout

#class for the neural network model
class Net(Module):
    def __init__(self, wv_dim):
        super(Net, self).__init__()
        self.drop_prob = 0.2
        self.drop = Dropout(self.drop_prob)
        self.dim = wv_dim

        #The linear fully connected layer at the end of the network that outputs a classification
        self.out_layers = Sequential(
            BatchNorm1d(self.dim),    
            Linear(self.dim, 256),
            BatchNorm1d(256),
            ReLU(),
            Linear(256, 512),
            BatchNorm1d(512),
            ReLU(),
            Linear(512, 1024),
            BatchNorm1d(1024),
            self.drop,
            ReLU(),
            Linear(1024, 1024),
            #BatchNorm1d(1024),
            ReLU(),
            Linear(1024, 1024),
            self.drop,
            BatchNorm1d(1024),
            ReLU(),
            Linear(1024, 512),
            BatchNorm1d(512),
            ReLU(),
            Linear(512, 256),
            ReLU(),
            Linear(256, 256),
            ReLU(),
            Linear(256, 256),
            BatchNorm1d(256),
            ReLU(),
            Linear(256, 256),
            BatchNorm1d(256),
            ReLU(),
            Linear(256, 64),
            BatchNorm1d(64),
            ReLU(),
            Linear(64, 64),
            BatchNorm1d(64),
            ReLU(),
            Linear(64, 16),            
            BatchNorm1d(16),
            ReLU(),
            Linear(16, 1),
            Sigmoid()
        )


    #Define the forward pass through the model 
    def forward(self, x):
        x = torch.sum(x, 1)
        x = x.reshape((x.shape[0],x.shape[1]))
        x = self.out_layers(x)
        return x