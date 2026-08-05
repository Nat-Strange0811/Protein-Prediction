# Parent class
# PyTorch
import torch
from configs import ClassifierConfig

from mlp_classifier import MLPClassifier

'''
Class - SimpleMLP:

This class implements a simple Multi-Layer Perceptron (MLP) classifier. It inherits from the abstract base class MLPClassifier, which 
enforces the implementation of specific methods in subclasses. The SimpleMLP class defines the architecture of the MLP model, including 
the input dimension, hidden layers, activation functions, dropout rates, and output layer.

Attributes:
- input_dim: The dimension of the input features.
- dropout_rate: The dropout rate for regularization.
- activation_function: The activation function to be used in the hidden layers.
- model_name: The name of the model.
- device: The device (CPU or GPU) on which the computations will be performed.
- hidden_dims: A list of integers specifying the dimensions of the hidden layers.

Methods:
- __init__(self, input_dim, dropout_rate, activation_function, model_name, hidden_dims, device=None): Initializes the SimpleMLP model 
with the specified parameters and builds the architecture.
- build(self): Constructs the architecture of the MLP model based on the specified input dimension, hidden layer dimensions, activation
functions, and dropout rates.
- forward(self, x): Defines the forward pass of the MLP model, which takes an input tensor x and returns the output of the model after 
passing through the defined layers.
'''
class SimpleMLP(MLPClassifier):
    
    # Initialize the SimpleMLP model with specified parameters and activates the build method to construct the architecture
    def __init__(self, input_dim, dropout_rate, config: ClassifierConfig):
        super().__init__()
        self.ID = "SimpleMLP"
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        self.activation_function = self.resolve_activation_function(config.activation_function)
        self.hidden_dims = config.hidden_dims
        self.build()

    
    '''
    Method - build:
    
    Constructs the architecture of the MLP model.
    
    Inputs:
    - None
    
    Outputs:
    - None
    
    Raises:
    - None
    '''
    def build(self):
        # Empty list to hold the layers of the MLP model
        layers = []
        # Create a list of dimensions for the layers, starting with the input dimension followed by the hidden layer dimensions
        dims = [self.input_dim] + self.hidden_dims
        
        # Loop through the dimensions to create the layers of the MLP model
        for i in range(len(dims) - 1):
            # For each layer, add a Linear layer, followed by the specified activation function and a Dropout layer (if dropout_rate > 0)
            layers.extend([
                torch.nn.Linear(dims[i], dims[i + 1]),
                self.activation_function(),
                torch.nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else torch.nn.Identity()
            ])
            
        # Add the final output layer with a single output unit and a Sigmoid activation function for binary classification
        layers.append(torch.nn.Linear(dims[-1], 1))
        layers.append(torch.nn.Sigmoid())
        
        # Create a Sequential model with the defined layers
        self.model = torch.nn.Sequential(*layers)
    
    # Defines the forward pass of the model.
    def forward(self, x):
        return self.model(x)
        
        
