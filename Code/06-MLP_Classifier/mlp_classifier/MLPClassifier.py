# PyTorch
# Abstract Base Class
from abc import ABC, abstractmethod

import torch

'''
Class - MLPClassifier:

This class serves as an abstract base class for Multi-Layer Perceptron (MLP) classifiers. It inherits from torch.nn.Module and ABC 
(Abstract Base Class) to enforce the implementation of specific methods in subclasses. torch.nn.Module enables access to PyTorch's 
neural network functionalities.

Attributes:
- device: The device (CPU or GPU) on which the computations will be performed.

Methods:
- __init__(self, device=None): Initializes the MLPClassifier with an optional device parameter.
- forward(self, x): Abstract method to define the forward pass of the MLP classifier.
- build(self): Abstract method to build the architecture of the MLP classifier.
'''

class MLPClassifier(torch.nn.Module, ABC):
    
    ACTIVATION_REGISTRY = {
        'relu': torch.nn.ReLU,
        'gelu': torch.nn.GELU,
        'tanh': torch.nn.Tanh,
        'sigmoid': torch.nn.Sigmoid,
    }
    
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def build(self):
        pass
    
    def resolve_activation_function(self, activation_name):
        if activation_name not in self.ACTIVATION_REGISTRY:
            raise ValueError(f"Unknown activation '{activation_name}'. Available: {list(self.ACTIVATION_REGISTRY.keys())}")
        return self.ACTIVATION_REGISTRY[activation_name]
    