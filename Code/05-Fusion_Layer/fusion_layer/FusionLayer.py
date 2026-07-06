# PyTorch
import torch

# Abstract Base Class
from abc import ABC, abstractmethod

'''
Class - FusionLayer:

Abstract base class for fusion layers defining a structure for combining multiple embeddings into a single representation. The class 
inherits from torch.nn.Module and ABC (Abstract Base Class) to enforce the implementation of specific methods in subclasses. 
torch.nn.Module enables access to PyTorch's neural network functionalities.

Attributes:
- device: The device (CPU or GPU) on which the computations will be performed.

Methods:
- __init__(self, device=None): Initializes the FusionLayer with an optional device parameter.
- forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor: Abstract method to define the forward pass for combining embeddings.
- output_dim(self) -> int: Abstract method to return the output dimension of the fused representation.
- build(self): Method to build the fusion layer, can be overridden in subclasses if needed.
'''
class FusionLayer(torch.nn.Module, ABC):
    
    def __init__(self, device=None):
        # Call the constructor of the parent class (torch.nn.Module) to initialize the module
        super().__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    @abstractmethod
    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        pass
    
    @abstractmethod
    def output_dim(self) -> int:
        pass
    
    def build(self):
        pass