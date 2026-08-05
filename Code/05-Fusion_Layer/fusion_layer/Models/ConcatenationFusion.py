#PyTorch import
import torch
from configs import FusionLayerConfig

# Abstract Base Class
from fusion_layer import FusionLayer

'''
Class - ConcatenationFusion:

This class implements a specific fusion strategy that concatenates multiple embeddings into a single representation.

Methods:
- __init__(self, d_model: int, n_modalities: int, device=None): Initializes the ConcatenationFusion layer with the specified model dimension, 
number of modalities, and optional device parameter.

- forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor: Implements the forward pass by concatenating the input embeddings along 
the feature dimension (dim=1).

- output_dim(self) -> int: Returns the output dimension of the fused representation, which is the product of the model dimension and 
the number of modalities.
'''
class ConcatenationFusion(FusionLayer):
    
    def __init__(self, d_model: int, n_modalities: int, config: FusionLayerConfig):
        if config is None:
            raise ValueError("Config dictionary must be provided for fusion layer initialization.")
            
        
        # Call the constructor of the parent class (FusionLayer) to initialize the module
        super().__init__()
        
        # Store the model dimension and number of modalities for later use
        self.d_model = d_model
        self.n_modalities = n_modalities
        self.ID = "ConcatenationFusion"
    
    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(embeddings, dim=1)
    
    def output_dim(self) -> int:
        return self.d_model * self.n_modalities