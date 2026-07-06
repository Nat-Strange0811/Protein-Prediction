#PyTorch imports
import torch
import torch.nn as nn

# typing imports
from typing import List

# FusionLayer and MLPClassifier imports
from fusion_layer import FusionLayer
from mlp_classifier import MLPClassifier

'''
Class - FusionModel:
This class defines a fusion model that combines multiple embeddings into a single representation using a specified fusion layer and then
classifies the fused representation using a specified MLP classifier. The model consists of projection layers to project the input
embeddings to a common dimension, normalization layers to normalize the projected embeddings, a fusion layer to combine the normalised
embeddings, and an MLP classifier to classify the fused representation.

Attributes:
-embedding_dims: A list of integers specifying the dimensions of the input embeddings.
-d_model: An integer specifying the dimension to which the input embeddings will be projected.
-fusion_layer: An instance of a FusionLayer that defines the fusion strategy to be used for combining the embeddings.
-mlp_classifier: An instance of an MLPClassifier that defines the architecture of the classifier
-projection_layers: A ModuleList containing Linear layers to project the input embeddings to the common dimension d_model.
-norms: A ModuleList containing LayerNorm layers to normalize the projected embeddings.

Methods:
-__init__(self, embedding_dims: list[int], d_model: int, fusion_layer: FusionLayer, mlp_classifier: MLPClassifier): Initializes the 
FusionModel with the specified parameters.
-forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor: Defines the forward pass of the FusionModel.
'''
class FusionModel(nn.Module):
    # Innit method to initialize the FusionModel with the specified parameters.
    def __init__(self,
                 embedding_dims: list[int],
                 d_model: int,
                 fusion_layer: FusionLayer,
                 mlp_classifier: MLPClassifier,):
        
        super().__init__()
        
        # For each embedding dimension, create a Linear layer to project the embedding to the common dimension d_model
        self.projection_layers = nn.ModuleList([
            nn.Linear(embedding_dim, d_model) for embedding_dim in embedding_dims
        ])
        
        # For each embedding dimension, create a LayerNorm layer to normalize the projected embedding
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in embedding_dims
        ])
        
        self.fusion_layer = fusion_layer
        self.mlp_classifier = mlp_classifier
        
    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        
        # Project each embedding to the common dimension d_model and normalize it using the corresponding projection and normalization layers
        projected_embeddings = [
            self.norms[i](self.projection_layers[i](embeddings[i])) for i in range(len(embeddings))
        ]
        
        # Fuse the normalized embeddings using the specified fusion layer
        fused_embedding = self.fusion_layer(projected_embeddings)
        
        return self.mlp_classifier(fused_embedding)
    