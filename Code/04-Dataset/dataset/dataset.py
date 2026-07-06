#PyTorch imports
import torch
from torch.utils.data import Dataset

#Path and typing imports
from pathlib import Path
from typing import List, Dict

'''
Class - ProteinDataset:

ProteinDataset is a custom PyTorch Dataset class that handles loading and managing protein embeddings and their corresponding labels.
The three methods are required for any PyTorch Dataset class: __init__, __len__, and __getitem__.

Attributes:
- embedding_dir (Path): Directory containing the embeddings for each modality.
- modalities (List[str]): List of modality names to be loaded.
- labels_path (str): Path to the file containing the labels.
- embeddings (Dict[str, torch.Tensor]): Dictionary storing embeddings for each modality.
- prot_ids (List[str]): List of protein IDs corresponding to the embeddings and labels.
- labels (torch.Tensor): Tensor containing the labels for each protein.

Methods:
- __init__(self, embedding_dir: str, modalities: List[str], labels_path: str): Initialises the dataset by loading embeddings and labels
- __len__(self): Returns the number of samples in the dataset.
- __getitem__(self, idx): Returns the embeddings, label, and protein ID for a given index.
'''
class ProteinDataset(Dataset):
    
    '''
    Method - __init__:
    
    Initialises the ProteinDataset by loading embeddings and labels from the specified directories and files. Ensures that the protein IDs
    for each modality are aligned with the protein IDs in the labels file and each other.
    
    Inputs:
    - embedding_dir (str): Directory containing the embeddings for each modality.
    - modalities (List[str]): List of modality names to be loaded.
    - labels_path (str): Path to the file containing the labels.
    
    Raises:
    - ValueError: If the protein IDs for any modality do not match the protein IDs in the labels file or if the order of protein IDs differs between modalities.
    - ValueError: If the number of embedding directories does not match the number of modalities.
    
    Returns:
    - None
    '''
    def __init__(self, embedding_dirs: List[str], modalities: List[str], labels_path: str):
        # Initialize the dataset by loading embeddings and labels
        self.embedding_dirs = [Path(dir) for dir in embedding_dirs]
        self.modalities = modalities
        
        if len(self.embedding_dirs) != len(self.modalities):
            raise ValueError("Number of embedding directories must match number of modalities.")
        
        # Load labels
        label_data = torch.load(labels_path, weights_only=True)
        label_prot_ids = set(label_data['prot_ids'])
        
        # Load embeddings per modality
        self.embeddings: Dict[str, torch.Tensor] = {}
        self.prot_ids = None
        
        # Iterate through each modality to load embeddings and validate alignment with labels
        for i, directory in enumerate(self.embedding_dirs):
            modality = self.modalities[i]
            # Load embeddings for the modality
            data = torch.load(directory, weights_only=True)
            modality_prot_ids = set(data['prot_ids'])
            
            # Validate modaity and label proteins are identical
            if modality_prot_ids != label_prot_ids:
                missing = label_prot_ids - modality_prot_ids
                extra = modality_prot_ids - label_prot_ids
                raise ValueError(
                    f"Modality {modality} misaligned with labels. "
                    f"Missing: {missing}, Extra: {extra}"
                )
            
            # Validate that the order of protein IDs is consistent across modalities
            if self.prot_ids is None:
                self.prot_ids = data['prot_ids']
            elif self.prot_ids != data['prot_ids']:
                raise ValueError(f"Modality {modality} prot_ids order differs from previous modalities")
            
            # Store the embeddings for the modality
            self.embeddings[modality] = data['embeddings']
        
        # Store labels and protein IDs
        self.labels = label_data['labels']
        self.prot_ids = label_data['prot_ids']
    
    #__len__ function returns the number of samples in the dataset
    def __len__(self):
        return len(self.prot_ids)
    
    #__getitem__ function returns the embeddings, label, and protein ID for a given index
    def __getitem__(self, idx):
        embeddings = [self.embeddings[modality][idx] for modality in self.modalities]
        label = self.labels[idx]
        prot_id = self.prot_ids[idx]
        return embeddings, label, prot_id