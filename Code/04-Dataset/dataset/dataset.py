#PyTorch imports
#Path and typing imports
from pathlib import Path

import torch
from torch.utils.data import Dataset

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
    - embedding_dirs (List[str]): List of directories containing the embeddings for each modality.
    - modalities (List[str]): List of modality names to be loaded.
    - labels_path (str): Path to the file containing the labels.
    
    Raises:
    - ValueError: If the protein IDs for any modality do not match the protein IDs in the labels file or if the order of protein IDs differs between modalities.
    - ValueError: If the number of embedding directories does not match the number of modalities.
    
    Returns:
    - None
    '''
    def __init__(self, embedding_dirs: list[str], modalities: list[str], labels_path: str):
        # Initialize the dataset by loading embeddings and labels
        self.embedding_dirs = [Path(dir) for dir in embedding_dirs]
        self.modalities = modalities
        
        if len(self.embedding_dirs) != len(self.modalities):
            raise ValueError("Number of embedding directories must match number of modalities.")
        
        # Load labels
        label_data = torch.load(labels_path, weights_only=True)
        self.prot_ids = label_data['prot_ids']
        label_prot_ids = set(label_data['prot_ids'])
        
        # Load embeddings per modality
        self.embeddings: dict[str, torch.Tensor] = {}
        
        # Iterate through each modality to load embeddings and validate alignment with labels
        for i, directory in enumerate(self.embedding_dirs):
            modality = self.modalities[i]
            # Load embeddings for the modality
            data = torch.load(directory, weights_only=True)
            modality_prot_ids = data['prot_ids']
            
            #Check for duplicates
            if len(modality_prot_ids) != len(set(modality_prot_ids)):
                seen = set()
                dupes = {pid for pid in modality_prot_ids if pid in seen or seen.add(pid)}
                raise ValueError(f"Duplicate protein IDs found in modality {modality}: {dupes}")
            
            # Validate modaity and label proteins are identical
            modality_id_set = set(modality_prot_ids)
            missing = label_prot_ids - modality_id_set
            if missing:
                raise ValueError(
                    f"Modality {modality} misaligned with labels. "
                    f"Missing: {missing}"
                )
            
            # Align the embeddings with the order of protein IDs in the labels file
            id_to_idx = {pid: idx for idx, pid in enumerate(modality_prot_ids)}
            select_idx = torch.tensor([id_to_idx[pid] for pid in self.prot_ids], dtype=torch.long)
            
            # Store the embeddings for the modality
            self.embeddings[modality] = data['embeddings'][select_idx]
        
        # Store labels
        self.labels = label_data['labels']
        
        self.ID = f"ProteinDataset_{'_'.join(self.modalities)}"
    
    #__len__ function returns the number of samples in the dataset
    def __len__(self):
        return len(self.prot_ids)
    
    #__getitem__ function returns the embeddings, label, and protein ID for a given index
    def __getitem__(self, idx):
        embeddings = [self.embeddings[modality][idx] for modality in self.modalities]
        label = self.labels[idx]
        prot_id = self.prot_ids[idx]
        return embeddings, label, prot_id