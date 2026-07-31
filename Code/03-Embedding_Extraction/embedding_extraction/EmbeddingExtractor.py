#Logging is an import that allows us to log messages in our code. It is a built-in module in Python that provides a flexible framework for emitting log messages from Python programs. We can use it to track events that happen when some software runs, which can be helpful for debugging and monitoring the software's behavior.
import logging

#ABC or abstract base class is a module that allows us to define base classes that cannot be instantiated but define a structure for derived classes.
from abc import ABC, abstractmethod

#Path import for saving
from pathlib import Path

import pandas as pd

#Torch is a popular open-source machine learning library developed by Facebook's AI Research lab. It provides a flexible and efficient platform for building and training deep learning models. The 'Tensor' class in PyTorch is a multi-dimensional array that can be used to store and manipulate data for machine learning tasks. It is similar to NumPy arrays but with additional capabilities for GPU acceleration and automatic differentiation, making it a fundamental building block for deep learning models in PyTorch.
import torch
from torch import Tensor

#Module logger - inheriting classes share this unless otherwise defined.
logger = logging.getLogger(__name__)

class EmbeddingExtractor(ABC):
    '''
    Class - EmbeddingExtractor:

    Abstract base class for extracting protein embeddings. This class defines the interface for any embedding extractor implementation, ensuring that all derived classes implement the necessary methods for extracting embeddings and providing the embedding dimensionality. 
    
    Attributes:
    - device: A string representing the device to use for computations (e.g., 'cpu' or 'cuda'). If not specified, it defaults to 'cuda' if a GPU is available, otherwise it defaults to 'cpu'.
    - _session: A requests.Session object that is configured with retry logic to handle transient HTTP errors when making requests to retrieve embeddings.
    
    Methods:
    
        Abstract Methods:
        - extract(uniprot_id: str) -> Tensor: An abstract method that must be implemented by derived classes to extract embeddings for a given UniProt ID. It should return a Tensor containing the extracted embeddings for the specified UniProt ID.
        - embedding_dim() -> int: An abstract property that must be implemented by derived classes to return the dimensionality of the extracted embeddings.
        
        Concrete Methods:
        - extract_batch(uniprot_ids: list[str]) -> Tensor: A concrete method that takes a list of UniProt IDs and returns a Tensor containing the extracted embeddings for all specified UniProt IDs
        - fetch_sequence(uniprot_id: str) -> str: A helper method that retrieves the amino acid sequence for a given UniProt ID using the UniProt API. It returns the sequence as a string.
        
        Helper Methods:
        - _build_session(max_retries: int, backoff_factor: float) -> requests.Session: A helper method that creates and configures a requests.Session object with retry logic based on the specified maximum number of retries and backoff factor.
        - _get(url: str) -> requests.Response: A helper method that performs a GET request to the specified URL using the configured session and handles any HTTP errors that may occur during the request.
    '''
    
    
    def __init__(self):

        #Set the device to run on, gpu if available, otherwise cpu.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.uni_prot_cache = pd.read_parquet(
            "/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/NN_input/uni_prot_cache.parquet"
        ).set_index("Uni_Prot_ID")
        
    @abstractmethod
    def extract(self, uniprot_id: str):
        """
        Method - Extract:
        
        Args:
        - uniprot_id: A string representing the UniProt ID of the protein for which we want to extract embeddings.
        
        Returns:
        - A Tensor containing the extracted embeddings for the specified UniProt ID (returning on cpu is convention as it allows for easier interoperability with other libraries and frameworks that may not support GPU acceleration).
        
        Raises:
        - requests.HTTPError: If the HTTP request to retrieve the embeddings fails (e.g., due to network issues, server errors, or invalid UniProt ID).
        """
        ...
        
    @abstractmethod
    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method - Filter:
        
        Args:
        - df: A pandas DataFrame containing protein data that needs to be filtered based on the extractor's compatibility.
        
        Returns:
        - A pandas DataFrame containing only the rows that are compatible with the extractor's requirements.
        """
    
    def save(self, embeddings: Tensor, uniprot_ids: list[str], output_path: str):
        """
        Method - Save:
        
        Args:
        - embeddings: A Tensor containing the extracted embeddings to be saved.
        - uniprot_ids: A list of strings representing the UniProt IDs corresponding to the embeddings.
        - output_path: A string representing the file path where the embeddings should be saved.
        
        Returns:
        - None
        
        Raises:
        - IOError: If there is an error while saving the embeddings to the specified output path.
        """


        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            existing = torch.load(output_path)
            all_ids = existing['prot_ids'] + uniprot_ids
            all_embeddings = torch.cat([existing['embeddings'], embeddings], dim=0)
        else:
            all_ids = uniprot_ids
            all_embeddings = embeddings

        torch.save({
            'embeddings': all_embeddings,
            'prot_ids': all_ids
        }, output_path)
        
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """
        Method - Embedding Dimension:
        
        Returns:
        - An integer representing the dimensionality of the extracted embeddings.
        """
        ...
    
        
        
    def extract_batch(self, uniprot_ids: list[str]) -> Tensor:
        """
        Method - Extract Batch:
        
        Args:
        - uniprot_ids: A list of strings representing the UniProt IDs of the proteins for which we want to extract embeddings.
        
        Returns:
        - A Tensor containing the extracted embeddings for the specified UniProt IDs (returning on cpu is convention as it allows for easier interoperability with other libraries and frameworks that may not support GPU acceleration).
        
        Raises:
        - requests.HTTPError: If the HTTP request to retrieve the embeddings fails (e.g., due to network issues, server errors, or invalid UniProt IDs).
        - KeyError: If any of the UniProt IDs are not found in a precomputed embedding store (e.g. PINNACLE).
        """
        ...
        
        #Extract embeddings for each UniProt ID in the list and stack them into a single Tensor. Cast to float32 here (rather than in each extractor) so every modality is stored at a consistent precision regardless of what dtype its underlying model natively returns (e.g. Forge returns bfloat16).
        embeddings = []
        for uniprot_id in uniprot_ids:
            embedding = self.extract(uniprot_id).float()
            self.save(embedding.unsqueeze(0), [uniprot_id], self.save_dir)
            embeddings.append(embedding)
        return torch.stack(embeddings, dim=0)
    
    def fetch_sequence(self, uniprot_id: str) -> str:
        """
        Method - Fetch Sequence:
        
        Args:
        - uniprot_id: A string representing the UniProt ID of the protein for which we want to fetch the amino acid sequence.
        
        Returns:
        - A string containing the amino acid sequence for the specified UniProt ID.
        """
        
        return self.uni_prot_cache.loc[uniprot_id, "sequence"]