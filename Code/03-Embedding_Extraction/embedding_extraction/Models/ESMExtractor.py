#Base class
from embedding_extraction import EmbeddingExtractor

from configs import ExtractorConfig

#Built-in python imports
import logging

#ESM imports
from esm.sdk.forge import ESM3ForgeInferenceClient
from esm.sdk.api import (
    ESMProtein,
    ESMProteinError,
    LogitsConfig
)

#Pytorch imports
import torch
from torch import Tensor

#env imports
from dotenv import load_dotenv
import os

load_dotenv("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Environment/protein-prediction.env")
logger = logging.getLogger(__name__)

class ESMExtractor(EmbeddingExtractor):
    """
    Class - ESMExtractor:
    
    Extracts protein embeddings using the ESM-C model from Meta's ESM (Evolutionary Scale Modeling) family of protein language models. 
    This class implements the abstract methods defined in the EmbeddingExtractor base class to provide functionality specific to the ESM-C model for embedding extraction.
    
    Requires a Forge API token from https://forge.evolutionaryscale.ai which is free for academic use.
    
    Attributes:
    - token: A string representing the Forge API token used for authentication when making requests to the Forge API to retrieve embeddings.
    - model_name: A string representing the specific ESM-C model variant to use for embedding
    - device: A string representing the device to use for computations (e.g., 'cpu' or 'cuda'). If not specified, it defaults to 'cuda' if a GPU is available, otherwise it defaults to 'cpu'.
    - max_retries: An integer representing the maximum number of retries for HTTP requests when retrieving embeddings from the Forge API. Default is 3.
    - backoff_factor: A float representing the backoff factor for retrying HTTP requests when retrieving embeddings from the Forge API. Default is 0.5.
    - client: An instance of the ESM3ForgeInferenceClient class from the ESM SDK, which is used to interact with the Forge API for embedding retrieval.
    
    Methods:
    - extract(uniprot_id: str) -> Tensor: Implements the abstract method from the base class to extract embeddings for a given UniProt ID using the ESM-C model via the Forge API.
    - embedding_dim() -> int: Implements the abstract property from the base class to return the dimensionality of the embeddings produced by the specified ESM-C model variant.
    - _embed_sequence
    """
    
    FORGE_URL = "https://biohub.ai"
    
    def __init__(
                self,
                config: ExtractorConfig,
                token: str = os.getenv("ESM_token")
                ):
        
        model = config.model
        
        device = config.device
        max_retries = config.max_retries
        backoff_factor = config.backoff_factor
        self.save_dir = config.embedding_loc
        
        self.model_name = model
        
        self._embedding_dim = None  # Initialize embedding dimension to None, will be set after first extraction
        
        #Initalise the base class
        super().__init__(device=device, max_retries=max_retries, backoff_factor=backoff_factor)
        
        #Our client which enables us to interact with the Forge API, we define our token in environment level variables so as to avoid hardcoding it in our codebase
        self._client = ESM3ForgeInferenceClient(
            model = model,
            url = self.FORGE_URL,
            token = token
        )
        
        logger.info(f"Initialized ESMExtractor with model {model} on device {self.device}")
        
    @property
    def embedding_dim(self) -> int:
        """
        Method - embedding_dim:
        
        Returns the dimensionality of the embeddings produced by the specified ESM-C model variant.
        
        Args:
        - None
        
        Returns:
        - An integer representing the dimensionality of the embeddings produced by the specified ESM-C model
        
        Raises:
        - RuntimeError: If the embedding dimension is not set (e.g., if extract has not been called on at least one protein yet).
        """
        
        #Depending on the exact version of the model we utilise, slightly different embedding dimensions are produced, we set this dynamically based on the first extracted embedding to ensure flexibility.
        if self._embedding_dim is None:
            raise RuntimeError(
                "Embedding dimension is not set - call extract on at least one protein first."
            )
        return self._embedding_dim
        
    def extract(self, uniprot_id: str) -> Tensor:
        """
        Method - extract:
        
        Extracts embeddings for a given UniProt ID using the ESM-C model via the Forge API.
        
        Args:
        - uniprot_id: A string representing the UniProt ID of the protein for which to extract embeddings.
        
        Returns:
        - A Tensor containing the extracted embeddings for the specified UniProt ID (returning on cpu is convention as it allows for easier interoperability with other libraries and frameworks that may not support GPU acceleration).
        
        Raises:
        -requests.HTTPError: If the HTTP request to retrieve the embeddings fails (e.g., due to network issues, server errors, or invalid UniProt ID).
        -ValueError: If the input sequence exceeds the maximum length supported by the ESM-C model (e.g., 2048 amino acids), as longer sequences may not be processed correctly by the model.
        -RuntimeError: If forge API returns an error
        """
        
        #First, we fetch the amino acid sequence from UniProt
        sequence = self.fetch_sequence(uniprot_id)
        
        #ESM can only handle proteins <= 2048 amino acids in length, at the moment anything longer is excluded from our dataset
        if len(sequence) > 2048:
            raise ValueError(
                f"Input sequence length {len(sequence)} exceeds maximum supported length of 2048 amino acids for ESM-C model."
            )
        
        logger.debug(
            "Extracting ESM C embedding | id = %s | sequence length = %d",
            uniprot_id,
            len(sequence)
        )
        
        #We then call the method to embed the sequence using the Forge API, passing the UniProt ID as a label for logging purposes.
        return self._embed_sequence(sequence, label=uniprot_id)
    
    def filter(self, df):
        """
        Method - filter:
        
        Filters a DataFrame of protein sequences to retain only those that are compatible with the ESM-C model for embedding extraction.
        
        Args:
        - df: A pandas DataFrame containing protein sequences and associated metadata, including a "sequence" column with amino acid sequences.
        
        Returns:
        - A filtered pandas DataFrame containing only the rows with sequences that are compatible with the ESM-C model (i.e., sequences of length <= 2048 amino acids).
        """
        
        df = df[df["length"] <= 2048]
        
        return df
        
        
    
    def _embed_sequence(self, sequence: str, label: str) -> Tensor:
        """
        Method - _embed_sequence:
        
        A helper method that takes an amino acid sequence and a label (e.g., UniProt ID) and retrieves the corresponding embeddings from the Forge API using the ESM-C model.
        
        Args:
        - sequence: A string representing the amino acid sequence of the protein for which to extract embeddings.
        - label: A string representing a label (e.g., UniProt ID) to associate with the extracted embeddings for logging purposes.
        
        Returns:
        - A Tensor containing the extracted embeddings for the specified sequence (returning on cpu is convention as it allows for easier interoperability with other libraries and frameworks that may not support GPU acceleration).
        
        Raises:
        - RuntimeError: If forge API returns an error during embedding retrieval, specifically an ESMProteinError
        """
        
        #First we create an ESMProtein object using the input sequence
        protein = ESMProtein(sequence=sequence)
        #We then tokenise the sequence using the Forge API client
        protein_tensor = self._client.encode(protein)
        
        #Check if an error has been returned by the Forge API during tokenisation, if so we raise a RuntimeError with details about the error and the associated label for easier debugging.
        if isinstance(protein_tensor, ESMProteinError):
            raise RuntimeError(
                f"Forge API returned an error during tokenisation for label {label}: {protein_tensor}"
            )
            
        #We then call the logits method of the Forge API client, specifying we want to return the embeddings for the sequence. The output contains both the logits and the embeddings, but we are only interested in the embeddings for our use case.
        output = self._client.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True)
        )
        
        #Check again for an error
        if isinstance(output, ESMProteinError):
            raise RuntimeError(
                f"Forge API returned an error for label {label} during logits retrieval: {output}"
            )
        
        #The ouput embeddings include special start and end tokens, we remove these and then take the mean across the sequence length dimension to get a single embedding vector for the entire protein sequence. This pooled embedding is then returned on the CPU.
        embeddings = output.embeddings[1:-1, :]
        pooled = embeddings.mean(dim=0)
        
        if self._embedding_dim is None:
            self._embedding_dim = pooled.shape[0]
            logger.info(
                "Set embedding dimension to %d based on first extracted embedding for label %s",
                self._embedding_dim,
                label
            )
        
        return pooled.cpu()