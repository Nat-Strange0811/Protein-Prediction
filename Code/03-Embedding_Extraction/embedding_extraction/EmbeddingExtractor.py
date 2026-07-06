#Logging is an import that allows us to log messages in our code. It is a built-in module in Python that provides a flexible framework for emitting log messages from Python programs. We can use it to track events that happen when some software runs, which can be helpful for debugging and monitoring the software's behavior.
import logging
#ABC or abstract base class is a module that allows us to define base classes that cannot be instantiated but define a structure for derived classes.
from abc import ABC, abstractmethod

#Path import for saving
from pathlib import Path

#Requests is a library that allows us to send HTTP requests in Python. It provides a simple and elegant way to interact with web services and APIs. We can use it to make GET, POST, PUT, DELETE, and other types of HTTP requests, and it also supports features like authentication, sessions, and retries.
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
    
    #UNIPROT_FASTA_URL for amino acid fetching
    UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    
    
    def __init__(self,
                 device: str | None = None,
                 max_retries: int = 3,
                 backoff_factor: float = 0.5
                 ):
        
        #Set the device to run on, gpu if available, otherwise cpu.
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self._session = self._build_session(max_retries, backoff_factor)
        
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
        - KeyError: If the UniProt ID is not found in a precomputed embedding store (e.g. PINNACLE).
        """
        ...
    
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

        torch.save({
            'embeddings': embeddings,
            'prot_ids': uniprot_ids
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
        
        #Extract embeddings for each UniProt ID in the list and stack them into a single Tensor.
        embeddings = [self.extract(uniprot_id) for uniprot_id in uniprot_ids]
        return torch.stack(embeddings, dim=0)
    
    def fetch_sequence(self, uniprot_id: str) -> str:
        """
        Method - Fetch Sequence:
        
        Args:
        - uniprot_id: A string representing the UniProt ID of the protein for which we want to fetch the amino acid sequence.
        
        Returns:
        - A string containing the amino acid sequence for the specified UniProt ID.
        
        Raises:
        - requests.HTTPError: If the HTTP request to retrieve the sequence fails (e.g., due to network issues, server errors, or invalid UniProt ID).
        """
        
        url = self.UNIPROT_FASTA_URL.format(uniprot_id=uniprot_id)
        response = self._get(url)
        
        #Fasta format is >header\nseqeuence, so we split on newline and take the second part as the sequence.
        lines = response.text.strip().split('\n')
        sequence = ''.join(lines[1:])  # Join all lines after the header to get the full sequence
        
        logger.debug(
            "Fetched sequence for UniProt ID %s: %d", uniprot_id, len(sequence)
        )
        
        return sequence
    
    def _build_session(self, max_retries: int, backoff_factor: float) -> requests.Session:
        """
        Method - Build Session:
        
        Args:
        - max_retries: An integer representing the maximum number of retries for failed HTTP requests.
        - backoff_factor: A float representing the backoff factor for retrying failed HTTP requests (e.g., 0.5 means that the delay between retries will be 0.5 seconds, then 1 second, then 2 seconds, etc.).
        
        Returns:
        - A configured requests.Session object with retry logic.
        """
        
        #The retry strategy utilises the library import Retry from urllib3.util.retry to build our http session with retry logic. We specify the total number of retries, the backoff factor, 
        #and the HTTP codes to retry on. 
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False
        )
        
        #The HTTPAdapter is then used to mount the retry strategy to both http and https requests in the session.
        adapter = HTTPAdapter(max_retries=retry_strategy)
        
        #We then create a requests.Session and mount the adapter to both http and https requests, ensuring that all requests made through this session will have the retry logic applied.
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        return session
    
    def _get(self, url: str) -> requests.Response:
        """
        Method - Get:
        
        Args:
        - url: A string representing the URL to which the GET request should be made.
        
        Returns:
        - A requests.Response object containing the response from the GET request.
        
        Raises:
        - requests.HTTPError: If the HTTP request fails (e.g., due to network issues, server errors, or invalid URL).
        """
        
        #We log the URL being requested at the debug level, then we perform the GET request using the configured session. If the response is not successful (i.e., response.ok is False), we raise an 
        #HTTPError with details about the failure.
        logger.debug("GET %s", url)
        response = self._session.get(url, timeout = 30)
        
        if not response.ok:
            raise requests.HTTPError(
                f"Request failed [{response.status_code}] for URL: {url}",
                response=response
            )
            
        return response