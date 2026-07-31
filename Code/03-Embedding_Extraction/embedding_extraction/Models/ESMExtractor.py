import logging
import os

from configs import ExtractorConfig
from dotenv import load_dotenv
from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig
from esm.sdk.forge import ESM3ForgeInferenceClient
from tenacity import RetryError
from torch import Tensor

from embedding_extraction import EmbeddingExtractor

load_dotenv("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Environment/protein-prediction.env")
logger = logging.getLogger(__name__)

class ESMExtractor(EmbeddingExtractor):
    """
    Class - ESMExtractor:
    
    Extracts protein embeddings using the ESM-C model from Meta's ESM (Evolutionary Scale Modeling) family of protein language models. 
    This class implements the abstract methods defined in the EmbeddingExtractor base class to provide functionality specific to the ESM-C model for embedding extraction.
    
    Requires a Forge API token from https://forge.evolutionaryscale.ai which is free for academic use.

    Attributes:
    - tokens: A list of Forge API tokens used for authentication when making requests to the Forge API. When one token is exhausted (e.g. out of credits), the extractor automatically rotates to the next one.
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
                tokens: list[str] | None = None
                ):

        self.save_dir = config.save_dir

        self.model_name = config.model

        self._embedding_dim = None  # Initialize embedding dimension to None, will be set after first extraction

        #Initalise the base class
        super().__init__()

        #We support multiple Forge API tokens so that we can rotate onto a fresh one when the current one runs out of credits, rather than the whole extraction run failing. Tokens are defined in environment level variables so as to avoid hardcoding them in our codebase.
        self._tokens = tokens if tokens is not None else self._load_tokens_from_env()
        if not self._tokens:
            raise RuntimeError(
                "No Forge API tokens found. Set ESM_tokens (comma-separated) or ESM_token in the environment."
            )
        self._token_idx = 0

        #Our client which enables us to interact with the Forge API, initialised with the first available token.
        self._client = self._build_client(self._tokens[self._token_idx])

        logger.info(
            f"Initialized ESMExtractor with model {self.model_name} on device {self.device} "
            f"({len(self._tokens)} Forge API token(s) available)"
        )

    @staticmethod
    def _load_tokens_from_env() -> list[str]:
        """
        Method - _load_tokens_from_env:

        Loads the pool of Forge API tokens to rotate through from environment variables, preferring the comma-separated ESM_tokens variable and falling back to the single legacy ESM_token variable.

        Args:
        - None

        Returns:
        - A list of token strings (possibly empty if none are set).
        """

        tokens_env = os.getenv("ESM_tokens")
        if tokens_env:
            return [token.strip() for token in tokens_env.split(",") if token.strip()]

        legacy_token = os.getenv("ESM_token")
        return [legacy_token] if legacy_token else []

    def _build_client(self, token: str) -> ESM3ForgeInferenceClient:
        return ESM3ForgeInferenceClient(
            model = self.model_name,
            url = self.FORGE_URL,
            token = token
        )

    def _rotate_token(self) -> bool:
        """
        Method - _rotate_token:

        Switches the client over to the next token in the pool.

        Args:
        - None

        Returns:
        - True if a fresh token was available and the client was rebuilt with it, False if the pool is exhausted.
        """

        if self._token_idx + 1 >= len(self._tokens):
            return False

        self._token_idx += 1
        self._client = self._build_client(self._tokens[self._token_idx])

        logger.warning(
            "Rotating to Forge API token %d/%d after the previous one stopped working.",
            self._token_idx + 1,
            len(self._tokens)
        )

        return True

    @staticmethod
    def _unwrap_result(call, *args, **kwargs):
        """
        Method - _unwrap_result:

        Calls a Forge SDK client method (e.g. self._client.encode) and returns its result, unwrapping the underlying ESMProteinError even in the case where the SDK's own internal backoff retries (for 429/502/504) were exhausted, since in that case it raises a tenacity.RetryError rather than returning the error value directly.

        Args:
        - call: The bound client method to invoke.
        - *args, **kwargs: Arguments to pass to that method.

        Returns:
        - Whatever the call returns on success, or the underlying ESMProteinError once the SDK has given up retrying.
        """

        try:
            return call(*args, **kwargs)
        except RetryError as retry_error:
            return retry_error.last_attempt.result()

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
        - df: A pandas DataFrame containing protein data, including a "Uni_Prot_ID" column.

        Returns:
        - A filtered pandas DataFrame containing only the rows whose current UniProt sequence length is <= 2048 amino acids.
        """

        #Sequence length is fetched live here rather than trusting the "length" column cached at UniProt-mapping time, since UniProt records can change between when that cache was built and when extraction runs.
        lengths = df["Uni_Prot_ID"].apply(lambda uniprot_id: len(self.fetch_sequence(uniprot_id)))

        return df[(lengths <= 2048) & (lengths > 0)]
        
        
    
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
        - RuntimeError: If the Forge API returns an error during embedding retrieval that isn't resolved by rotating through the available tokens (specifically an ESMProteinError).
        """

        #First we create an ESMProtein object using the input sequence
        protein = ESMProtein(sequence=sequence)

        #We retry the encode+logits round trip on the current client. The SDK already retries 429/502/504 internally with its own backoff, so a failure reaching us here means that backoff didn't help - we rotate to the next available token and retry, and only raise once every token has been exhausted.
        while True:
            #We then tokenise the sequence using the Forge API client
            protein_tensor = self._unwrap_result(self._client.encode, protein)

            #Check if an error has been returned by the Forge API during tokenisation, if so we rotate to the next token and retry, or raise a RuntimeError with details about the error and the associated label if no tokens remain.
            if isinstance(protein_tensor, ESMProteinError):
                if self._rotate_token():
                    continue
                raise RuntimeError(
                    f"Forge API returned an error during tokenisation for label {label} "
                    f"and no further tokens are available to rotate to: {protein_tensor}"
                )

            #We then call the logits method of the Forge API client, specifying we want to return the embeddings for the sequence. The output contains both the logits and the embeddings, but we are only interested in the embeddings for our use case.
            output = self._unwrap_result(
                self._client.logits,
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True)
            )

            #Check again for an error
            if isinstance(output, ESMProteinError):
                if self._rotate_token():
                    continue
                raise RuntimeError(
                    f"Forge API returned an error for label {label} during logits retrieval "
                    f"and no further tokens are available to rotate to: {output}"
                )

            break

        #The Forge API returns embeddings shaped (batch=1, sequence_length, dim). We drop the batch dim, then trim the start/end tokens along the sequence axis and mean-pool over it to get a single embedding vector for the entire protein sequence. This pooled embedding is then returned on the CPU.
        embeddings = output.embeddings[0, 1:-1, :]
        pooled = embeddings.mean(dim=0)
        
        if self._embedding_dim is None:
            self._embedding_dim = pooled.shape[0]
            logger.info(
                "Set embedding dimension to %d based on first extracted embedding for label %s",
                self._embedding_dim,
                label
            )
        
        return pooled.cpu()