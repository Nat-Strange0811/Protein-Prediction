import logging
import os

import h5py
import torch
from configs import ExtractorConfig
from dscript.alphabets import Uniprot21
from dscript.pretrained import get_pretrained
from torch import Tensor

from embedding_extraction import EmbeddingExtractor

logger = logging.getLogger(__name__)

class DScriptExtractor(EmbeddingExtractor):
    """
    Class - DScriptExtractor:
    
    Extracts protein embeddings using the D-Script model, this is a PPI (protein-protein interaction) model that takes two protein sequences
    as input and outputs a prediction for interaction. We construct the embeddings by using a panel of proteins to indicate the general 
    interaction profile of the protein of interest.
    """
    
    def __init__(self, config: ExtractorConfig, mode: str = "extract"):
        
        self.model_name = config.model
        self.panel = config.panel

        self.save_dir = config.save_dir
        self.panel_embeddings_file = f'/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Embeddings/Panels/{self.panel}_{self.model_name}.h5'
        
        self.model = get_pretrained(version=self.model_name)
        self.model.eval()


        #Initalise the base class
        super().__init__()
        
        self.lm_model = get_pretrained("lm_v1")
        if self.device == "cuda":
            self.lm_model = self.lm_model.cuda()
        self.lm_model.eval()
               
        if mode == "extract":
            #We check for the fidelity of the panel embeddings in this function
            self.extract_panel_embeddings()
            
            with h5py.File(self.panel_embeddings_file, "r") as h5f:
                self.panel_embeddings = [
                    torch.from_numpy(h5f[uniprot_id][:]).unsqueeze(0).to(self.device)
                    for uniprot_id in sorted(h5f.keys())
                ]

            self._embedding_dim = len(self.panel_embeddings)
            self.model.to(self.device)
            
        self.ID = f"D_Script_{self.model_name}_panel_{self.panel}"

        
    @property
    def embedding_dim(self) -> int:
        """
        Method - embedding_dim:
        
        Returns the dimensionality of the embeddings produced by the specified DScript model.
        
        Args:
        - None
        
        Returns:
        - An integer representing the dimensionality of the embeddings produced by the specified DScript model
        """
        
        return self._embedding_dim
    
    
    def extract(self, uniprot_id: str) -> Tensor:
        """
        Method - extract:
        
        Extracts embeddings for a given UniProt ID using the D_Script model, each protein is run against a specified panel to model
        interactivity.
        
        Args:
        - uniprot_id: A string representing the UniProt ID of the protein for which to extract embeddings.
        
        Returns:
        - A Tensor containing the extracted embeddings for the specified UniProt ID (returning on cpu is convention as it allows for easier interoperability with other libraries and frameworks that may not support GPU acceleration).
        
        Raises:
        
        """
        
        #First, we fetch the amino acid sequence from UniProt
        sequence = self.fetch_sequence(uniprot_id)
        
        logger.debug(
            "D_script embedding | id = %s | panel = %d",
            uniprot_id,
            self.panel
        )
        
        #We then call the method to embed the sequence
        return self._embed_sequence(sequence, label=uniprot_id)
        
    def filter(self, df):
        """
        Method - filter:

        Filters a DataFrame of protein sequences to retain only those that are compatible with the D_Script model.

        Args:
        - df: A pandas DataFrame containing protein data, including a "Uni_Prot_ID" column.

        Returns:
        - Same DataFrame as passed in, as D_Script does not have any restrictions.
        """

        #Sequence length is fetched live here rather than trusting the "length" column cached at UniProt-mapping time, since UniProt records can change between when that cache was built and when extraction runs.
        lengths = df["Uni_Prot_ID"].apply(lambda uniprot_id: len(self.fetch_sequence(uniprot_id)))

        return df[(lengths <= 2000) & (lengths > 0)]
    
    def _lm_embed(self, sequence: str) -> Tensor:
        """
        Method - _lm_embed:

        Embeds a given amino acid sequence using the language model (LM) component of the D_Script model.

        Args:
        - sequence: A string representing the amino acid sequence to embed.

        Returns:
        - A Tensor containing the embedded representation of the sequence.
        """
        
        alphabet = Uniprot21()
        x = torch.from_numpy(alphabet.encode(sequence.encode("utf-8"))).long().unsqueeze(0)
        
        if self.device == "cuda":
            x = x.cuda()
        with torch.no_grad():
            return self.lm_model.transform(x).cpu()
    
    def _embed_sequence(self, sequence: str, label: str) -> Tensor:
        """
        Method - _embed_sequence:

        Embeds a given amino acid sequence using the D_Script model. Scoring against a panel of proteins to generate the interaction probabilities for the protein of interest.

        Args:
        - sequence: A string representing the amino acid sequence to embed.
        - label: A string representing the label for the sequence.

        Returns:
        - A Tensor containing the embedded representation of the sequence.
        """
        
        #lm_embed always returns a CPU tensor internally regardless of use_cuda, so it must be moved to self.device explicitly to match the model and panel_embeddings.
        sequence_embed = self._lm_embed(sequence).to(self.device)

        with torch.no_grad():
            scores = [self.model(sequence_embed, panel_member) for panel_member in self.panel_embeddings]
            
        return torch.stack(scores).cpu()
            
    def extract_panel_embeddings(self):
        """
        Method - extract_panel_embeddings:
        
        Extracts embeddings for a predefined panel of proteins using the D_Script model and saves them to disk.
        
        Args:
        - None
        
        Returns:
        - None (embeddings are saved to disk)
        
        Raises:
        - RuntimeError: If the panel number is not valid (e.g., if it does not correspond to a predefined set of panels).
        """
        
        panel_path = f"/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Panels/panel_{self.panel}.txt"
        
        if not os.path.exists(panel_path):
            raise RuntimeError(
                "Panel not found, please submit a valid panel to the extractor"
            )
            
        if os.path.exists(self.panel_embeddings_file):
            panel_length = sum(1 for _ in open(panel_path))
            try:
                with h5py.File(self.panel_embeddings_file, "r") as h5f:
                    uniprots_in_file = set(h5f.keys())
            except OSError as e:
                print(f"Error reading panel embeddings file {self.panel_embeddings_file}: {e}. Re-extracting embeddings.")
                os.remove(self.panel_embeddings_file)
                uniprots_in_file = set()
                
            if len(uniprots_in_file) == panel_length:
                print(f"Panel embeddings for model {self.model_name} and panel {self.panel} already exist and match the expected length, skipping extraction.")
                return
            else:
                print(f"Panel embeddings for model {self.model_name} and panel {self.panel} already exist but do not match the expected length, re-extracting embeddings.")
        else:
            os.makedirs(os.path.dirname(self.panel_embeddings_file), exist_ok=True)
            uniprots_in_file = set()

        with h5py.File(self.panel_embeddings_file, "a") as h5f, open(panel_path, "r") as f:
            for line in f:
                uniprot_id = line.strip()
                
                if uniprot_id in uniprots_in_file:
                    continue

                try:
                    sequence = self.fetch_sequence(uniprot_id)
                    if not sequence:
                        raise RuntimeError(f"UniProt Cache has no sequence data for panel member {uniprot_id} - cannot build panel embeddings.")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to fetch sequence for UniProt ID: {uniprot_id}"
                    ) from e

                embedding = self._lm_embed(sequence)

                h5f.create_dataset(uniprot_id, data=embedding.squeeze(0).cpu().numpy())
                
        