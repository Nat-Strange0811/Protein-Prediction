from configs import Config, ExtractorConfig, ClassifierConfig, FusionLayerConfig
from embedding_extraction import ESMExtractor

from pathlib import Path

import torch
import pandas as pd


def extract_embeddings(data : str, extractor_config : ExtractorConfig):
    """
    Extract embeddings for the given data using the specified extractor configuration.

    Args:
        data (list): List of protein sequences or identifiers.
        extractor_config (ExtractorConfig): Configuration for the embedding extractor.
    """
    
    extractor_registry = {
        "esm": ESMExtractor
    }

    extractor_class = extractor_registry.get(extractor_config.extractor_type)
    if not extractor_class:
        raise ValueError(f"Unknown extractor: {extractor_config.extractor_type}")

    df = pd.read_csv(data, sep = ',')

    prot_ids = df["Uni_Prot_ID"].tolist()
    
    extractor = extractor_class(extractor_config)
    embeddings = extractor.extract_batch(prot_ids)
    embedding_dim = extractor.embedding_dim
    extractor.save(embeddings, prot_ids, extractor_config.embedding_loc)
    
    return embedding_dim
    
    
def extract_dimensions_from_existing_embeddings(embedding_dir : str):
    return torch.load(embedding_dir)['embeddings'].shape[1]

def save_labels(path : str, save_dir : str):
    """
    Extract protein IDs and labels from the raw data CSV and save them in the format expected by ProteinDataset.

    Args:
        path (str): Path to the raw CSV file containing "Uni_Prot_ID" and "label" columns.
        save_dir (str): Path to save the resulting labels file to.
    """

    df = pd.read_csv(path, sep = ',')

    prot_ids = df["Uni_Prot_ID"].tolist()
    labels = torch.tensor(df["label"].tolist(), dtype=torch.float32)

    save_dir = Path(save_dir)
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'labels': labels,
        'prot_ids': prot_ids
    }, save_dir)