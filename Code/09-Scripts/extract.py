import os
from pathlib import Path

import pandas as pd
import torch
from configs import ExtractorConfig
from embedding_extraction import ESMExtractor, DScriptExtractor


def cached_prot_ids_current(cache_path : str, raw_csv : str, exact: bool = False) -> bool:
    """
    Check whether a cached embeddings/labels file (anything saved with a 'prot_ids' key)
    still matches the current raw_csv's Uni_Prot_ID set.

    load_dataset() only regenerates a cache if the file is missing, not if raw_csv's
    contents changed underneath it (e.g. after a data_prep.py threshold tweak) - so a
    stale cache would otherwise be reused silently, or cause a ProteinDataset alignment
    crash further downstream instead of at the source.
    """
    if not os.path.exists(cache_path):
        return False

    current_ids = set(pd.read_csv(raw_csv)["Uni_Prot_ID"])
    cached_ids = set(torch.load(cache_path, weights_only=True)["prot_ids"])
    if exact:
        return current_ids == cached_ids
    return current_ids.issubset(cached_ids)


def extract_embeddings(data : str, extractor_config : ExtractorConfig):
    """
    Extract embeddings for the given data using the specified extractor configuration.

    Args:
        data (list): List of protein sequences or identifiers.
        extractor_config (ExtractorConfig): Configuration for the embedding extractor.
    """
    
    extractor_registry = {
        "esm": ESMExtractor,
        "dscript": DScriptExtractor
    }

    extracted_protids = set(torch.load(extractor_config.save_dir, weights_only=True)["prot_ids"]) if os.path.exists(extractor_config.save_dir) else set()
    
    extractor_class = extractor_registry.get(extractor_config.extractor_type)
    if not extractor_class:
        raise ValueError(f"Unknown extractor: {extractor_config.extractor_type}")

    df = pd.read_csv(data, sep = ',')

    prot_ids = df["Uni_Prot_ID"].tolist()
    prot_ids_to_extract = [prot_id for prot_id in prot_ids if prot_id not in extracted_protids]
    
    extractor = extractor_class(extractor_config)
    extractor.extract_batch(prot_ids_to_extract)
    embedding_dim = extractor.embedding_dim
    
    return embedding_dim
    
    
def extract_dimensions_from_existing_embeddings(embedding_dir : str) -> int:
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