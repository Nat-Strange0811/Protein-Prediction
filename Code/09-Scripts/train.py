import argparse
from pathlib import Path

import pandas as pd
import yaml
from configs import ClassifierConfig, Config, ExtractorConfig, FusionLayerConfig
from data_prep import prepare_data
from dataset import ProteinDataset
from dotenv import load_dotenv
from extract import (
    cached_prot_ids_current,
    extract_dimensions_from_existing_embeddings,
    extract_embeddings,
    save_labels,
)
from fusion_layer import ConcatenationFusion
from fusion_model import FusionModel
from mlp_classifier import SimpleMLP
from trainer import Trainer

load_dotenv("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Environment/protein-prediction.env")

def load_config(config_path, config_type):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.
        config_type (str): Type of configuration to load ('extractor', 'classifier', 'fusion_layer', or 'config').
    
    Returns:
        dict: Configuration parameters as a dictionary.
    """
    
    config_registry = {
        "extractor": ExtractorConfig,
        "classifier": ClassifierConfig,
        "fusion_layer": FusionLayerConfig,
        "config": Config
    }
    
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    
    return config_registry[config_type](**config_data)


def load_dataset(config):
    """
    Load the ProteinDataset based on the provided configuration.

    Args:
        config (dict): Configuration parameters.
    
    Returns:
        ProteinDataset: An instance of the ProteinDataset.
    """
    
    modalities = config.model.modalities
    raw_csv = config.paths.raw_csv
    
    labels_path = Path(config.paths.labels_path)
    embedding_dim = []
    embedding_dirs = []
    
    for extractor_config_path in modalities.values():
        extractor_config = load_config(extractor_config_path, "extractor")
        embedding_dir = extractor_config.save_dir
        embedding_dirs.append(embedding_dir)
        
        if cached_prot_ids_current(embedding_dir, raw_csv):
            embedding_dim.append(extract_dimensions_from_existing_embeddings(embedding_dir))
        else:
            embedding_dim.append(extract_embeddings(raw_csv, extractor_config))

    if not cached_prot_ids_current(str(labels_path), raw_csv):
        save_labels(raw_csv, labels_path)
    
    dataset = ProteinDataset(embedding_dirs, list(modalities.keys()), labels_path)
    return dataset, embedding_dim

def load_fusion_layer(config):
    """
    Load the ConcatenationFusion layer based on the provided configuration.

    Args:
        config (dict): Configuration parameters.
    
    Returns:
        ConcatenationFusion: An instance of the ConcatenationFusion layer.
        
    Raises:
        ValueError: If required configuration parameters are missing.
    """
    
    fusion_registry ={
        "concatenation": ConcatenationFusion
    }
    
    d_model = config.model.d_model
    n_modalities = len(config.model.modalities)
    fusion_config = load_config(config.model.fusion, "fusion_layer")
    
    fusion_layer = fusion_registry.get(fusion_config.model)(d_model, n_modalities, fusion_config)
    return fusion_layer

def load_classifier(config, input_dim):
    """
    Load the SimpleMLP classifier based on the provided configuration.

    Args:
        config (dict): Configuration parameters.
        input_dim (int): The input dimension for the classifier.
    
    Returns:
        SimpleMLP: An instance of the SimpleMLP classifier.
    """
    
    classifier_registry = {
        "SimpleMLP": SimpleMLP
    }
    
    classifier_config = load_config(config.model.mlp, "classifier")
    
    classifier = classifier_registry.get(classifier_config.model)(input_dim, classifier_config)
    return classifier


def main():
    
    print("---------------Extracting arguments---------------\n")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    args = parser.parse_args()
    config_path = args.config
    
    print(f"---------------Loading configuration from {config_path}---------------\n")
    config = load_config(config_path, "config")
    
    print(f"---------------Running data preparation---------------\n")
    raw_csv_prefix = str(Path(config.paths.raw_csv).with_suffix(''))
    prepare_data(
        [load_config(path, "extractor") for path in config.model.modalities.values()],
        raw_csv_prefix,
        backoff_factor=config.uni_prot.backoff_factor,
        retries=config.uni_prot.retries,
    )
    
    print(f"---------------Loading dataset---------------\n")
    dataset, embedding_dims = load_dataset(config)
    print(f"---------------Loading fusion layer---------------")
    fusionLayer = load_fusion_layer(config)
    
    fusionLayer_output_dim = fusionLayer.output_dim()
    
    print(f"---------------Loading classifier---------------\n")
    classifier = load_classifier(config, fusionLayer_output_dim)
    print(f"---------------Loading fusion model---------------\n")
    fusion_model = FusionModel(embedding_dims, config.model.d_model, fusionLayer, classifier)
    
    print(f"---------------Initializing trainer---------------\n")
    trainer = Trainer(fusion_model, dataset, config)
    print(f"---------------Starting training---------------\n")
    trainer.train(config.training.epochs)
    print(f"---------------Evaluating best model on test set---------------\n")
    trainer.evaluate()
    
if __name__ == "__main__":
    main()
    
    
    
    