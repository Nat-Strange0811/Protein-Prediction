from extract import extract_embeddings, extract_dimensions_from_existing_embeddings, save_labels
from configs import Config, ExtractorConfig, ClassifierConfig, FusionLayerConfig
from dataset import ProteinDataset
from fusion_layer import ConcatenationFusion
from mlp_classifier import SimpleMLP
from fusion_model import FusionModel
from trainer import Trainer

from dotenv import load_dotenv
from pathlib import Path

import yaml
import os
import argparse

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
    
    for modality, extractor_config_path in modalities.items():
        extractor_config = load_config(extractor_config_path, "extractor")
        embedding_dir = extractor_config.embedding_loc
        embedding_dirs.append(embedding_dir)
        
        if not os.path.exists(embedding_dir):
            embedding_dim.append(extract_embeddings(raw_csv, extractor_config))
        else:
            embedding_dim.append(extract_dimensions_from_existing_embeddings(embedding_dir))
    
    if not labels_path.exists():
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    args = parser.parse_args()
    config_path = args.config
    
    config = load_config(config_path, "config")
    
    dataset, embedding_dims = load_dataset(config)
    fusionLayer = load_fusion_layer(config)
    
    fusionLayer_output_dim = fusionLayer.output_dim()
    
    classifier = load_classifier(config, fusionLayer_output_dim)
    fusion_model = FusionModel(embedding_dims, config.model.d_model, fusionLayer, classifier)
    
    trainer = Trainer(fusion_model, dataset, config)
    trainer.train(config.training.epochs)
    trainer.evaluate()
    
    
    
    
    