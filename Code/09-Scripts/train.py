import argparse
import os
import random
from pathlib import Path

import numpy as np
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

    if not cached_prot_ids_current(str(labels_path), raw_csv, exact=True):
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
    
    classifier = classifier_registry.get(classifier_config.model)(input_dim, config.training.dropout_rate, classifier_config)
    return classifier

def run_training(config, dataset, embedding_dims, evaluate=False):
    """
    Run the training process for the model based on the provided configuration and dataset.

    Args:
        config (dict): Configuration parameters.
        dataset (ProteinDataset): The dataset to use for training and evaluation.
        embedding_dims (list): List of embedding dimensions for each modality.
    
    Returns:
        dict: A dictionary containing the results of the training process.
    """
    
    fusionLayer = load_fusion_layer(config)
    fusionLayer_output_dim = fusionLayer.output_dim()
    
    classifier = load_classifier(config, fusionLayer_output_dim)
    
    fusion_model = FusionModel(embedding_dims, config.model.d_model, fusionLayer, classifier)
    
    trainer = Trainer(fusion_model, dataset, config)
    trainer.train(config.training.epochs)
    
    if evaluate:
        trainer.evaluate()
        
    return trainer

def parameter_sweep(config, dataset, embedding_dims):
    """
    Perform a parameter sweep to find the best configuration for the model.

    Args:
        config (dict): Configuration parameters.
        dataset (ProteinDataset): The dataset to use for training and evaluation.
        embedding_dims (list): List of embedding dimensions for each modality.
    
    Returns:
        dict: The best configuration found during the parameter sweep.
        pd.DataFrame: A DataFrame containing the results of the parameter sweep.
    """
    
    rows = []
    best_results = None
    best_config = None
    best_trainer = None
        
    learning_rate_range = config.training.learning_rate_range
    weight_decay_range = config.training.weight_decay_range
    dropout_rate_range = config.training.dropout_rate_range
    
    rng = np.random.default_rng(config.training.seed)
    
    learning_rates = rng.uniform(np.log(learning_rate_range[0]), np.log(learning_rate_range[1]), 5)
    weight_decays = rng.uniform(np.log(weight_decay_range[0]), np.log(weight_decay_range[1]), 5)
    dropout_rates = rng.uniform(dropout_rate_range[0], dropout_rate_range[1], 5)
    
    learning_rates = np.exp(learning_rates)
    weight_decays = np.exp(weight_decays)
    
    for lr in learning_rates:
        for wd in weight_decays:
            for dr in dropout_rates:
                config_copy = config.model_copy(deep=True)
                
                config_copy.training.learning_rate = lr
                config_copy.training.weight_decay = wd
                config_copy.training.dropout_rate = dr
                
                print(f"Running training with learning_rate={lr}, weight_decay={wd}, dropout_rate={dr}\n")
                
                trainer = run_training(config_copy, dataset, embedding_dims)
                run_results = trainer.results

                run_results_list = [lr, wd, dr, run_results["best_acc"], run_results["best_auc"], run_results["best_epoch"], run_results["train_auc"]]
                rows.append(run_results_list)
                
                if best_results is None or run_results["best_auc"] > best_results["best_auc"]:
                    best_results = run_results
                    best_config = config_copy
                    best_trainer = trainer

    results = pd.DataFrame(rows, columns=["learning_rate", "weight_decay", "dropout_rate", "accuracy", "auc", "epoch", "train_auc"])

    return best_config, results, best_trainer

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
    
    print(f"---------------Running parameter sweep---------------\n")
    best_config, results, best_trainer = parameter_sweep(config, dataset, embedding_dims)
    
    results_path = Path(config.paths.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    
    seed_1 = random.randint(0, 10000)
    seed_2 = random.randint(0, 10000)
    seed_3 = random.randint(0, 10000)
    
    config_seed_1 = best_config.model_copy(deep=True)
    config_seed_1.training.seed = seed_1
    
    config_seed_2 = best_config.model_copy(deep=True)
    config_seed_2.training.seed = seed_2
    
    config_seed_3 = best_config.model_copy(deep=True)
    config_seed_3.training.seed = seed_3
    
    for i, config_seed in enumerate([config_seed_1, config_seed_2, config_seed_3], start=1):
        print(f"---------------Running training for seed {config_seed.training.seed}---------------\n")
        run_training(config_seed, dataset, embedding_dims, evaluate=True)
        
    for file in Path(config.paths.checkpoint_dir).glob("best_model_*.pt"):
        if file != Path(config.paths.checkpoint_dir) / f'best_model_{best_trainer.ID}.pt':
            os.remove(file)
    
if __name__ == "__main__":
    main()
    
    
    
    