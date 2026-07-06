from pydantic import BaseModel
from typing import Dict, List

class ModelConfig(BaseModel):
    d_model: int
    modalities: Dict[str, str]
    fusion: str
    mlp: str

class PathsConfig(BaseModel):
    embedding_dir: str
    labels_path: str
    checkpoint_dir: str
    raw_csv: str
    
class TrainingConfig(BaseModel):
    batch_size: int
    epochs: int
    patience: int
    
class Config(BaseModel):
    model: ModelConfig
    paths: PathsConfig
    training: TrainingConfig
    device: str
    
class ExtractorConfig(BaseModel):
    extractor_type: str
    model: str
    device: str
    max_retries: int
    backoff_factor: float
    embedding_loc: str
    
class ClassifierConfig(BaseModel):
    model: str
    dropout_rate: float
    activation_function: str
    hidden_dims: List[int]
    device: str
    
class FusionLayerConfig(BaseModel):
    model: str
    device: str
    