from pydantic import BaseModel


class ModelConfig(BaseModel):
    d_model: int
    modalities: dict[str, str]
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

class ExtractorConfig(BaseModel):
    extractor_type: str
    model: str
    max_retries: int
    backoff_factor: float
    panel: int = 0

    @property
    def save_dir(self) -> str:
        base = "/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Embeddings"
        if self.extractor_type == "dscript":
            return f"{base}/{self.panel}_{self.model}.pt"
        return f"{base}/{self.model}.pt"

class ClassifierConfig(BaseModel):
    model: str
    dropout_rate: float
    activation_function: str
    hidden_dims: list[int]

class FusionLayerConfig(BaseModel):
    model: str
