from dataclasses import dataclass
import yaml

@dataclass
class ModelConfig:
    base_model_path: str
    max_seq_length: int
    embedding_dim: int
    use_fp16: bool

@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    triplet_margin: float
    augmentation_factor: int
    subsample_fraction: float
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

@dataclass
class DataConfig:
    test_data_path: str
    output_dir: str

@dataclass
class InferenceConfig:
    batch_size: int
    distance_metric: str

class Config:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        self.model = ModelConfig(**config_dict["model"])
        self.training = TrainingConfig(**config_dict["training"])
        self.data = DataConfig(**config_dict["data"])
        self.inference = InferenceConfig(**config_dict["inference"])
