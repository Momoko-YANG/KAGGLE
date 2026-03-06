#!/usr/bin/env python3
import sys
sys.path.append('.')

from src.config.model_config import Config
from src.data.loader import DataLoader
from src.data.preprocessor import TextPreprocessor
from src.data.triplet_dataset import TripletDatasetCreator
from src.models.embedding_model import EmbeddingModel
from src.models.trainer import ModelTrainer
from src.utils.logging_utils import setup_logging


def main():
    setup_logging()
    config = Config()
    
    # Load data
    loader = DataLoader()
    df = loader.load_test_data(config.data.test_data_path)
    
    # Initialize model
    model_wrapper = EmbeddingModel(
        model_path=config.model.base_model_path,
        max_seq_length=config.model.max_seq_length,
        use_fp16=config.model.use_fp16
    )
    model = model_wrapper.load_model()
    
    # Create training dataset
    triplet_creator = TripletDatasetCreator(config.training)
    train_dataset = triplet_creator.create_triplet_dataset(df)
    
    # Train
    trainer = ModelTrainer(model, config.training)
    trainer.train(train_dataset, config.data.output_dir)

if __name__ == "__main__":
    main()