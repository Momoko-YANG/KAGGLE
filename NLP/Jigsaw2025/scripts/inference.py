#!/usr/bin/env python3
import sys
sys.path.append('.')

from src.config.model_config import Config
from src.data.loader import DataLoader
from src.data.preprocessor import TextPreprocessor
from src.models.embedding_model import EmbeddingModel
from src.features.embeddings import EmbeddingGenerator
from src.features.centroids import CentroidBuilder
from src.inference.predictor import ViolationPredictor
from src.utils.logging_utils import setup_logging
import pandas as pd

def main():
    setup_logging()
    config = Config()
    
    # Load data
    loader = DataLoader()
    df = loader.load_test_data(config.data.test_data_path)
    
    # Load model
    model_wrapper = EmbeddingModel(
        model_path=f"{config.data.output_dir}/final",
        max_seq_length=config.model.max_seq_length
    )
    model_wrapper.load_model()
    
    # Generate embeddings
    preprocessor = TextPreprocessor()
    embedding_generator = EmbeddingGenerator(model_wrapper, batch_size=config.inference.batch_size)
    text_to_embedding, rule_embeddings = embedding_generator.build_dataframe_embeddings(df, preprocessor)
    
    # Build centroids
    centroid_builder = CentroidBuilder()
    rule_centroids = centroid_builder.build_rule_centroids(
        df, text_to_embedding, rule_embeddings, preprocessor
    )
    
    # Predict
    predictor = ViolationPredictor(config.inference.distance_metric)
    row_ids, predictions = predictor.predict(
        df, text_to_embedding, rule_centroids, preprocessor
    )
    
    # Save submission
    submission = pd.DataFrame({
        'row_id': row_ids,
        'rule_violation': predictions
    })
    submission.to_csv('data/submissions/submission.csv', index=False)

if __name__ == "__main__":
    main()
