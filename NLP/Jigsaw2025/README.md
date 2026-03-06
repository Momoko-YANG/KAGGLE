# kaggle-Jigsaw2025

A machine learning system for detecting rule violations in text using fine-tuned sentence transformers and centroid-based classification.
This project is adapted from the Kaggle competition Jigsaw 2025, with the primary purpose of transforming a competition notebook into a production-ready MLE project. The goal is to gain hands-on experience with end-to-end engineering workflows — including model training, inference service deployment, and CI/CD integration.

## 📋 Project Overview

This project implements a sophisticated text classification system that:
- Fine-tunes BGE (BAAI General Embedding) models on rule-violation pairs
- Uses triplet loss to learn semantic representations
- Employs centroid-based distance metrics for prediction
- Handles multiple rules and their specific violation patterns

## 🏗️ Project Structure

```
rule-violation-detection/
├── config/                 # Configuration files
│   ├── config.yaml        # Main configuration
│   └── model_config.py    # Config data classes
├── src/                   # Source code
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # Model definitions and training
│   ├── features/         # Feature engineering
│   ├── inference/        # Prediction logic
│   └── utils/            # Utility functions
├── scripts/              # Execution scripts
│   ├── train.py         # Training script
│   └── inference.py     # Inference script
├── tests/               # Unit tests
├── notebooks/           # Jupyter notebooks
├── data/                # Data directory
└── models/              # Saved models

```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Momoko-YANG/kaggle-Jigsaw2025.git
cd kaggle-Jigsaw2025

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirement.txt

# Install package in development mode
pip install -e .
```

### Configuration

Edit `config/config.yaml` to adjust:
- Model paths and hyperparameters
- Training settings (epochs, batch size, learning rate)
- Data paths
- Inference settings

### Training

```bash
# Train the model
python scripts/train.py

# Or with custom config
python scripts/train.py --config config/custom_config.yaml
```

### Inference

```bash
# Run inference on test set
python scripts/inference.py

# Output will be saved to data/submissions/submission.csv
```

## 📊 Model Architecture

### Embedding Model
- **Base Model**: BAAI/bge-small-en-v1.5
- **Max Sequence Length**: 128 tokens
- **Embedding Dimension**: 768
- **Pooling**: Mean pooling

### Training Strategy
- **Loss Function**: Triplet Loss
- **Margin**: 0.25
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Precision**: FP16 (for efficiency)

### Prediction Method
- Centroid-based classification
- Euclidean distance metrics
- Rule-specific violation patterns

## 📁 Data Format

### Input CSV Format
```csv
row_id,body,rule,positive_example_1,positive_example_2,negative_example_1,negative_example_2
1,"Comment text","Rule description","Violation 1","Violation 2","Compliant 1","Compliant 2"
```

### Output CSV Format
```csv
row_id,rule_violation
1,0.234
2,-0.156
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data_loader.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Training Time | ~15 min (on GPU) |
| Inference Time | ~2 min (1000 samples) |
| Model Size | ~400 MB |

## 🔧 Advanced Usage

### Custom Text Preprocessing

```python
from src.data.preprocessor import TextPreprocessor

preprocessor = TextPreprocessor()
cleaned_text = preprocessor.clean_text("Your text with https://example.com")
unique_texts = preprocessor.collect_unique_texts(dataframe)
```

### Custom Model Training

```python
from src.models.embedding_model import EmbeddingModel
from src.models.trainer import ModelTrainer
from config.model_config import Config

config = Config()
model = EmbeddingModel(
    model_path=config.model.base_model_path,
    max_seq_length=128,
    use_fp16=True
)
model.load_model()

trainer = ModelTrainer(model, config.training)
trainer.train(train_dataset, output_dir="./custom_models")
```

### Batch Inference

```python
from src.inference.predictor import ViolationPredictor

predictor = ViolationPredictor(distance_metric="euclidean")
row_ids, predictions = predictor.predict(
    df=test_df,
    text_to_embedding=embeddings_dict,
    rule_centroids=centroids_dict,
    text_preprocessor=preprocessor
)
```

## 🛠️ Development

### Code Style
This project follows PEP 8 guidelines. Format code with:
```bash
# Install formatters
pip install black isort flake8

# Format code
black src/ tests/
isort src/ tests/

# Check style
flake8 src/ tests/
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement feature in appropriate module
3. Add unit tests in `tests/`
4. Update documentation
5. Submit pull request

## 📚 API Reference

### DataLoader
```python
class DataLoader:
    @staticmethod
    def load_test_data(file_path: str) -> pd.DataFrame
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> bool
```

### TextPreprocessor
```python
class TextPreprocessor:
    @staticmethod
    def clean_text(text: str) -> str
    
    @staticmethod
    def collect_unique_texts(df: pd.DataFrame) -> List[str]
```

### EmbeddingModel
```python
class EmbeddingModel:
    def __init__(self, model_path: str, max_seq_length: int, use_fp16: bool)
    def load_model(self) -> SentenceTransformer
    def encode(self, texts: List[str], batch_size: int) -> np.ndarray
    def save(self, output_path: str)
```

### CentroidBuilder
```python
class CentroidBuilder:
    @staticmethod
    def build_rule_centroids(
        df: pd.DataFrame,
        text_to_embedding: Dict[str, np.ndarray],
        rule_embeddings: Dict[str, np.ndarray],
        text_preprocessor
    ) -> Dict
```

### ViolationPredictor
```python
class ViolationPredictor:
    def __init__(self, distance_metric: str)
    def predict(
        self,
        df: pd.DataFrame,
        text_to_embedding: Dict,
        rule_centroids: Dict,
        text_preprocessor
    ) -> Tuple[list, np.ndarray]
```

## 🐛 Troubleshooting

### Common Issues

**Issue: CUDA out of memory**
```yaml
# Solution: Reduce batch size in config.yaml
training:
  batch_size: 16  # Reduce from 32
```

**Issue: Model loading fails**
```python
# Solution: Check model path and ensure model exists
import os
assert os.path.exists(model_path), f"Model not found at {model_path}"
```

**Issue: Slow inference**
```yaml
# Solution: Enable FP16 and increase batch size
model:
  use_fp16: true
inference:
  batch_size: 128
```

## 📊 Monitoring and Logging

Logs are saved to `logs/` directory with timestamps:
```
logs/
├── rule_violation_20250101_120000.log
└── rule_violation_20250102_143000.log
```

View logs in real-time:
```bash
tail -f logs/rule_violation_*.log
```



## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 🙏 Acknowledgments

- BAAI for the BGE embedding models
- Sentence Transformers library
- Kaggle Jigsaw competition organizers


