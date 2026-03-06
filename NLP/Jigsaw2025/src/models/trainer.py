import logging

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import TripletLoss

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train sentence transformer with triplet loss."""

    def __init__(self, model, training_config):
        self.model = model
        self.config = training_config

    def train(self, train_dataset: Dataset, output_dir: str):
        """Fine-tune model on triplet dataset."""
        logger.info(f"Training on {len(train_dataset)} triplets")

        loss = TripletLoss(model=self.model, triplet_margin=self.config.triplet_margin)

        dataset_size = len(train_dataset)
        steps_per_epoch = max(1, dataset_size // self.config.batch_size)
        max_steps = steps_per_epoch * self.config.epochs

        args = SentenceTransformerTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=0,
            logging_steps=max(1, max_steps // 4),
            save_strategy="epoch",
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            max_grad_norm=self.config.max_grad_norm,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=True,
            dataloader_drop_last=False,
            max_steps=max_steps,
            report_to="none",
        )

        trainer = SentenceTransformerTrainer(model=self.model, args=args, train_dataset=train_dataset, loss=loss)

        trainer.train()

        final_path = f"{output_dir}/final"
        self.model.save(final_path)
        logger.info(f"Training completed. Model saved to {final_path}")

        return self.model
