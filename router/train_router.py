#!/usr/bin/env python3
"""Script to train the local DistilBERT sequence classifier for prompt difficulty."""

import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

# Ensure we can import agent modules if needed
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    from torch.optim import AdamW
except ImportError as exc:
    print(f"Error importing dependencies: {exc}")
    print("Please run: pip install transformers torch")
    sys.exit(1)


class PromptDataset(Dataset):
    """Custom PyTorch dataset for tokenizing prompts."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def train_model():
    """Fine-tunes DistilBERT on prompt difficulty labels and saves weights."""
    print("Initializing DistilBERT training script...")

    # Define paths
    root_dir = Path(__file__).resolve().parents[1]
    dataset_path = root_dir / "router" / "router_dataset.json"
    save_path = root_dir / "models" / "router"

    # Check for dataset
    if not dataset_path.exists():
        print(f"Error: Dataset not found at '{dataset_path}'")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("Error: Dataset is empty. Please add examples to 'router_dataset.json' before training.")
        sys.exit(1)

    print(f"Loaded {len(data)} training examples.")

    # Parse dataset
    prompts = []
    labels = []
    for idx, item in enumerate(data):
        prompt = item.get("prompt")
        label = item.get("label")
        if not prompt or not label:
            print(f"Warning: Skipping malformed item at index {idx}.")
            continue
        prompts.append(prompt)
        # 0 = easy, 1 = hard
        labels.append(0 if label.lower() == "easy" else 1)

    if not prompts:
        print("Error: No valid examples to train on.")
        sys.exit(1)

    # Auto-detect hardware device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device for training.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders) device for training.")
    else:
        device = torch.device("cpu")
        print("Using CPU device for training (slower).")

    # Load tokenizer and model
    print("Downloading/loading pretrained DistilBERT tokenizer and model...")
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # Tokenize input data
    print("Tokenizing prompt inputs...")
    encodings = tokenizer(
        prompts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors=None  # We convert to tensor in PromptDataset
    )

    dataset = PromptDataset(encodings, labels)
    # Smaller batch size to prevent OOM
    batch_size = min(8, len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Calculate class weights to handle dataset imbalance
    num_easy = labels.count(0)
    num_hard = labels.count(1)
    num_easy = max(1, num_easy)
    num_hard = max(1, num_hard)
    total = len(labels)
    weight_easy = total / (2.0 * num_easy)
    weight_hard = total / (2.0 * num_hard)
    class_weights = torch.tensor([weight_easy, weight_hard], dtype=torch.float, device=device)
    print(f"Class distribution: {num_easy} easy, {num_hard} hard.")
    print(f"Applied loss weights: easy={weight_easy:.2f}, hard={weight_hard:.2f}")

    # Setup optimizer, custom loss function, and training hyperparameters
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    epochs = 3

    print(f"Starting fine-tuning for {epochs} epochs (batch size: {batch_size})...")
    model.train()
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Move inputs to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            loss = loss_fn(outputs.logits, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Calculate metrics
            preds = torch.argmax(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == batch_labels).item()
            total_predictions += len(batch_labels)

            if (batch_idx + 1) % 2 == 0 or (batch_idx + 1) == len(dataloader):
                print(f"  [Epoch {epoch} Batch {batch_idx + 1}/{len(dataloader)}] current batch loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        epoch_acc = (correct_predictions / total_predictions) * 100.0
        print(f"Epoch {epoch}/{epochs} Completed - Avg Loss: {avg_loss:.4f} | Training Accuracy: {epoch_acc:.1f}%")

    elapsed = time.perf_counter() - start_time
    print(f"Training completed successfully in {elapsed:.1f}s.")

    # Save model and tokenizer
    print(f"Saving fine-tuned model and tokenizer to '{save_path}'...")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Model saved successfully. You are ready to run the three-tier routing pipeline!")


if __name__ == "__main__":
    train_model()
