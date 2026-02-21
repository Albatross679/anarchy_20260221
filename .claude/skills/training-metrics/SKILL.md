---
name: training-metrics
description: |
  Collect comprehensive training metrics for PyTorch models, including per-epoch statistics, final model outputs, and attention matrices, saved as structured JSON.
  Use when: (1) Implementing or modifying training loops for neural networks, (2) Adding metrics collection to transformer models, (3) Setting up training output files for analysis, (4) The user asks to collect, save, or export training data and metrics.
---

# Training Metrics Collection

Collect comprehensive training metrics for PyTorch models, including per-epoch statistics, final model outputs, and attention matrices.

## Trigger

Use when:
- Implementing or modifying training loops for neural networks
- Adding metrics collection to transformer models
- Setting up training output files for analysis
- The user asks to collect or save training data

## Output Format

Save metrics to `output/<part>_training_metrics.json` with the following structure:

```json
{
  "metadata": {
    "task": "<task_name>",
    "timestamp": "<ISO 8601 UTC>",
    "device": "<cuda|cpu>",
    "hyperparameters": {
      "model": { ... },
      "training": { ... }
    }
  },
  "epochs": [
    {
      "epoch": 1,
      "train_loss": 0.0,
      "train_accuracy": 0.0,
      "dev_accuracy": 0.0,
      "gradient_norm": 0.0,
      "elapsed_seconds": 0.0
    }
  ],
  "final_results": {
    "train_accuracy": 0.0,
    "dev_accuracy": 0.0,
    "total_time_seconds": 0.0
  },
  "final_inference": {
    "example_input": "<input_string>",
    "log_probs": [[...], [...], ...],
    "attention_weights": [[[...], ...], ...]
  }
}
```

## Required Metrics

### 1. Metadata Section
- `task`: Name of the task (e.g., "letter_counting", "language_model")
- `timestamp`: ISO 8601 format with UTC timezone (e.g., "2026-01-30T22:08:22Z")
- `device`: Training device ("cuda" or "cpu")
- `hyperparameters.model`: All model architecture parameters
- `hyperparameters.training`: All training parameters (epochs, learning rate, etc.)

### 2. Per-Epoch Metrics
Collect after each epoch:
- `epoch`: Epoch number (1-indexed)
- `train_loss`: Average loss over training set
- `train_accuracy`: Accuracy on training set
- `dev_accuracy`: Accuracy on dev/validation set
- `gradient_norm`: Average gradient norm across training steps
- `elapsed_seconds`: Cumulative time since training started

### 3. Final Results
After training completes:
- `train_accuracy`: Final accuracy on full training set
- `dev_accuracy`: Final accuracy on dev set
- `total_time_seconds`: Total training time

### 4. Final Inference Output (Required)
Run one inference on a sample and save:
- `example_input`: The input string used for inference
- `log_probs`: The probability matrix (e.g., 20x3 for letter counting)
  - Shape: `[seq_len, num_classes]`
  - Values: Log probabilities from model output
  - Convert to nested list for JSON serialization
- `attention_weights`: Attention matrices from all layers
  - Shape: `[num_layers, seq_len, seq_len]`
  - One 2D matrix per transformer layer
  - Convert to nested list for JSON serialization

## Implementation

### Collecting Final Inference

After training, before returning the model:

```python
# Run inference on first dev example
model.eval()
with torch.no_grad():
    sample_ex = dev[0]
    log_probs, attn_maps = model.forward(sample_ex.input_tensor)

    # Convert to JSON-serializable format
    metrics_data["final_inference"] = {
        "example_input": sample_ex.input,
        "log_probs": log_probs.cpu().numpy().tolist(),
        "attention_weights": [attn.cpu().numpy().tolist() for attn in attn_maps]
    }
```

### Precision

- Round floating point metrics to 4 decimal places
- Round time values to 2 decimal places
- Keep full precision for log probabilities and attention weights

## Rules

1. **Always overwrite** the metrics file after each epoch (not append)
2. **Save incrementally** so partial results are available if training is interrupted
3. **Include final inference** with both log_probs matrix and attention weights
4. **Use first dev example** for final inference (consistent across runs)
5. **Convert tensors** to nested Python lists for JSON serialization
6. **Ensure output directory exists** before saving (`output/` by default)

## Example

For a letter counting task with sequence length 20 and 3 classes:

```json
{
  "metadata": { ... },
  "epochs": [ ... ],
  "final_results": {
    "train_accuracy": 0.9963,
    "dev_accuracy": 0.9954,
    "total_time_seconds": 417.13
  },
  "final_inference": {
    "example_input": "i like movies a lot ",
    "log_probs": [
      [-0.0012, -7.234, -8.456],
      [-0.0015, -6.892, -9.123],
      ...
    ],
    "attention_weights": [
      [
        [0.95, 0.03, 0.02, ...],
        [0.12, 0.85, 0.03, ...],
        ...
      ]
    ]
  }
}
```
