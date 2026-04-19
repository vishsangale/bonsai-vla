# Experiment Results: SimpleViT vs. Vanilla ViT Baseline

This document summarizes the results of the modernized Vision Transformer (SimpleViT) compared against the vanilla Vision Transformer baseline on the CIFAR-10 dataset.

## Experiment Configuration

| Feature | SimpleViT | Vanilla ViT |
| :--- | :--- | :--- |
| **Epochs** | 300 | 300 |
| **Architecture** | SimpleViT (QK-Norm, GAP) | VisionTransformer (CLS Token) |
| **Positional Encoding** | 2D Sinusoidal | Learned |
| **Augmentation** | **RandAugment** | Standard (Crop/Flip) |
| **Optimizer** | AdamW | AdamW |
| **Learning Rate** | 6e-4 (scaled by batch) | 3e-4 |
| **Weight Decay** | 0.1 | 1e-4 |
| **Batch Size** | 512 | 64 |

## Key Metrics Comparison

The SimpleViT run demonstrates a significant performance boost and better generalization.

| Metric | SimpleViT (Modernized) | Vanilla ViT (Baseline) | Delta |
| :--- | :--- | :--- | :--- |
| **Final Validation Accuracy** | **87.46%** | 84.09% | **+3.37%** |
| **Best Validation Accuracy** | **87.47%** | 84.17% | **+3.30%** |
| **Final Validation Loss** | **0.459** | 1.200 | **-0.741** |
| **Final Training Accuracy** | 85.10% | 99.97% | -14.87% |
| **Final Training Loss** | 0.433 | 0.001 | +0.432 |

## Analysis & Observations

1.  **Superior Generalization**: The vanilla ViT baseline suffered from extreme overfitting, reaching nearly 100% training accuracy while validation accuracy plateaued around 84%. SimpleViT, despite having lower training accuracy (due to RandAugment making the training task harder), achieved significantly higher validation accuracy (87.46%).
2.  **Effective Regularization**: The combination of **RandAugment**, **QK-Normalization**, and higher **Weight Decay** (0.1 vs 1e-4) effectively prevented the model from memorizing the training set.
3.  **Stability**: The validation loss for SimpleViT (0.459) is much lower and more stable than the baseline (1.200), which likely started to diverge or saturate early due to overfitting.
4.  **Modern Enhancements**: The transition to Global Average Pooling (GAP) and fixed 2D sinusoidal embeddings in SimpleViT appears to provide a more robust representation for small-scale datasets like CIFAR-10 compared to the traditional CLS token and learned embeddings.

## Conclusion

The SimpleViT modernization was highly successful, providing a **+3.37%** absolute improvement in top-1 accuracy on CIFAR-10. This establishs a much stronger baseline for future VLA (Vision-Language-Action) model development.

---
*Results extracted on: 2026-04-18*
*Run ID: bonsai_simple_vit_cifar10_300ep*
