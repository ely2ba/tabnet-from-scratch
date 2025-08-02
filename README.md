

# TabNet from Scratch (PyTorch Implementation)

This project is a **complete PyTorch re-implementation** of [TabNet](https://arxiv.org/abs/1908.07442), a deep-learning architecture for tabular data introduced by Google Research (2019). Unlike typical multilayer perceptrons, TabNet uses **sequential attention**, **sparse feature selection**, and **interpretable decision steps** to achieve high performance while maintaining transparency.

---

## Objectives

* Reconstruct TabNet from first principles using modular PyTorch
* Implement core components:

  * Sparsemax activation
  * FeatureTransformer (shared and step-specific layers)
  * AttentiveTransformer (for sparse attention masks)
  * Multi-step TabNet decision architecture
* Apply the model to real-world data (Adult Income dataset)
* Benchmark against traditional models (XGBoost, LightGBM)
* Add interpretability tools for visualizing feature masks (planned)

---

## Features

* Modular, low-level PyTorch code without high-level wrappers
* Sequential attention via Sparsemax (Martins & Astudillo, 2016)
* End-to-end trainable for classification tasks
* Easily extensible for research or practical experimentation

---

## Project Structure

```
tabnet-from-scratch/
├── src/               # Core model components (GLU, Transformers, TabNet)
├── notebooks/         # Testing and training notebooks
├── utils/             # (planned) Data loading, metric utilities
├── visuals/           # (coming soon)
└── README.md          # This document
```

---

## Results: Adult Income Dataset

| Model         | Test Accuracy     |
| ------------- | ----------------- |
| TabNet        | **78.5 %**        |
| XGBoost       | 87.3 % (baseline) |
| LightGBM      | 86.7 %            |
| MLP (2-layer) | 76.4 %            |

*Note:* TabNet is trained from scratch without heavy tuning. Accuracy may improve with additional regularization, learning-rate schedules, or data augmentation techniques (e.g., VIME, mixup).

---

## Why This Project Matters

Tabular data underpins critical fields such as economics, finance, and healthcare, yet deep-learning models often lag behind tree-based methods on structured data.

**TabNet** offers an elegant solution by blending:

* **Interpretability** through sparse, step-wise attention
* **Deep-learning flexibility** with sequential feature selection
* **Gradient-based learning** that works end-to-end on tabular datasets

This project:

* Demystifies TabNet’s inner workings for research and education
* Provides a foundation for exploring fairness, multitask learning, or causal inference with neural networks
* Offers a transparent alternative to black-box tabular models, aligning with domains that require explainability

---

## Status

* Core implementation complete
* Basic benchmarking done
* Feature attribution and interpretability tools in progress
* More rigorous comparisons with CatBoost and tabular MLP variants planned

---

## License

This project is licensed under the MIT License. Contributions are welcome!

---
