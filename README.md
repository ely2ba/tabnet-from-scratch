# TabNet from Scratch (In Progress)

This project is a full PyTorch reimplementation of TabNet, a deep learning architecture for tabular data introduced by Google Research (2019). TabNet leverages sequential attention and sparse feature selection to learn interpretable, high-performing representations from structured inputs.

---

## 🔍 Objectives

- Reconstruct the full TabNet architecture from first principles, including:
  - Feature Transformer
  - Attentive Transformer
  - Sparsemax activation
- Apply the model to real-world tabular datasets (e.g., Adult Income)
- Benchmark against traditional models such as XGBoost and LightGBM
- Visualize learned attention masks and feature importances across decision steps

---

## 🛠️ Features (Planned)

- Modular PyTorch implementation of:
  - Sparsemax activation
  - Feature and Attentive Transformers
  - Multi-step sequential attention flow
- Benchmarking utilities and training scripts
- Visual tools for exploring:
  - Feature masks at each decision step
  - Attention progression and model interpretability

---

## 📁 Project Structure (coming soon)

```
tabnet-from-scratch/
├── src/               Core model implementation
├── notebooks/         Training and evaluation workflows
├── utils/             Preprocessing and data loading
├── visuals/           Attention mask visualizations
└── README.md          This document
```

---

## 🚧 Status

🟡 Implementation in progress.  
✔️ Sparsemax implemented and tested. Core architecture components in progress.
Initial experiments and benchmarking expected in April 2025.

---

## 📜 License

MIT
