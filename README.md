# TabNet from Scratch (PyTorch Implementation)

This project is a **complete PyTorch reimplementation** of [TabNet](https://arxiv.org/abs/1908.07442), a deep learning architecture for tabular data introduced by Google Research (2019). Unlike typical MLPs, TabNet uses **sequential attention**, **sparse feature selection**, and **interpretable decision steps** to achieve high performance while maintaining transparency.

---

## 🔍 Objectives

- ✅ Reconstruct TabNet from first principles using modular PyTorch.
- ✅ Implement core components:
  - Sparsemax activation
  - FeatureTransformer (shared + step-specific layers)
  - AttentiveTransformer (for sparse attention masks)
  - Multi-step TabNet decision architecture
- ✅ Apply to real-world data (Adult Income dataset)
- ✅ Benchmark against traditional models (XGBoost, LightGBM)
- 🔜 Add interpretability tools for visualizing feature masks

---

## 🛠️ Features

- Modular, low-level PyTorch code (no high-level DL wrappers)
- Sequential attention using Sparsemax (Martins & Astudillo, 2016)
- End-to-end trainable with classification objective
- Easily extensible for research or practical experimentation

---

## 📁 Project Structure

```
tabnet-from-scratch/
├── src/               # Core model components (GLU, Transformers, TabNet)
├── notebooks/         # Testing and training notebooks
├── utils/             # (planned) Data loading, metric utilities
├── visuals/           # (coming soon)
└── README.md          This document
```

---

## 📈 Results (Adult Income Dataset)

| Model        | Test Accuracy |
|--------------|----------------|
| TabNet       | **78.5%**       |
| XGBoost      | 87.3% (baseline)|
| LightGBM     | 86.7%           |
| MLP (2-layer)| 76.4%           |

Note: TabNet is trained from scratch without heavy tuning. With additional regularization, learning rate schedules, and data augmentation (e.g., VIME, mixup), accuracy may improve further.

---

## 🤔 Why This Project Matters

Tabular data underpins critical fields like economics, finance, and healthcare — yet deep learning models often lag behind tree-based methods on structured data.

**TabNet** introduces an elegant solution by blending:
- **Interpretability** via sparse, step-wise attention
- **Deep learning flexibility** with sequential feature selection
- **Gradient-based learning** that works end-to-end on tabular datasets

This project:
- Helps demystify the inner workings of TabNet for research & education
- Can serve as a foundation for exploring fairness, multitask learning, or causal inference with neural networks
- Offers a transparent alternative to black-box tabular models — aligning with needs in fields requiring model explainability

---

## 🚧 Status

- ✅ Core implementation complete
- ✅ Basic benchmarking done
- 🔜 Feature attribution & interpretability tools
- 🔜 More rigorous comparisons with CatBoost and tabular MLP variants

---

## 📜 License

This project is licensed under the MIT License. Contributions welcome!

---

