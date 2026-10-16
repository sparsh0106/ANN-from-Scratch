# 🧠 Artificial Neural Network (ANN) from Scratch

A complete, mathematical implementation of a multi-layer **Artificial Neural Network (ANN)** written in **pure NumPy** without using deep learning frameworks such as PyTorch or TensorFlow. The network is trained and evaluated on the classic **Iris flower classification dataset**.

---

## 📂 Repository Structure

- `scratch_ann.ipynb`: Step-by-step Jupyter Notebook implementing forward propagation, ReLU & Softmax activations, cross-entropy loss, analytical backpropagation, and gradient descent optimization.
- `Iris.csv`: The Iris dataset (150 samples, 4 botanical features, 3 species classes).
- `README.md`: Project documentation.

---

## ⚙️ Key Concepts Implemented

1. **Data Preprocessing**:
   - Feature standardization (`StandardScaler`) and one-hot target encoding (`OneHotEncoder`).
2. **Forward Propagation**:
   - Linear transformations ($Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$) with **ReLU** hidden activations and numerically stable **Softmax** output probabilities.
3. **Loss Computation**:
   - Multi-class **Categorical Cross-Entropy** loss.
4. **Analytical Backpropagation**:
   - Explicit chain-rule gradients ($rac{\partial \mathcal{L}}{\partial W^{[l]}}, rac{\partial \mathcal{L}}{\partial b^{[l]}}$) across all layers and iterative parameter updates via mini-batch/batch gradient descent.

---

## 🚀 Getting Started

```bash
git clone https://github.com/sparsh0106/ANN-from-Scratch.git
cd ANN-from-Scratch
pip install numpy pandas scikit-learn matplotlib jupyter
jupyter notebook scratch_ann.ipynb
```

---

## 👤 Author

**Sparsh Agarwal** — [GitHub](https://github.com/sparsh0106)
