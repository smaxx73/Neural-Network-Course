# Session 8: Generalization & Regularization
## When Good Training Goes Bad

**Course: Neural Networks for Engineers**  
**Duration: 2 hours**

---

## Table of Contents

### Part I — Concepts (≈ 45 min)
1. [Recap: What We Know So Far](#recap)
2. [The Generalization Problem](#generalization)
3. [Train / Validation / Test Splits](#splits)
4. [Overfitting & Underfitting](#overfitting)
5. [Regularization Techniques](#regularization)
6. [Modern Optimizers](#optimizers)

### Part II — Mini-Projects (≈ 75 min)
7. [Mini-Project A: The Overfitting Lab](#project-a)
8. [Mini-Project B: Regularization Showdown](#project-b)
9. [Mini-Project C: Optimizer Olympics](#project-c)

---

# Part I — Concepts

---

## 1. Recap: What We Know So Far {#recap}

### What We've Learned

✅ **Perceptron & MLP**: From single neurons to multi-layer networks (Sessions 2–4)  
✅ **Gradient descent & backpropagation**: Automatic weight learning (Sessions 5–6)  
✅ **Classification**: Sigmoid/softmax, cross-entropy, evaluation metrics (Session 7)  
✅ **Open question**: Our spiral classifier hits 95% on training data — but will it work on **new** data?

### 🤔 Quick Questions (from Session 7's "Think About")

**Q1:** Does 95% training accuracy mean the model will work well on new spirals?

<details>
<summary>Click to reveal answer</summary>
**Not necessarily.** The model may have **memorized** the training data instead of learning the underlying pattern. We need to test on data the model has never seen — this is the **generalization** problem.
</details>

**Q2:** What if we increased the hidden layer to 500 neurons?

<details>
<summary>Click to reveal answer</summary>
More neurons = more capacity to memorize. With 500 hidden neurons on a 300-sample spiral dataset, the model could fit every single point perfectly — including noise! It would likely **overfit**: perfect on training data, poor on new data.
</details>

**Q3:** How would you know if your model is too simple vs too complex?

<details>
<summary>Click to reveal answer</summary>
Compare **training loss** vs **validation loss**. If training loss is low but validation loss is high, the model is too complex (overfitting). If both are high, the model is too simple (underfitting). This is what **learning curves** show us.
</details>

---

## 2. The Generalization Problem {#generalization}

### The Central Question of Machine Learning

We don't care about performance on data we've already seen. We care about **unseen data**.

**Generalization** = the ability of a model to perform well on new, previously unseen data.

### A Cautionary Tale

Imagine fitting a polynomial to noisy data:

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# True function: y = sin(x)
x_true = np.linspace(0, 2 * np.pi, 200)
y_true = np.sin(x_true)

# Training data (15 noisy samples)
x_train = np.sort(np.random.uniform(0, 2 * np.pi, 15))
y_train = np.sin(x_train) + np.random.randn(15) * 0.3

# Fit polynomials of increasing degree
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
degrees = [2, 5, 14]
titles = ['Degree 2 (Underfitting)', 'Degree 5 (Good fit)', 'Degree 14 (Overfitting)']

for ax, deg, title in zip(axes, degrees, titles):
    coeffs = np.polyfit(x_train, y_train, deg)
    y_fit = np.polyval(coeffs, x_true)
    
    ax.plot(x_true, y_true, 'g--', linewidth=2, label='True function', alpha=0.7)
    ax.scatter(x_train, y_train, c='blue', s=60, zorder=5, 
               edgecolors='black', label='Training data')
    ax.plot(x_true, y_fit, 'r-', linewidth=2, label=f'Polynomial (deg {deg})')
    
    y_pred_train = np.polyval(coeffs, x_train)
    train_err = np.mean((y_train - y_pred_train) ** 2)
    test_err = np.mean((y_true - y_fit) ** 2)
    
    ax.set_title(f'{title}\nTrain MSE: {train_err:.3f} | Test MSE: {test_err:.3f}', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-2, 2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Observation:** Degree 14 has the **lowest** training error but the **highest** test error. It memorized the noise!

### The Bias-Variance Tradeoff (Intuitive)

| | Too Simple (Underfitting) | Just Right | Too Complex (Overfitting) |
|---|---|---|---|
| **Training error** | High | Low | Very low |
| **Test error** | High | Low | **High** |
| **Problem** | Can't capture the pattern | — | Memorizes noise |
| **Bias** | High (wrong assumptions) | Low | Low |
| **Variance** | Low (stable predictions) | Low | High (sensitive to training data) |

The goal is to find the sweet spot: complex enough to capture the pattern, simple enough to ignore the noise.

---

## 3. Train / Validation / Test Splits {#splits}

### Why Three Sets?

| Set | Purpose | When used | Typical size |
|---|---|---|---|
| **Training** | Learn weights | Every epoch | 60–80% |
| **Validation** | Tune hyperparameters, detect overfitting | During training | 10–20% |
| **Test** | Final, unbiased evaluation | Once, at the very end | 10–20% |

### The Golden Rule

> **Never** use test data for any decision during training or model selection. It must remain untouched until the final evaluation.

If you peek at test data to tune your model, the test score becomes biased — it no longer reflects true generalization.

### Learning Curves: The Diagnostic Tool

Plot **training loss** and **validation loss** over epochs:

```
Loss                                Loss                                Loss
 │ ╲                                │ ╲                                │ ╲  train
 │  ╲                               │  ╲  train                       │   ╲────────────
 │   ╲───── val                     │   ╲──────                       │          ╱──── val
 │    ╲──── train                   │    ╲───── val                   │        ╱
 │                                  │                                 │      ╱
 └──────────── Epoch                └──────────── Epoch               └──────────── Epoch
   UNDERFITTING                       GOOD FIT                          OVERFITTING
   Both losses high                   Both losses low                   Gap between curves
   Model too simple                   and close together                Model too complex
```

**Key diagnostic:** the **gap** between training and validation loss tells you about overfitting.

---

## 4. Overfitting & Underfitting {#overfitting}

### Causes and Cures

| Problem | Cause | Symptoms | Solutions |
|---|---|---|---|
| **Underfitting** | Model too simple | Both train & val loss high | More neurons/layers, train longer, reduce regularization |
| **Overfitting** | Model too complex or not enough data | Train loss ≪ val loss | Regularization, more data, simpler model, early stopping |

### Early Stopping

The simplest regularization technique: **stop training when validation loss starts increasing**.

The idea: save the best weights (lowest validation loss), and restore them when patience runs out.

---

## 5. Regularization Techniques {#regularization}

### The Idea

Regularization = adding **constraints** or **penalties** to prevent the model from becoming too complex.

Think of it as telling the model: "Don't just fit the data — keep things **simple**."

### L2 Regularization (Weight Decay)

Add a penalty for large weights to the loss:

$$
L_{\text{total}} = L_{\text{data}} + \frac{\lambda}{2} \sum_l \| W^{(l)} \|^2_F
$$

Where $\lambda$ controls the regularization strength and $\| W \|^2_F = \sum_{ij} w_{ij}^2$ is the Frobenius norm.

**Effect on gradient:**

$$
\frac{\partial L_{\text{total}}}{\partial W^{(l)}} = \frac{\partial L_{\text{data}}}{\partial W^{(l)}} + \lambda W^{(l)}
$$

The update becomes:

$$
W^{(l)} \leftarrow (1 - \eta\lambda) W^{(l)} - \eta \frac{\partial L_{\text{data}}}{\partial W^{(l)}}
$$

That factor $(1 - \eta\lambda)$ **shrinks** the weights at every step — hence the name **weight decay**.

**Intuition:** Large weights create sharp, complex decision boundaries. Penalizing large weights encourages smoother, simpler boundaries that generalize better.

### L1 Regularization (Sparsity)

$$
L_{\text{total}} = L_{\text{data}} + \lambda \sum_l \| W^{(l)} \|_1
$$

**Difference from L2:** L1 drives weights to **exactly zero**, creating sparse networks. L2 shrinks weights toward zero but rarely makes them exactly zero.

| | L1 | L2 |
|---|---|---|
| **Penalty** | Sum of $|w|$ | Sum of $w^2$ |
| **Effect** | Sparse weights (feature selection) | Small weights (smooth boundaries) |
| **Gradient** | $\lambda \cdot \text{sign}(w)$ | $\lambda \cdot w$ |

### Dropout (Intuitive)

During training, **randomly set** a fraction $p$ of hidden neurons to zero at each forward pass.

```
Without dropout:          With dropout (p=0.5):

  x₁ ── h₁ ── h₄ ──       x₁ ── h₁ ── ╳  ──
       ╲╱    ╲╱               ╲╱    ╲╱
       ╱╲    ╱╲               ╱╲    ╱╲
  x₂ ── h₂ ── h₅ ── ŷ     x₂ ── ╳  ── h₅ ── ŷ
       ╲╱    ╲╱               ╲╱    ╲╱
       ╱╲    ╱╲               ╱╲    ╱╲
  x₃ ── h₃ ── h₆ ──       x₃ ── h₃ ── h₆ ──

  All neurons active        h₂ and h₄ "dropped"
```

**Why it works:**
- Forces the network to not rely on any single neuron
- Like training an **ensemble** of smaller networks
- Uses **inverted dropout**: scale surviving neurons by $\frac{1}{1-p}$ during training so nothing changes at test time

---

## 6. Modern Optimizers {#optimizers}

### SGD with Momentum

**Idea:** Accumulate a "velocity" — like a ball rolling downhill with inertia.

$$
v \leftarrow \beta v + (1 - \beta) \nabla L
$$
$$
w \leftarrow w - \eta v
$$

Where $\beta \approx 0.9$ is the momentum coefficient.

**Effect:** Smooths out oscillations and accelerates through consistent gradient directions.

### Adam (Adaptive Moment Estimation)

**Idea:** Adapt the learning rate **per weight**, using both first moment (mean) and second moment (variance) of gradients.

$$
m \leftarrow \beta_1 m + (1 - \beta_1) \nabla L \qquad \text{(mean of gradients)}
$$
$$
v \leftarrow \beta_2 v + (1 - \beta_2) (\nabla L)^2 \qquad \text{(variance of gradients)}
$$
$$
w \leftarrow w - \eta \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
$$

Where $\hat{m}$ and $\hat{v}$ are bias-corrected: $\hat{m} = \frac{m}{1 - \beta_1^t}$, $\hat{v} = \frac{v}{1 - \beta_2^t}$.

**Default hyperparameters:** $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

### Practical Guidelines

| Situation | Recommended optimizer |
|---|---|
| First try / don't know | **Adam** (lr=0.001) |
| Want best generalization | **SGD + Momentum** (lr=0.01, requires tuning) |
| Small dataset | **Adam** (converges faster) |
| Large dataset + long training | **SGD + Momentum** (often generalizes better) |

---

# Part II — Mini-Projects

### Shared Toolkit

All mini-projects reuse code from Sessions 5–7. **Copy this at the top of your notebook.**

```python
import numpy as np
import matplotlib.pyplot as plt

# ─── Utilities from previous sessions ───

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def softmax(z):
    z_shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def categorical_cross_entropy(y_true, y_hat):
    N = y_true.shape[1]
    y_hat_clipped = np.clip(y_hat, 1e-7, 1 - 1e-7)
    return -np.sum(y_true * np.log(y_hat_clipped)) / N

def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def classification_report(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    print(f"{'Class':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print("-" * 44)
    for k in range(n_classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        support = cm[k, :].sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{'Class '+str(k):>8} {prec:8.3f} {rec:8.3f} {f1:8.3f} {support:8d}")

# ─── Datasets ───

def generate_moons(n_samples=500, noise=0.2, seed=42):
    """Two interleaving half-circles."""
    np.random.seed(seed)
    n = n_samples // 2
    theta1 = np.linspace(0, np.pi, n)
    theta2 = np.linspace(0, np.pi, n)
    X1 = np.vstack([np.cos(theta1), np.sin(theta1)])
    X2 = np.vstack([np.cos(theta2) + 0.5, -np.sin(theta2) + 0.5])
    X = np.hstack([X1, X2]) + np.random.randn(2, n_samples) * noise
    y_labels = np.hstack([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    y_oh = np.zeros((2, n_samples))
    y_oh[y_labels, np.arange(n_samples)] = 1
    idx = np.random.permutation(n_samples)
    return X[:, idx], y_oh[:, idx], y_labels[idx]

def generate_spiral(n_per_class=150, n_classes=3, noise=0.25, seed=42):
    """Spiral dataset from Session 7."""
    np.random.seed(seed)
    N = n_per_class * n_classes
    X = np.zeros((2, N))
    y = np.zeros(N, dtype=int)
    for k in range(n_classes):
        s, e = k * n_per_class, (k + 1) * n_per_class
        r = np.linspace(0.2, 1.0, n_per_class)
        theta = np.linspace(k * 4.0, (k + 1) * 4.0, n_per_class) + np.random.randn(n_per_class) * noise
        X[0, s:e] = r * np.cos(theta)
        X[1, s:e] = r * np.sin(theta)
        y[s:e] = k
    y_oh = np.zeros((n_classes, N))
    y_oh[y, np.arange(N)] = 1
    idx = np.random.permutation(N)
    return X[:, idx], y_oh[:, idx], y[idx]
```

---

## 7. Mini-Project A: The Overfitting Lab {#project-a}

### 🎯 Goal

Build a complete train/val/test pipeline **from scratch**, deliberately overfit a model, diagnose it, then cure it.

**Skills reused:** MLP forward/backward (Session 6), CCE loss + softmax (Session 7), evaluation metrics (Session 7).  
**New skills:** Data splitting, learning curves, early stopping.

---

### Phase 1 — Implement Train/Val/Test Split

**Task:** Write a function that shuffles data and splits it into three sets.

```python
def train_val_test_split(X, y, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split data into train, validation, and test sets.
    
    Parameters:
    -----------
    X : array, shape (n_features, N)
    y : array, shape (n_classes, N)
    val_ratio, test_ratio : float
    
    Returns:
    --------
    X_train, y_train, X_val, y_val, X_test, y_test
    """
    np.random.seed(seed)
    N = X.shape[1]
    
    # TODO: Generate shuffled indices
    indices = ___
    
    # TODO: Compute split boundaries
    n_test = ___
    n_val = ___
    n_train = ___
    
    # TODO: Slice indices into three groups
    train_idx = ___
    val_idx = ___
    test_idx = ___
    
    return (X[:, train_idx], y[:, train_idx],
            X[:, val_idx], y[:, val_idx],
            X[:, test_idx], y[:, test_idx])
```

<details>
<summary>Solution</summary>

```python
def train_val_test_split(X, y, val_ratio=0.15, test_ratio=0.15, seed=42):
    np.random.seed(seed)
    N = X.shape[1]
    indices = np.random.permutation(N)
    n_test = int(N * test_ratio)
    n_val = int(N * val_ratio)
    n_train = N - n_val - n_test
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return (X[:, train_idx], y[:, train_idx],
            X[:, val_idx], y[:, val_idx],
            X[:, test_idx], y[:, test_idx])
```
</details>

**Verify your split:**

```python
X_all, y_all_oh, y_all_lbl = generate_moons(n_samples=400, noise=0.2)
X_tr, y_tr, X_va, y_va, X_te, y_te = train_val_test_split(X_all, y_all_oh)

print(f"Train: {X_tr.shape[1]}, Val: {X_va.shape[1]}, Test: {X_te.shape[1]}")
# Expected: Train: 280, Val: 60, Test: 60

# Also keep label versions for accuracy computation
y_tr_l = np.argmax(y_tr, axis=0)
y_va_l = np.argmax(y_va, axis=0)
y_te_l = np.argmax(y_te, axis=0)
```

### Phase 2 — Write the Training Loop with Val Tracking

Here is our base MLP (same as Session 7):

```python
class MLP:
    """Multi-class MLP from Session 7."""
    
    def __init__(self, n_input, n_hidden, n_classes, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(n_hidden, n_input) * np.sqrt(2.0 / n_input)
        self.b1 = np.zeros((n_hidden, 1))
        self.W2 = np.random.randn(n_classes, n_hidden) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros((n_classes, 1))
    
    def forward(self, X):
        self.z1 = self.W1 @ X + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = softmax(self.z2)
        return self.a2
    
    def loss(self, y_true):
        return categorical_cross_entropy(y_true, self.a2)
    
    def backward(self, X, y_true, lr):
        N = X.shape[1]
        delta2 = (self.a2 - y_true) / N
        dW2 = delta2 @ self.a1.T
        db2 = np.sum(delta2, axis=1, keepdims=True)
        delta1 = (self.W2.T @ delta2) * (self.z1 > 0).astype(float)
        dW1 = delta1 @ X.T
        db1 = np.sum(delta1, axis=1, keepdims=True)
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
    
    def predict(self, X):
        self.forward(X)
        return np.argmax(self.a2, axis=0)
```

**Task:** Write the function that trains the model and records train + val loss at each epoch. The key constraint: **the validation set must never influence the weights.**

```python
def train_with_tracking(model, X_tr, y_tr, X_va, y_va, lr, n_epochs):
    """
    Train the model and record train + val loss at each epoch.
    
    IMPORTANT: Validation data must NOT influence weights!
    
    Returns:
    --------
    train_losses, val_losses : lists of float
    """
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        # TODO (3 steps):
        # 1. Forward pass on TRAINING data
        # 2. Record training loss
        # 3. Backward pass (updates weights) on TRAINING data
        ___
        ___
        ___
        
        # TODO (2 steps):
        # 4. Forward pass on VALIDATION data (no backward!)
        # 5. Record validation loss
        ___
        ___
    
    return train_losses, val_losses
```

<details>
<summary>Solution</summary>

```python
def train_with_tracking(model, X_tr, y_tr, X_va, y_va, lr, n_epochs):
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        # Train step
        model.forward(X_tr)
        train_losses.append(model.loss(y_tr))
        model.backward(X_tr, y_tr, lr)
        
        # Val step (forward ONLY — no backward!)
        model.forward(X_va)
        val_losses.append(model.loss(y_va))
    
    return train_losses, val_losses
```

Note: we record training loss **before** the backward pass so it reflects the same weights used for the validation loss.
</details>

### Phase 3 — Overfit on Purpose and Diagnose

**Task:** Train a **deliberately overpowered model** (200 hidden neurons for a 2D, 280-sample dataset). Then write the plotting code to produce a 1×2 figure: learning curves on the left, decision boundary on the right.

```python
# Train the overpowered model
model_big = MLP(n_input=2, n_hidden=200, n_classes=2, seed=42)
t_losses, v_losses = train_with_tracking(model_big, X_tr, y_tr, X_va, y_va,
                                          lr=1.0, n_epochs=3000)

# Print final metrics
train_acc = np.mean(model_big.predict(X_tr) == y_tr_l) * 100
val_acc = np.mean(model_big.predict(X_va) == y_va_l) * 100
test_acc = np.mean(model_big.predict(X_te) == y_te_l) * 100
print(f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Test: {test_acc:.1f}%")

# TODO: Create a figure with 2 subplots side by side (figsize 16x6)
#
# Left plot — Learning curves:
#   - Plot t_losses as 'Train loss' (blue, linewidth=2)
#   - Plot v_losses as 'Val loss' (orange, linewidth=2)
#   - Add xlabel ('Epoch'), ylabel ('CCE Loss'), title, legend, grid
#
# Right plot — Decision boundary:
#   - Create a meshgrid covering the data range with ±0.5 margin
#   - Forward the grid through model_big.predict()
#   - Use contourf to show predicted class regions
#   - Scatter training points (circles) and val points (squares)
#     colored by true label with cmap='bwr'
#
# This is YOUR first diagnostic figure — make it readable!

fig, axes = plt.subplots(___, ___, figsize=(___))

# Left: learning curves
ax = axes[0]
___

# Right: decision boundary
ax = axes[1]
___

plt.tight_layout()
plt.show()
```

<details>
<summary>Solution</summary>

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: learning curves
ax = axes[0]
ax.plot(t_losses, label='Train loss', linewidth=2, color='blue')
ax.plot(v_losses, label='Val loss', linewidth=2, color='orange')
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('CCE Loss', fontsize=14)
ax.set_title('Learning Curves — 200 Hidden Neurons', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Right: decision boundary
ax = axes[1]
xx, yy = np.meshgrid(
    np.linspace(X_all[0].min()-0.5, X_all[0].max()+0.5, 200),
    np.linspace(X_all[1].min()-0.5, X_all[1].max()+0.5, 200))
grid = np.vstack([xx.ravel(), yy.ravel()])
Z = model_big.predict(grid).reshape(xx.shape)
ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=['#ADD8E6', '#FFCCCB'], alpha=0.4)
ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
ax.scatter(X_tr[0], X_tr[1], c=y_tr_l, cmap='bwr', edgecolors='black', s=40, alpha=0.7, label='Train')
ax.scatter(X_va[0], X_va[1], c=y_va_l, cmap='bwr', edgecolors='black', s=40, marker='s', alpha=0.7, label='Val')
ax.set_title('Decision Boundary', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```
</details>

**Diagnostic questions** (write answers in your notebook):

1. At roughly which epoch does the validation loss start diverging from the training loss?
2. Is the decision boundary smooth or jagged? What does that tell you?
3. What is the gap between train and test accuracy?

### Phase 4 — Implement Early Stopping

**Task:** Write an early stopping training loop. You must track the best validation loss, save/restore weights, and stop when patience runs out.

```python
def train_with_early_stopping(model, X_tr, y_tr, X_va, y_va,
                               lr, max_epochs, patience=50):
    """
    Train with early stopping: stop when val loss hasn't improved
    for `patience` epochs. Restore the best weights at the end.
    
    Returns:
    --------
    train_losses, val_losses : lists
    best_epoch : int
    """
    train_losses, val_losses = [], []
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    # TODO: Save initial copies of all 4 model weight arrays.
    # Hint: use .copy() — without it you'd save references, not snapshots!
    best_W1 = ___
    best_b1 = ___
    best_W2 = ___
    best_b2 = ___
    
    for epoch in range(max_epochs):
        # ── Train step ──
        model.forward(X_tr)
        train_losses.append(model.loss(y_tr))
        model.backward(X_tr, y_tr, lr)
        
        # ── Val step ──
        model.forward(X_va)
        val_loss = model.loss(y_va)
        val_losses.append(val_loss)
        
        # TODO: If this is a new best val loss →
        #       update best_val_loss, best_epoch, and save weight copies
        if ___:
            ___
        
        # TODO: If patience is exceeded → print a message and break
        if ___:
            print(f"Early stopping at epoch {epoch} "
                  f"(best was epoch {best_epoch}, val loss {best_val_loss:.4f})")
            break
    
    # TODO: Restore the best weights into the model
    ___
    
    return train_losses, val_losses, best_epoch
```

<details>
<summary>Solution</summary>

```python
def train_with_early_stopping(model, X_tr, y_tr, X_va, y_va,
                               lr, max_epochs, patience=50):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_epoch = 0
    
    best_W1 = model.W1.copy()
    best_b1 = model.b1.copy()
    best_W2 = model.W2.copy()
    best_b2 = model.b2.copy()
    
    for epoch in range(max_epochs):
        model.forward(X_tr)
        train_losses.append(model.loss(y_tr))
        model.backward(X_tr, y_tr, lr)
        
        model.forward(X_va)
        val_loss = model.loss(y_va)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_W1 = model.W1.copy()
            best_b1 = model.b1.copy()
            best_W2 = model.W2.copy()
            best_b2 = model.b2.copy()
        
        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch} "
                  f"(best was epoch {best_epoch}, val loss {best_val_loss:.4f})")
            break
    
    model.W1 = best_W1
    model.b1 = best_b1
    model.W2 = best_W2
    model.b2 = best_b2
    
    return train_losses, val_losses, best_epoch
```
</details>

**Test it:**

```python
model_es = MLP(n_input=2, n_hidden=200, n_classes=2, seed=42)
t_es, v_es, best_ep = train_with_early_stopping(
    model_es, X_tr, y_tr, X_va, y_va,
    lr=1.0, max_epochs=3000, patience=100
)

test_acc_es = np.mean(model_es.predict(X_te) == y_te_l) * 100
print(f"Test accuracy (no early stop):   {test_acc:.1f}%")
print(f"Test accuracy (early stopping):  {test_acc_es:.1f}%")
```

### Phase 5 — Three-Way Comparison

**Task:** Train a **right-sized model** (10 hidden neurons, 3000 epochs, no early stopping) and produce a 1×3 figure comparing the learning curves of all three approaches. Add a vertical red dashed line at `best_ep` on the early-stopping panel. Print test accuracy for each in the panel title.

```python
# TODO: Train the small model
model_small = MLP(n_input=2, n_hidden=___, n_classes=2, seed=42)
t_small, v_small = train_with_tracking(model_small, X_tr, y_tr, X_va, y_va,
                                        lr=1.0, n_epochs=___)
test_acc_small = np.mean(model_small.predict(X_te) == y_te_l) * 100

# TODO: Create 1×3 figure
# Panel 1: "200 neurons, no reg"     — t_losses vs v_losses
# Panel 2: "200 neurons, early stop" — t_es vs v_es, vertical line at best_ep
# Panel 3: "10 neurons"              — t_small vs v_small
# Title each panel with its test accuracy.
fig, axes = plt.subplots(___, ___, figsize=(___))
___

plt.tight_layout()
plt.show()
```

<details>
<summary>Solution</summary>

```python
model_small = MLP(n_input=2, n_hidden=10, n_classes=2, seed=42)
t_small, v_small = train_with_tracking(model_small, X_tr, y_tr, X_va, y_va,
                                        lr=1.0, n_epochs=3000)
test_acc_small = np.mean(model_small.predict(X_te) == y_te_l) * 100

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

data = [
    (t_losses, v_losses, f'200 neurons, no reg (Test: {test_acc:.1f}%)'),
    (t_es, v_es, f'200 neurons, early stop (Test: {test_acc_es:.1f}%)'),
    (t_small, v_small, f'10 neurons (Test: {test_acc_small:.1f}%)'),
]

for ax, (tl, vl, title) in zip(axes, data):
    ax.plot(tl, label='Train', linewidth=2)
    ax.plot(vl, label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

axes[1].axvline(x=best_ep, color='red', linestyle='--', linewidth=2, label=f'Stop @ {best_ep}')
axes[1].legend()

plt.suptitle('Model Comparison', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
```
</details>

**Write your conclusions:**
- Which model generalizes best?
- Is a big model with early stopping better or worse than a right-sized model?
- When would you prefer one approach over the other?

---

## 8. Mini-Project B: Regularization Showdown {#project-b}

### 🎯 Goal

**Implement** L2 regularization and dropout inside an MLP, then run a controlled experiment to determine which technique works best.

**Skills reused:** MLP backpropagation (Session 6), training pipeline (Project A).  
**New skills:** L2 gradient penalty, inverted dropout, controlled experiments.

---

### Phase 1 — Implement L2 Regularization

**Task:** Extend the MLP with an L2 penalty. You need to change **3 things** compared to the base MLP: the `loss` method, and two lines in `backward` (one for each weight matrix).

Start from this skeleton — the `___` blanks are yours to fill:

```python
class RegularizedMLP:
    """MLP with L2 regularization and dropout."""
    
    def __init__(self, n_input, n_hidden, n_classes,
                 l2_lambda=0.0, dropout_rate=0.0, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(n_hidden, n_input) * np.sqrt(2.0 / n_input)
        self.b1 = np.zeros((n_hidden, 1))
        self.W2 = np.random.randn(n_classes, n_hidden) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros((n_classes, 1))
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
    
    def forward(self, X, training=True):
        self.X = X
        self.z1 = self.W1 @ X + self.b1
        self.a1 = np.maximum(0, self.z1)
        
        # Phase 2 will add dropout here
        
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = softmax(self.z2)
        return self.a2
    
    def loss(self, y_true):
        data_loss = categorical_cross_entropy(y_true, self.a2)
        # TODO: Compute L2 penalty = (lambda / 2) * (||W1||² + ||W2||²)
        # Remember: ||W||² = sum of all squared elements
        l2_penalty = ___
        return data_loss + l2_penalty
    
    def data_loss(self, y_true):
        """Loss WITHOUT regularization penalty — for fair comparison."""
        return categorical_cross_entropy(y_true, self.a2)
    
    def backward(self, X, y_true, lr):
        N = X.shape[1]
        delta2 = (self.a2 - y_true) / N
        
        # TODO: Weight gradient for W2 = (data gradient) + (L2 gradient)
        # Recall from Part I: ∂L_total/∂W = ∂L_data/∂W + lambda * W
        dW2 = delta2 @ self.a1.T + ___
        db2 = np.sum(delta2, axis=1, keepdims=True)    # Biases: no L2
        
        delta1 = (self.W2.T @ delta2) * (self.z1 > 0).astype(float)
        
        # TODO: Same for W1
        dW1 = delta1 @ X.T + ___
        db1 = np.sum(delta1, axis=1, keepdims=True)    # Biases: no L2
        
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
    
    def predict(self, X):
        self.forward(X, training=False)
        return np.argmax(self.a2, axis=0)
```

<details>
<summary>Solution — the 3 changed lines</summary>

```python
# In loss():
l2_penalty = (self.l2_lambda / 2) * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))

# In backward():
dW2 = delta2 @ self.a1.T + self.l2_lambda * self.W2
dW1 = delta1 @ X.T + self.l2_lambda * self.W1
```

We do **not** regularize biases — only weights.
</details>

**Verify:** With `l2_lambda=0.0`, the model should behave identically to the base MLP.

```python
# Sanity check: L2=0 should match base MLP
np.random.seed(42)
model_base = MLP(2, 10, 2, seed=42)
model_reg = RegularizedMLP(2, 10, 2, l2_lambda=0.0, seed=42)

X_test_input = np.random.randn(2, 5)
y_test_input = np.zeros((2, 5)); y_test_input[0] = 1

model_base.forward(X_test_input)
model_reg.forward(X_test_input)

print(f"Outputs match: {np.allclose(model_base.a2, model_reg.a2)}")  # Should be True
```

### Phase 2 — Implement Dropout

**Task:** Add inverted dropout to the `forward` method. You need to:
1. Generate a random binary mask during training (keep each neuron with probability $1-p$)
2. Zero out dropped neurons and scale survivors by $\frac{1}{1-p}$
3. Do **nothing** during inference (`training=False`)

```python
# Replace the "Phase 2 will add dropout here" comment in forward() with:

    # Dropout (inverted)
    if training and self.dropout_rate > 0:
        # TODO: Create binary mask — each entry is 1 with prob (1 - dropout_rate), 0 otherwise
        # Hint: np.random.rand(*shape) gives uniform [0, 1); compare with dropout_rate
        self.mask = ___
        
        # TODO: Apply mask and scale to maintain expected value
        self.a1 = ___
```

<details>
<summary>Solution</summary>

```python
    if training and self.dropout_rate > 0:
        self.mask = (np.random.rand(*self.a1.shape) > self.dropout_rate).astype(float)
        self.a1 = self.a1 * self.mask / (1 - self.dropout_rate)
```

**Why `> dropout_rate`?** If `dropout_rate = 0.3`, we want to **keep** 70% of neurons, so we keep entries where `rand > 0.3`.

**Why divide by `(1 - dropout_rate)`?** Zeroing out 30% of neurons reduces the total signal by 30%. Scaling the survivors by $1/0.7$ compensates, so the expected activation stays the same. At test time we use all neurons — no scaling needed.
</details>

**Verify your dropout:**

```python
np.random.seed(99)
test_model = RegularizedMLP(2, 20, 2, dropout_rate=0.5)
test_X = np.random.randn(2, 1)

# With dropout
test_model.forward(test_X, training=True)
n_zero = np.sum(test_model.a1 == 0)
n_total = test_model.a1.size
print(f"Zeroed: {n_zero}/{n_total} ({n_zero/n_total*100:.0f}%) — expect ~50%")

# Without dropout (all neurons active)
test_model.forward(test_X, training=False)
n_zero_inf = np.sum(test_model.a1 == 0)
print(f"Zeroed at inference: {n_zero_inf}/{n_total} — expect only ReLU zeros")
```

### Phase 3 — Write the Experiment Loop

**Task:** Train 4 configurations on the spiral dataset and collect results. You must handle the subtlety that **training uses dropout** but **loss recording must not** (or the curves will be noisy).

```python
X_all, y_all_oh, y_all_lbl = generate_spiral(n_per_class=100, noise=0.3)
X_tr, y_tr, X_va, y_va, X_te, y_te = train_val_test_split(X_all, y_all_oh)
y_tr_l, y_va_l, y_te_l = np.argmax(y_tr, 0), np.argmax(y_va, 0), np.argmax(y_te, 0)

configs = [
    {"name": "No reg",          "l2": 0.0,   "drop": 0.0},
    {"name": "L2 (λ=0.01)",     "l2": 0.01,  "drop": 0.0},
    {"name": "Dropout (p=0.3)", "l2": 0.0,   "drop": 0.3},
    {"name": "L2 + Dropout",    "l2": 0.005, "drop": 0.2},
]

results = {}

for cfg in configs:
    # TODO: Create model, train for 4000 epochs (lr=0.8)
    # CAREFUL: 
    #   - Call forward(X_tr, training=True) + backward() for the training step
    #   - Call forward(X_tr, training=False) + data_loss() for recording train loss
    #   - Call forward(X_va, training=False) + data_loss() for recording val loss
    # Store: train_losses, val_losses, test_acc, model
    
    model = RegularizedMLP(n_input=2, n_hidden=100, n_classes=3,
                           l2_lambda=cfg["l2"], dropout_rate=cfg["drop"], seed=42)
    t_hist, v_hist = [], []
    
    for epoch in range(4000):
        ___  # training step
        
        ___  # record clean losses
    
    test_acc = np.mean(model.predict(X_te) == y_te_l) * 100
    results[cfg["name"]] = {
        "train_loss": t_hist, "val_loss": v_hist,
        "test_acc": test_acc, "model": model
    }
    print(f"{cfg['name']:>18s}: Test Acc = {test_acc:.1f}%")
```

<details>
<summary>Solution</summary>

```python
for cfg in configs:
    model = RegularizedMLP(n_input=2, n_hidden=100, n_classes=3,
                           l2_lambda=cfg["l2"], dropout_rate=cfg["drop"], seed=42)
    t_hist, v_hist = [], []
    
    for epoch in range(4000):
        # Train with dropout active
        model.forward(X_tr, training=True)
        model.backward(X_tr, y_tr, lr=0.8)
        
        # Record clean losses (no dropout noise)
        model.forward(X_tr, training=False)
        t_hist.append(model.data_loss(y_tr))
        model.forward(X_va, training=False)
        v_hist.append(model.data_loss(y_va))
    
    test_acc = np.mean(model.predict(X_te) == y_te_l) * 100
    results[cfg["name"]] = {
        "train_loss": t_hist, "val_loss": v_hist,
        "test_acc": test_acc, "model": model
    }
    print(f"{cfg['name']:>18s}: Test Acc = {test_acc:.1f}%")
```
</details>

### Phase 4 — Build the Comparison Dashboard

**Task:** Create a **3×4 figure** (3 rows, 4 columns — one column per config):
- **Row 1:** Learning curves (train + val loss)
- **Row 2:** Decision boundary on test data (contourf + scatter)
- **Row 3:** Histogram of all weights (W1 and W2 concatenated, 50 bins, xlim ±3)

Title each column with config name + test accuracy. Title each row 3 panel with the weight standard deviation.

```python
fig, axes = plt.subplots(3, 4, figsize=(22, 14))

# Pre-compute the decision boundary grid (shared across all panels)
xx, yy = np.meshgrid(
    np.linspace(X_all[0].min()-0.3, X_all[0].max()+0.3, 200),
    np.linspace(X_all[1].min()-0.3, X_all[1].max()+0.3, 200))
grid = np.vstack([xx.ravel(), yy.ravel()])

for col, (name, res) in enumerate(results.items()):
    # TODO — Row 1: Learning curves
    ax = axes[0, col]
    ___
    
    # TODO — Row 2: Decision boundary with test points
    ax = axes[1, col]
    ___
    
    # TODO — Row 3: Weight histogram
    ax = axes[2, col]
    ___

plt.suptitle('Regularization Showdown — Spiral (100 hidden neurons)', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
```

<details>
<summary>Solution</summary>

```python
fig, axes = plt.subplots(3, 4, figsize=(22, 14))

xx, yy = np.meshgrid(
    np.linspace(X_all[0].min()-0.3, X_all[0].max()+0.3, 200),
    np.linspace(X_all[1].min()-0.3, X_all[1].max()+0.3, 200))
grid = np.vstack([xx.ravel(), yy.ravel()])

for col, (name, res) in enumerate(results.items()):
    # Row 1: Learning curves
    ax = axes[0, col]
    ax.plot(res["train_loss"], label='Train', linewidth=1.5)
    ax.plot(res["val_loss"], label='Val', linewidth=1.5)
    ax.set_title(f'{name}\nTest: {res["test_acc"]:.1f}%', fontsize=13)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('CCE Loss')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Row 2: Decision boundary
    ax = axes[1, col]
    Z = res["model"].predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5],
                colors=['#ADD8E6', '#FFCCCB', '#90EE90'], alpha=0.4)
    for k, c in enumerate(['blue', 'red', 'green']):
        mask = y_te_l == k
        ax.scatter(X_te[0, mask], X_te[1, mask], c=c, edgecolors='black', s=25, alpha=0.8)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    
    # Row 3: Weight histogram
    ax = axes[2, col]
    all_w = np.concatenate([res["model"].W1.flatten(), res["model"].W2.flatten()])
    ax.hist(all_w, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.set_title(f'Weight std = {np.std(all_w):.3f}', fontsize=12)
    ax.set_xlabel('Weight value')
    ax.set_xlim(-3, 3)

plt.suptitle('Regularization Showdown — Spiral (100 hidden neurons)', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
```
</details>

### Phase 5 — Analysis Report

Answer in your notebook (3–5 sentences each):

1. Which configuration has the **smallest train-val gap**? What does this mean?
2. Compare the decision boundaries: which are **smooth** vs **jagged**? How does this relate to generalization?
3. How did L2 change the weight distribution compared to "no reg"? What about dropout?
4. Which configuration achieved the **best test accuracy**?
5. **Bonus experiment:** Change L2 to `l2_lambda=0.1` (10× larger). Re-run, observe, and explain what happens.

---

## 9. Mini-Project C: Optimizer Olympics {#project-c}

### 🎯 Goal

**Implement** SGD with Momentum and Adam optimizers from scratch, then race them on the spiral dataset and a challenging 2D function.

**Skills reused:** Gradient descent (Session 5), MLP backprop (Session 6), training pipeline (Project A).  
**New skills:** Momentum, adaptive learning rates, optimizer abstraction.

---

### Phase 1 — Refactor the MLP for External Optimizers

**Task:** Modify the MLP to **return gradients** instead of applying updates internally. This decouples the model from the optimizer.

```python
class FlexMLP:
    """MLP that returns gradients — optimizer is external."""
    
    def __init__(self, n_input, n_hidden, n_classes, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(n_hidden, n_input) * np.sqrt(2.0 / n_input)
        self.b1 = np.zeros((n_hidden, 1))
        self.W2 = np.random.randn(n_classes, n_hidden) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros((n_classes, 1))
    
    def forward(self, X):
        self.X = X
        self.z1 = self.W1 @ X + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = softmax(self.z2)
        return self.a2
    
    def loss(self, y_true):
        return categorical_cross_entropy(y_true, self.a2)
    
    def compute_gradients(self, y_true):
        """
        Compute and RETURN gradients — do NOT update any weights.
        
        Returns: [dW1, db1, dW2, db2] — same order as self.params
        """
        N = self.X.shape[1]
        
        # TODO: Backprop math from Session 6 — but return gradients instead of updating
        delta2 = ___
        dW2 = ___
        db2 = ___
        delta1 = ___
        dW1 = ___
        db1 = ___
        
        return [dW1, db1, dW2, db2]
    
    @property
    def params(self):
        """Parameter list — same order as compute_gradients returns."""
        return [self.W1, self.b1, self.W2, self.b2]
    
    def predict(self, X):
        self.forward(X)
        return np.argmax(self.a2, axis=0)
```

<details>
<summary>Solution</summary>

```python
def compute_gradients(self, y_true):
    N = self.X.shape[1]
    delta2 = (self.a2 - y_true) / N
    dW2 = delta2 @ self.a1.T
    db2 = np.sum(delta2, axis=1, keepdims=True)
    delta1 = (self.W2.T @ delta2) * (self.z1 > 0).astype(float)
    dW1 = delta1 @ self.X.T
    db1 = np.sum(delta1, axis=1, keepdims=True)
    return [dW1, db1, dW2, db2]
```
</details>

### Phase 2 — Implement Three Optimizers

**Vanilla SGD** is given as a reference. **You implement Momentum and Adam.**

```python
class SGD:
    """Vanilla SGD — given as reference."""
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def step(self, params, grads):
        """params and grads: lists of arrays, same length and order."""
        for p, g in zip(params, grads):
            p -= self.lr * g
```

**Task:** Implement SGD with Momentum.

```python
class SGDMomentum:
    """
    SGD with momentum.
    
    Formulas:
        v_i ← beta * v_i + (1 - beta) * g_i
        p_i ← p_i - lr * v_i
    """
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocities = None   # Will be initialized on first call
    
    def step(self, params, grads):
        # Initialize velocities to zeros on first call
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]
        
        for i, (p, g) in enumerate(zip(params, grads)):
            # TODO: Update velocity (exponential moving average of gradients)
            self.velocities[i] = ___
            
            # TODO: Update parameter using velocity
            p -= ___
```

<details>
<summary>Solution — Momentum</summary>

```python
for i, (p, g) in enumerate(zip(params, grads)):
    self.velocities[i] = self.beta * self.velocities[i] + (1 - self.beta) * g
    p -= self.lr * self.velocities[i]
```
</details>

**Task:** Implement Adam. This is harder — you need first moments, second moments, bias correction, and a time step counter.

```python
class Adam:
    """
    Adam optimizer.
    
    Formulas:
        m_i ← β₁ * m_i + (1 - β₁) * g_i            (first moment)
        v_i ← β₂ * v_i + (1 - β₂) * g_i²            (second moment)
        m̂_i = m_i / (1 - β₁^t)                       (bias correction)
        v̂_i = v_i / (1 - β₂^t)                       (bias correction)
        p_i ← p_i - lr * m̂_i / (√v̂_i + ε)
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None      # First moment estimates
        self.v = None      # Second moment estimates
        self.t = 0         # Time step counter
    
    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        self.t += 1
        
        for i, (p, g) in enumerate(zip(params, grads)):
            # TODO: Update biased first moment estimate (m)
            self.m[i] = ___
            
            # TODO: Update biased second moment estimate (v)
            # Note: element-wise square of gradient
            self.v[i] = ___
            
            # TODO: Compute bias-corrected estimates (m_hat, v_hat)
            m_hat = ___
            v_hat = ___
            
            # TODO: Update parameter
            p -= ___
```

<details>
<summary>Solution — Adam</summary>

```python
for i, (p, g) in enumerate(zip(params, grads)):
    self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
    self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
    
    m_hat = self.m[i] / (1 - self.beta1 ** self.t)
    v_hat = self.v[i] / (1 - self.beta2 ** self.t)
    
    p -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
```
</details>

### Phase 3 — Race on the Spiral Dataset

**Task:** Write the training loop that trains the **same architecture** with each optimizer. Each optimizer gets a **fresh model from the same seed** for fair comparison.

```python
X_all, y_all_oh, y_all_lbl = generate_spiral(n_per_class=150, noise=0.25)
X_tr, y_tr, X_va, y_va, X_te, y_te = train_val_test_split(X_all, y_all_oh)
y_te_l = np.argmax(y_te, axis=0)

# TODO: Define three optimizer instances
# SGD:      lr=1.0
# Momentum: lr=1.0, beta=0.9
# Adam:     lr=0.01  (Adam typically uses smaller lr)
optimizer_configs = [
    ("SGD (lr=1.0)",      ___),
    ("Momentum (lr=1.0)", ___),
    ("Adam (lr=0.01)",    ___),
]

race_results = {}

for name, optimizer in optimizer_configs:
    # TODO:
    # 1. Create a FRESH FlexMLP (seed=42 for all — fair comparison)
    # 2. Train for 3000 epochs:
    #    a. Forward pass
    #    b. Record train loss
    #    c. Compute gradients (NOT weight update!)
    #    d. optimizer.step(params, grads) — this applies the update
    #    e. Forward on val, record val loss
    # 3. Compute test accuracy
    # 4. Store results

    model = ___
    t_hist, v_hist = [], []
    
    for epoch in range(3000):
        ___
    
    test_acc = ___
    race_results[name] = {"train": t_hist, "val": v_hist, "test_acc": test_acc}
    print(f"{name:>25s}: Test Acc = {test_acc:.1f}%")
```

<details>
<summary>Solution</summary>

```python
optimizer_configs = [
    ("SGD (lr=1.0)",      SGD(lr=1.0)),
    ("Momentum (lr=1.0)", SGDMomentum(lr=1.0, beta=0.9)),
    ("Adam (lr=0.01)",    Adam(lr=0.01)),
]

race_results = {}

for name, optimizer in optimizer_configs:
    model = FlexMLP(n_input=2, n_hidden=50, n_classes=3, seed=42)
    t_hist, v_hist = [], []
    
    for epoch in range(3000):
        model.forward(X_tr)
        t_hist.append(model.loss(y_tr))
        grads = model.compute_gradients(y_tr)
        optimizer.step(model.params, grads)
        
        model.forward(X_va)
        v_hist.append(model.loss(y_va))
    
    test_acc = np.mean(model.predict(X_te) == y_te_l) * 100
    race_results[name] = {"train": t_hist, "val": v_hist, "test_acc": test_acc}
    print(f"{name:>25s}: Test Acc = {test_acc:.1f}%")
```
</details>

### Phase 4 — Visualize the Race

**Task:** Create a 1×3 figure:
1. All training losses overlaid (compare convergence speed)
2. Zoom on first 500 epochs (where differences are most visible)
3. Bar chart of final test accuracy (with value labels on each bar)

```python
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# TODO: Panel 1 — all training losses, full range
ax = axes[0]
___

# TODO: Panel 2 — zoom first 500 epochs
ax = axes[1]
___

# TODO: Panel 3 — bar chart with accuracy labels above each bar
ax = axes[2]
___

plt.tight_layout()
plt.show()
```

<details>
<summary>Solution</summary>

```python
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Panel 1: Full range
ax = axes[0]
for (name, res), c in zip(race_results.items(), colors):
    ax.plot(res["train"], label=name, linewidth=2, color=c, alpha=0.8)
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Train Loss', fontsize=14)
ax.set_title('Training Convergence', fontsize=16)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 2: Zoom
ax = axes[1]
for (name, res), c in zip(race_results.items(), colors):
    ax.plot(res["train"][:500], label=name, linewidth=2, color=c, alpha=0.8)
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Train Loss', fontsize=14)
ax.set_title('First 500 Epochs', fontsize=16)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 3: Bar chart
ax = axes[2]
names = list(race_results.keys())
accs = [race_results[n]["test_acc"] for n in names]
bars = ax.bar(range(len(names)), accs, color=colors, edgecolor='black', alpha=0.8)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.split('(')[0].strip() for n in names], fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=14)
ax.set_title('Final Test Accuracy', fontsize=16)
ax.set_ylim(0, 100)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{acc:.1f}%', ha='center', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```
</details>

### Phase 5 — The Gauntlet: Rosenbrock Function

The Rosenbrock function has a narrow curved valley — vanilla SGD struggles while momentum and Adam navigate it well. This makes the difference between optimizers dramatically visible.

```python
def rosenbrock(w1, w2):
    """Famous test function: minimum at (1, 1), narrow curved valley."""
    return (1 - w1) ** 2 + 100 * (w2 - w1 ** 2) ** 2

def rosenbrock_grad(w1, w2):
    """Analytical gradient of Rosenbrock."""
    dw1 = -2 * (1 - w1) + 200 * (w2 - w1 ** 2) * (-2 * w1)
    dw2 = 200 * (w2 - w1 ** 2)
    return np.array([dw1]), np.array([dw2])
```

**Task:** Run each optimizer for 2000 steps starting from $(-1, -1)$, record the trajectory, then plot all three on a contour plot of the function.

Learning rates: SGD → 0.001, Momentum → 0.001, Adam → 0.05  
(Adam can use a much larger lr because it normalizes by gradient variance)

```python
start = (-1.0, -1.0)
trajectories = {}

for name, opt in [("SGD", SGD(lr=0.001)),
                   ("Momentum", SGDMomentum(lr=0.001, beta=0.9)),
                   ("Adam", Adam(lr=0.05))]:
    w1, w2 = np.array([start[0]]), np.array([start[1]])
    path = [(w1[0], w2[0])]
    
    for _ in range(2000):
        # TODO: Compute gradient and take one optimizer step
        # Hint: rosenbrock_grad returns (dw1, dw2) as arrays
        # Then opt.step([w1, w2], [dw1, dw2])
        ___
        
        path.append((w1[0], w2[0]))
    
    trajectories[name] = path
    print(f"{name:>10s}: final = ({w1[0]:.4f}, {w2[0]:.4f}), "
          f"loss = {rosenbrock(w1[0], w2[0]):.6f}")

# TODO: Create contour plot with all three trajectories
# - Compute Z = rosenbrock(W1g, W2g) on a meshgrid
# - Use ax.contour with np.log(Z + 1) for level spacing
# - Mark minimum (1,1) with a gold star, start (-1,-1) with a black X
# - Draw each trajectory as a colored line
# - Add legend, labels, title

fig, ax = plt.subplots(figsize=(10, 8))
___
plt.show()
```

<details>
<summary>Solution</summary>

```python
for name, opt in [("SGD", SGD(lr=0.001)),
                   ("Momentum", SGDMomentum(lr=0.001, beta=0.9)),
                   ("Adam", Adam(lr=0.05))]:
    w1, w2 = np.array([start[0]]), np.array([start[1]])
    path = [(w1[0], w2[0])]
    
    for _ in range(2000):
        g1, g2 = rosenbrock_grad(w1[0], w2[0])
        opt.step([w1, w2], [g1, g2])
        path.append((w1[0], w2[0]))
    
    trajectories[name] = path
    print(f"{name:>10s}: final = ({w1[0]:.4f}, {w2[0]:.4f}), "
          f"loss = {rosenbrock(w1[0], w2[0]):.6f}")

fig, ax = plt.subplots(figsize=(10, 8))

w1_range = np.linspace(-2, 2, 300)
w2_range = np.linspace(-1.5, 3, 300)
W1g, W2g = np.meshgrid(w1_range, w2_range)
Z = rosenbrock(W1g, W2g)

ax.contour(W1g, W2g, np.log(Z + 1), levels=30, cmap='viridis', alpha=0.5)
ax.scatter(1, 1, color='gold', s=200, marker='*', zorder=10, label='Minimum (1,1)')
ax.scatter(start[0], start[1], color='black', s=100, marker='x', zorder=10, label='Start')

for (name, path), c in zip(trajectories.items(), colors):
    xs, ys = zip(*path)
    ax.plot(xs, ys, '-', linewidth=1.5, color=c, alpha=0.8, label=name)
    ax.scatter(xs[-1], ys[-1], color=c, s=80, edgecolors='black', zorder=8)

ax.set_xlabel('$w_1$', fontsize=14)
ax.set_ylabel('$w_2$', fontsize=14)
ax.set_title('Optimizer Trajectories on Rosenbrock Function', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.show()
```
</details>

### Phase 6 — Analysis Report

Answer in your notebook:

1. **Spiral dataset:** Which optimizer converged fastest (by epoch)? Which reached the best test accuracy?
2. **Rosenbrock:** Why does vanilla SGD barely move in 2000 steps? What does momentum do differently?
3. **Adam's trade-off:** Adam navigated the valley faster. But compare final loss values — which optimizer got closest to the true minimum $(1, 1)$?
4. **Practical takeaway:** You're starting a new project tomorrow. Which optimizer do you pick first, and with what learning rate?

---

## Summary

### What We Learned

✅ **Generalization**: Training accuracy ≠ real-world performance  
✅ **Train/Val/Test splits**: The honest way to evaluate models  
✅ **Learning curves**: The diagnostic tool for overfitting and underfitting  
✅ **Early stopping**: Stop when validation loss increases, restore best weights  
✅ **L2 regularization**: Penalize large weights → smoother boundaries  
✅ **Dropout**: Randomly disable neurons → implicit ensemble  
✅ **Modern optimizers**: Momentum smooths oscillations, Adam adapts per-weight

### Key Insights

1. **The overfitting recipe:**
   - Big model + small dataset + long training = memorization
   - Detect with learning curves (train-val gap)
   - Cure with: early stopping, regularization, more data, simpler model

2. **Regularization works by constraining complexity:**
   - L2 keeps weights small → smooth decision boundaries
   - Dropout prevents co-adaptation → robust features
   - They can be combined for stronger effect

3. **Optimizer choice matters:**
   - Adam is the safe default (fast, adaptive, forgiving)
   - SGD + Momentum can generalize better with careful tuning
   - Always compare on a validation set

### What's Next?

**Session 9: PyTorch Introduction**

In the next session, we'll learn:
- **Tensors & autograd**: Automatic differentiation (no more manual backprop!)
- **nn.Module**: Build networks declaratively
- **Training loop**: Optimizers, loss functions, and datasets — the PyTorch way
- **Rebuild**: Reimplement our MLP in PyTorch and compare

**The goal:** Transition from "understanding the math" to "using professional tools"!

### Before Next Session

**Think about:**
1. We manually implemented backpropagation, gradient descent, L2, dropout, momentum, and Adam. What parts were tedious and error-prone?
2. If a library could handle gradients automatically, what would you still need to implement yourself?
3. Install PyTorch: `pip install torch` (or see https://pytorch.org)

**Optional reading:**
- PyTorch "60 Minute Blitz" tutorial: https://pytorch.org/tutorials/
- "Why Momentum Really Works" — distill.pub

---

**End of Session 8** 🎓

**You now understand:**
- ✅ How to detect and diagnose overfitting
- ✅ How regularization prevents memorization
- ✅ How modern optimizers improve training

**Next up:** PyTorch — letting the framework do the heavy lifting! 🚀