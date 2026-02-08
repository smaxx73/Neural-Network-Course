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
    
    # Training error
    y_pred_train = np.polyval(coeffs, x_train)
    train_err = np.mean((y_train - y_pred_train) ** 2)
    
    # "Test" error on the true function
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

### Implementation

```python
def train_val_test_split(X, y, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split data into train, validation, and test sets.
    
    X shape: (n_features, N)
    y shape: (n_classes, N) or (1, N)
    """
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

```python
def train_with_early_stopping(model, X_train, y_train, X_val, y_val,
                               lr, max_epochs, patience=50):
    """
    Stop training when validation loss hasn't improved for `patience` epochs.
    Returns the best model weights.
    """
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_epoch = 0
    # Save best weights
    best_W1, best_b1 = model.W1.copy(), model.b1.copy()
    best_W2, best_b2 = model.W2.copy(), model.b2.copy()
    
    for epoch in range(max_epochs):
        # Train
        model.forward(X_train)
        train_loss = model.compute_loss(y_train)
        model.backward(y_train, lr)
        train_losses.append(train_loss)
        
        # Validate (forward only, no weight updates!)
        model.forward(X_val)
        val_loss = model.compute_loss(y_val)
        val_losses.append(val_loss)
        
        # Check improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_W1, best_b1 = model.W1.copy(), model.b1.copy()
            best_W2, best_b2 = model.W2.copy(), model.b2.copy()
        
        # Early stopping check
        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch} (best was epoch {best_epoch})")
            break
    
    # Restore best weights
    model.W1, model.b1 = best_W1, best_b1
    model.W2, model.b2 = best_W2, best_b2
    
    return train_losses, val_losses, best_epoch
```

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
W^{(l)} \leftarrow W^{(l)} - \eta \left( \frac{\partial L_{\text{data}}}{\partial W^{(l)}} + \lambda W^{(l)} \right) = (1 - \eta\lambda) W^{(l)} - \eta \frac{\partial L_{\text{data}}}{\partial W^{(l)}}
$$

That factor $(1 - \eta\lambda)$ **shrinks** the weights at every step — hence the name **weight decay**.

**Intuition:** Large weights create sharp, complex decision boundaries. Penalizing large weights encourages smoother, simpler boundaries that generalize better.

### L1 Regularization (Sparsity)

$$
L_{\text{total}} = L_{\text{data}} + \lambda \sum_l \| W^{(l)} \|_1
$$

Where $\| W \|_1 = \sum_{ij} |w_{ij}|$.

**Difference from L2:** L1 drives weights to **exactly zero**, creating sparse networks. L2 shrinks weights toward zero but rarely makes them exactly zero.

| | L1 | L2 |
|---|---|---|
| **Penalty** | Sum of $|w|$ | Sum of $w^2$ |
| **Effect** | Sparse weights (feature selection) | Small weights (smooth boundaries) |
| **Gradient** | $\lambda \cdot \text{sign}(w)$ | $\lambda \cdot w$ |
| **Most common** | When you suspect irrelevant features | General-purpose regularization |

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
- At test time: use all neurons, but scale activations by $(1-p)$

**Implementation sketch:**

```python
# During training:
mask = (np.random.rand(*a1.shape) > dropout_rate).astype(float)
a1_dropped = a1 * mask / (1 - dropout_rate)  # Scale to maintain expected value

# During inference:
a1_dropped = a1  # Use all neurons, no scaling needed (inverted dropout)
```

---

## 6. Modern Optimizers {#optimizers}

### Beyond Vanilla SGD

Plain gradient descent has limitations: it uses the same learning rate for all weights, and can oscillate in narrow valleys. Modern optimizers fix this.

### SGD with Momentum

**Idea:** Accumulate a "velocity" — like a ball rolling downhill with inertia.

$$
v \leftarrow \beta v + (1 - \beta) \nabla L
$$
$$
w \leftarrow w - \eta v
$$

Where $\beta \approx 0.9$ is the momentum coefficient.

**Effect:** Smooths out oscillations and accelerates through consistent gradients.

```
  Without momentum:              With momentum:
  ╱╲╱╲╱╲╱╲                      ╲
                ──→ slow              ╲
  zigzag path                          ╲──→ smooth, fast
```

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

Where $\hat{m}$ and $\hat{v}$ are bias-corrected estimates.

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

All mini-projects reuse and extend code from Sessions 5–7. Here is the shared codebase. **Copy this at the top of your notebook.**

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

def train_val_test_split(X, y, val_ratio=0.15, test_ratio=0.15, seed=42):
    np.random.seed(seed)
    N = X.shape[1]
    idx = np.random.permutation(N)
    n_test = int(N * test_ratio)
    n_val = int(N * val_ratio)
    i1, i2 = N - n_val - n_test, N - n_test
    return (X[:, idx[:i1]], y[:, idx[:i1]],
            X[:, idx[i1:i2]], y[:, idx[i1:i2]],
            X[:, idx[i2:]], y[:, idx[i2:]])

# ─── Datasets ───

def generate_moons(n_samples=500, noise=0.2, seed=42):
    """Two interleaving half-circles — a classic nonlinear binary dataset."""
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

Build a model that **overfits**, learn to **diagnose** it, then **cure** it.

**Skills reused:** MLP forward/backward (Session 6), CCE loss + softmax (Session 7), evaluation metrics (Session 7), data splitting (new), learning curves (new).

---

### Phase 1 — Build the Baseline MLP

We start from the `MultiClassMLP` from Session 7 and add support for **separate train/val evaluation**:

```python
class MLP:
    """
    Multi-class MLP: input → hidden (ReLU) → output (softmax + CCE).
    Extended from Session 7 with train/val tracking.
    """
    
    def __init__(self, n_input, n_hidden, n_classes, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(n_hidden, n_input) * np.sqrt(2.0 / n_input)
        self.b1 = np.zeros((n_hidden, 1))
        self.W2 = np.random.randn(n_classes, n_hidden) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros((n_classes, 1))
    
    def forward(self, X, training=True):
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
        self.forward(X, training=False)
        return np.argmax(self.a2, axis=0)
```

### Phase 2 — Generate Data and Split

```python
# Generate moons dataset
X_all, y_all_oh, y_all_labels = generate_moons(n_samples=400, noise=0.2)

# Split: 70% train, 15% val, 15% test
X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
    X_all, y_all_oh, val_ratio=0.15, test_ratio=0.15
)

# Also keep label versions for accuracy/metrics
y_train_lbl = np.argmax(y_train, axis=0)
y_val_lbl = np.argmax(y_val, axis=0)
y_test_lbl = np.argmax(y_test, axis=0)

print(f"Train: {X_train.shape[1]}, Val: {X_val.shape[1]}, Test: {X_test.shape[1]}")
```

### Phase 3 — Train a Deliberately Overfitting Model

**Task:** Create a model that is **way too powerful** for this dataset and watch it overfit.

```python
# Overpowered model: 200 hidden neurons for a 2D dataset with 280 training points!
model_overfit = MLP(n_input=2, n_hidden=200, n_classes=2, seed=42)

train_losses, val_losses = [], []
n_epochs = 3000
lr = 1.0

for epoch in range(n_epochs):
    # Train step
    model_overfit.forward(X_train)
    t_loss = model_overfit.loss(y_train)
    model_overfit.backward(X_train, y_train, lr)
    train_losses.append(t_loss)
    
    # Validation step (forward only!)
    model_overfit.forward(X_val)
    v_loss = model_overfit.loss(y_val)
    val_losses.append(v_loss)

# Final metrics
train_acc = np.mean(model_overfit.predict(X_train) == y_train_lbl) * 100
val_acc = np.mean(model_overfit.predict(X_val) == y_val_lbl) * 100
test_acc = np.mean(model_overfit.predict(X_test) == y_test_lbl) * 100

print(f"Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% | Test Acc: {test_acc:.1f}%")
```

### Phase 4 — Diagnose with Learning Curves

**Task:** Plot the learning curves and answer the diagnostic questions below.

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Learning curves
ax = axes[0]
ax.plot(train_losses, label='Train loss', linewidth=2, color='blue')
ax.plot(val_losses, label='Validation loss', linewidth=2, color='orange')
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('CCE Loss', fontsize=14)
ax.set_title('Learning Curves — Overfitting Model', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Decision boundary
ax = axes[1]
xx, yy = np.meshgrid(np.linspace(X_all[0].min()-0.5, X_all[0].max()+0.5, 200),
                      np.linspace(X_all[1].min()-0.5, X_all[1].max()+0.5, 200))
grid = np.vstack([xx.ravel(), yy.ravel()])
Z = model_overfit.predict(grid).reshape(xx.shape)

ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=['#ADD8E6', '#FFCCCB'], alpha=0.4)
ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
ax.scatter(X_train[0], X_train[1], c=y_train_lbl, cmap='bwr', 
           edgecolors='black', s=40, alpha=0.7, label='Train')
ax.scatter(X_val[0], X_val[1], c=y_val_lbl, cmap='bwr', 
           edgecolors='black', s=40, marker='s', alpha=0.7, label='Val')
ax.set_title('Decision Boundary', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Diagnostic questions** (discuss with your neighbor or write in your notebook):

1. At which epoch does the validation loss start diverging from the training loss?
2. Look at the decision boundary — is it smooth or jagged? What does that tell you?
3. What is the gap between train accuracy and test accuracy? Is this acceptable?

### Phase 5 — Cure it with Early Stopping

**Task:** Re-train the same architecture, but use `train_with_early_stopping` (from Section 4). Plot the new learning curves with a vertical line at the early stopping epoch.

```python
model_es = MLP(n_input=2, n_hidden=200, n_classes=2, seed=42)
t_losses, v_losses, best_ep = train_with_early_stopping(
    model_es, X_train, y_train, X_val, y_val,
    lr=1.0, max_epochs=3000, patience=100
)

# Compare
train_acc_es = np.mean(model_es.predict(X_train) == y_train_lbl) * 100
val_acc_es = np.mean(model_es.predict(X_val) == y_val_lbl) * 100
test_acc_es = np.mean(model_es.predict(X_test) == y_test_lbl) * 100

print(f"\nWithout early stopping: Train {train_acc:.1f}% | Val {val_acc:.1f}% | Test {test_acc:.1f}%")
print(f"With early stopping:    Train {train_acc_es:.1f}% | Val {val_acc_es:.1f}% | Test {test_acc_es:.1f}%")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t_losses, label='Train', linewidth=2, color='blue')
plt.plot(v_losses, label='Validation', linewidth=2, color='orange')
plt.axvline(x=best_ep, color='red', linestyle='--', linewidth=2, label=f'Early stop (epoch {best_ep})')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('CCE Loss', fontsize=14)
plt.title('Learning Curves with Early Stopping', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()
```

### Phase 6 — Compare with a Right-Sized Model

**Task:** Train a smaller model (e.g., 10 hidden neurons) for 3000 epochs. Does it overfit? Plot its learning curves alongside the big model's.

```python
model_small = MLP(n_input=2, n_hidden=10, n_classes=2, seed=42)
t_small, v_small = [], []

for epoch in range(3000):
    model_small.forward(X_train)
    t_small.append(model_small.loss(y_train))
    model_small.backward(X_train, y_train, lr=1.0)
    model_small.forward(X_val)
    v_small.append(model_small.loss(y_val))

test_acc_small = np.mean(model_small.predict(X_test) == y_test_lbl) * 100
print(f"Small model test accuracy: {test_acc_small:.1f}%")

# Compare all three
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, t_l, v_l, title in zip(axes,
    [train_losses, t_losses, t_small],
    [val_losses, v_losses, v_small],
    ['200 neurons (no reg)', '200 neurons (early stop)', '10 neurons']):
    ax.plot(t_l, label='Train', linewidth=2)
    ax.plot(v_l, label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Model Comparison: Learning Curves', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
```

**Write your conclusions:**
- Which model generalizes best?
- Is a big model with early stopping better or worse than a right-sized model?
- When would you prefer one approach over the other?

---

## 8. Mini-Project B: Regularization Showdown {#project-b}

### 🎯 Goal

Extend the MLP class with **L2 regularization** and **dropout**, then run a controlled experiment on the spiral dataset to determine which technique works best.

**Skills reused:** MLP class (Session 6–7), backpropagation math (Session 6), training loop (Session 5–6), evaluation metrics & confusion matrix (Session 7).

---

### Phase 1 — Build the Regularized MLP

**Task:** Extend the `MLP` class with L2 regularization and dropout. The key changes are in `backward` and `forward`.

```python
class RegularizedMLP:
    """
    MLP with L2 regularization and dropout.
    input → hidden (ReLU + dropout) → output (softmax + CCE)
    """
    
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
        self.z1 = self.W1 @ X + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        
        # Dropout (only during training!)
        if training and self.dropout_rate > 0:
            self.mask = (np.random.rand(*self.a1.shape) > self.dropout_rate).astype(float)
            self.a1 = self.a1 * self.mask / (1 - self.dropout_rate)  # Inverted dropout
        
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = softmax(self.z2)
        return self.a2
    
    def loss(self, y_true):
        data_loss = categorical_cross_entropy(y_true, self.a2)
        # L2 penalty
        l2_penalty = (self.l2_lambda / 2) * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return data_loss + l2_penalty
    
    def data_loss(self, y_true):
        """Loss without regularization penalty (for fair comparison)"""
        return categorical_cross_entropy(y_true, self.a2)
    
    def backward(self, X, y_true, lr):
        N = X.shape[1]
        delta2 = (self.a2 - y_true) / N
        dW2 = delta2 @ self.a1.T + self.l2_lambda * self.W2   # ← L2 gradient added
        db2 = np.sum(delta2, axis=1, keepdims=True)
        delta1 = (self.W2.T @ delta2) * (self.z1 > 0).astype(float)
        dW1 = delta1 @ X.T + self.l2_lambda * self.W1          # ← L2 gradient added
        db1 = np.sum(delta1, axis=1, keepdims=True)
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
    
    def predict(self, X):
        self.forward(X, training=False)
        return np.argmax(self.a2, axis=0)
```

### 🤔 Check Your Understanding

Before running experiments, answer these questions:

1. Why does the `forward` method take a `training` parameter?
2. Why do we divide by `(1 - dropout_rate)` in the dropout mask?
3. In `backward`, where exactly does L2 regularization enter? What would happen if we also regularized the biases?

<details>
<summary>Answers</summary>

1. **Dropout is only applied during training.** At test time, we use all neurons (with inverted scaling). If we dropped neurons during evaluation, the predictions would be noisy and unreliable.

2. **To keep the expected value unchanged.** If we zero out 30% of neurons, the surviving 70% must be scaled up by $1/0.7$ so that the total signal stays the same. This is **inverted dropout** — it means we don't need to change anything at test time.

3. **L2 enters as `+ self.l2_lambda * self.W`** added to each weight gradient. We typically don't regularize biases because: (a) biases don't interact with inputs, so they don't contribute to overfitting, and (b) regularizing biases can prevent the model from fitting the data mean.
</details>

### Phase 2 — The Experiment

**Task:** Run a controlled experiment comparing four configurations on the spiral dataset:

| Config | Hidden | L2 $\lambda$ | Dropout | Label |
|---|---|---|---|---|
| A | 100 | 0 | 0 | Baseline (no reg) |
| B | 100 | 0.01 | 0 | L2 only |
| C | 100 | 0 | 0.3 | Dropout only |
| D | 100 | 0.005 | 0.2 | L2 + Dropout |

```python
# Generate spiral and split
X_all, y_all_oh, y_all_lbl = generate_spiral(n_per_class=100, noise=0.3)
X_tr, y_tr, X_va, y_va, X_te, y_te = train_val_test_split(X_all, y_all_oh)
y_tr_l = np.argmax(y_tr, axis=0)
y_va_l = np.argmax(y_va, axis=0)
y_te_l = np.argmax(y_te, axis=0)

# Define configurations
configs = [
    {"name": "No reg",         "l2": 0.0,   "drop": 0.0},
    {"name": "L2 (λ=0.01)",    "l2": 0.01,  "drop": 0.0},
    {"name": "Dropout (p=0.3)","l2": 0.0,   "drop": 0.3},
    {"name": "L2+Dropout",     "l2": 0.005, "drop": 0.2},
]

results = {}
n_epochs = 4000
lr = 0.8

for cfg in configs:
    model = RegularizedMLP(n_input=2, n_hidden=100, n_classes=3,
                           l2_lambda=cfg["l2"], dropout_rate=cfg["drop"], seed=42)
    t_hist, v_hist = [], []
    
    for epoch in range(n_epochs):
        model.forward(X_tr, training=True)
        model.backward(X_tr, y_tr, lr)
        
        # Record DATA loss (without L2 penalty) for fair comparison
        model.forward(X_tr, training=False)
        t_hist.append(model.data_loss(y_tr))
        model.forward(X_va, training=False)
        v_hist.append(model.data_loss(y_va))
    
    train_acc = np.mean(model.predict(X_tr) == y_tr_l) * 100
    val_acc = np.mean(model.predict(X_va) == y_va_l) * 100
    test_acc = np.mean(model.predict(X_te) == y_te_l) * 100
    
    results[cfg["name"]] = {
        "train_loss": t_hist, "val_loss": v_hist,
        "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc,
        "model": model
    }
    
    print(f"{cfg['name']:>18s}: Train {train_acc:5.1f}% | Val {val_acc:5.1f}% | Test {test_acc:5.1f}%")
```

### Phase 3 — Visualize and Analyze

**Task:** Create a comprehensive comparison figure.

```python
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Row 1: Learning curves
for i, (name, res) in enumerate(results.items()):
    ax = axes[0, i]
    ax.plot(res["train_loss"], label='Train', linewidth=1.5, alpha=0.8)
    ax.plot(res["val_loss"], label='Val', linewidth=1.5, alpha=0.8)
    ax.set_title(f'{name}\nTest: {res["test_acc"]:.1f}%', fontsize=13)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('CCE Loss')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# Row 2: Decision boundaries
xx, yy = np.meshgrid(
    np.linspace(X_all[0].min()-0.3, X_all[0].max()+0.3, 200),
    np.linspace(X_all[1].min()-0.3, X_all[1].max()+0.3, 200))
grid = np.vstack([xx.ravel(), yy.ravel()])

for i, (name, res) in enumerate(results.items()):
    ax = axes[1, i]
    Z = res["model"].predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5],
                colors=['#ADD8E6', '#FFCCCB', '#90EE90'], alpha=0.4)
    for k, c in enumerate(['blue', 'red', 'green']):
        mask = y_te_l == k
        ax.scatter(X_te[0, mask], X_te[1, mask], c=c, edgecolors='black',
                   s=30, alpha=0.8)
    ax.set_title(f'Decision Boundary', fontsize=12)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

plt.suptitle('Regularization Showdown — Spiral Dataset (100 hidden neurons)',
             fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
```

### Phase 4 — Weight Analysis

**Task:** Compare the weight distributions across configurations. Regularization should produce smaller, more concentrated weights.

```python
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

for i, (name, res) in enumerate(results.items()):
    ax = axes[i]
    w1 = res["model"].W1.flatten()
    w2 = res["model"].W2.flatten()
    all_w = np.concatenate([w1, w2])
    
    ax.hist(all_w, bins=50, color=colors[i], alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.set_title(f'{name}\nstd={np.std(all_w):.3f}', fontsize=13)
    ax.set_xlabel('Weight value')
    ax.set_ylabel('Count')
    ax.set_xlim(-3, 3)

plt.suptitle('Weight Distributions', fontsize=16)
plt.tight_layout()
plt.show()
```

### Phase 5 — Write Your Report

Answer these questions in your notebook (3–5 sentences each):

1. **Which configuration achieved the best test accuracy?** Why do you think that is?
2. **Look at the decision boundaries.** Which are smooth? Which are jagged? How does this relate to generalization?
3. **Look at the weight distributions.** How did L2 regularization change the weights? What about dropout?
4. **What is the "overfitting gap"** (train acc − test acc) for each configuration? Which has the smallest gap?
5. **If you could only choose one regularization method**, which would it be and why?

---

## 9. Mini-Project C: Optimizer Olympics {#project-c}

### 🎯 Goal

Implement **SGD with Momentum** and **Adam** from scratch, then race them against vanilla SGD on a difficult optimization problem.

**Skills reused:** Gradient descent (Session 5), backpropagation (Session 6), MLP training (Session 6–7), loss visualization (Session 5).

---

### Phase 1 — Implement the Optimizers

**Task:** Complete the three optimizer classes below.

```python
class SGD:
    """Vanilla stochastic gradient descent."""
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def step(self, params, grads):
        """
        params: list of arrays [W1, b1, W2, b2, ...]
        grads:  list of arrays [dW1, db1, dW2, db2, ...]
        """
        for p, g in zip(params, grads):
            p -= self.lr * g


class SGDMomentum:
    """SGD with momentum."""
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocities = None
    
    def step(self, params, grads):
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]
        
        for i, (p, g) in enumerate(zip(params, grads)):
            # TODO: Update velocity and parameter
            # v = beta * v + (1 - beta) * g
            # p -= lr * v
            self.velocities[i] = self.beta * self.velocities[i] + (1 - self.beta) * g
            p -= self.lr * self.velocities[i]


class Adam:
    """Adam optimizer."""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
    
    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        self.t += 1
        
        for i, (p, g) in enumerate(zip(params, grads)):
            # Update biased first moment
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            # Update biased second moment
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
```

### Phase 2 — MLP with Pluggable Optimizer

**Task:** Modify the MLP to return gradients instead of updating weights internally, so we can plug in any optimizer.

```python
class FlexMLP:
    """MLP with external optimizer."""
    
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
        """Return gradients WITHOUT updating weights."""
        N = self.X.shape[1]
        delta2 = (self.a2 - y_true) / N
        dW2 = delta2 @ self.a1.T
        db2 = np.sum(delta2, axis=1, keepdims=True)
        delta1 = (self.W2.T @ delta2) * (self.z1 > 0).astype(float)
        dW1 = delta1 @ self.X.T
        db1 = np.sum(delta1, axis=1, keepdims=True)
        return [dW1, db1, dW2, db2]
    
    @property
    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]
    
    def predict(self, X):
        self.forward(X)
        return np.argmax(self.a2, axis=0)
```

### Phase 3 — The Race

**Task:** Train the same architecture with three optimizers on the spiral dataset.

```python
X_all, y_all_oh, y_all_lbl = generate_spiral(n_per_class=150, noise=0.25)
X_tr, y_tr, X_va, y_va, X_te, y_te = train_val_test_split(X_all, y_all_oh)
y_te_l = np.argmax(y_te, axis=0)

optimizer_configs = [
    ("SGD (lr=1.0)",        SGD(lr=1.0)),
    ("Momentum (lr=1.0)",   SGDMomentum(lr=1.0, beta=0.9)),
    ("Adam (lr=0.01)",      Adam(lr=0.01)),
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
    race_results[name] = {"train": t_hist, "val": v_hist, 
                          "test_acc": test_acc, "model": model}
    print(f"{name:>25s}: Test Acc = {test_acc:.1f}%")
```

### Phase 4 — Visualize the Race

```python
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Plot 1: Training loss (all on same axes)
ax = axes[0]
for (name, res), c in zip(race_results.items(), colors):
    ax.plot(res["train"], label=name, linewidth=2, color=c, alpha=0.8)
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Train Loss', fontsize=14)
ax.set_title('Training Convergence', fontsize=16)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 2: Zoom on first 500 epochs
ax = axes[1]
for (name, res), c in zip(race_results.items(), colors):
    ax.plot(res["train"][:500], label=name, linewidth=2, color=c, alpha=0.8)
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Train Loss', fontsize=14)
ax.set_title('Early Convergence (First 500 Epochs)', fontsize=16)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 3: Final test accuracy bar chart
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

### Phase 5 — The Gauntlet: Hard Loss Landscape

**Task:** Test the optimizers on a 2D function with a narrow valley (the Rosenbrock function). This is where momentum and Adam really shine.

```python
def rosenbrock(w1, w2):
    """Rosenbrock function: famous for its narrow curved valley."""
    return (1 - w1) ** 2 + 100 * (w2 - w1 ** 2) ** 2

def rosenbrock_grad(w1, w2):
    dw1 = -2 * (1 - w1) + 200 * (w2 - w1 ** 2) * (-2 * w1)
    dw2 = 200 * (w2 - w1 ** 2)
    return np.array([dw1]), np.array([dw2])

# Start far from the minimum at (1, 1)
start = (-1.0, -1.0)

# Run each optimizer
trajectories = {}

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
    print(f"{name:>10s}: final position = ({w1[0]:.4f}, {w2[0]:.4f}), "
          f"loss = {rosenbrock(w1[0], w2[0]):.6f}")

# Visualize
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

### Phase 6 — Write Your Report

Answer these questions:

1. **On the spiral dataset:** Which optimizer converged fastest in terms of epochs? Which reached the best final test accuracy?
2. **On Rosenbrock:** Why does vanilla SGD struggle in the narrow valley? How does momentum help?
3. **Adam vs Momentum:** Adam reached a reasonable solution faster. But look at the final loss values — which optimizer got closest to the true minimum?
4. **Practical recommendation:** You're about to start a new project. Which optimizer do you pick first, and why?

---

## Summary

### What We Learned

✅ **Generalization**: Training accuracy ≠ real-world performance  
✅ **Train/Val/Test splits**: The honest way to evaluate models  
✅ **Learning curves**: The diagnostic tool for overfitting and underfitting  
✅ **Early stopping**: Stop when validation loss increases  
✅ **L2 regularization**: Penalize large weights → smoother boundaries  
✅ **Dropout**: Randomly disable neurons → implicit ensemble  
✅ **Modern optimizers**: Momentum and Adam improve convergence

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
1. We manually implemented backpropagation, gradient descent, L2, and dropout. What parts were tedious and error-prone?
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
