# Session 7: Logistic Regression & Softmax
## From Regression to Classification

**Course: Neural Networks for Engineers**  
**Duration: 2 hours**

---

## Table of Contents

1. [Recap: What We Know So Far](#recap)
2. [The Classification Problem](#classification)
3. [Sigmoid as Probability: Logistic Regression](#logistic)
4. [Why MSE Fails for Classification](#mse-fails)
5. [Binary Cross-Entropy Loss](#bce)
6. [Implementing Logistic Regression from Scratch](#impl-logistic)
7. [Multi-Class Classification: Softmax](#softmax)
8. [Categorical Cross-Entropy Loss](#cce)
9. [Complete Classification Pipeline](#pipeline)
10. [Evaluation Metrics](#metrics)
11. [Final Exercises](#exercises)

---

## 1. Recap: What We Know So Far {#recap}

### What We've Learned

✅ **Loss functions**: MSE measures prediction error  
✅ **Gradient descent**: $w \leftarrow w - \eta \nabla L$  
✅ **Backpropagation**: Chain rule applied layer by layer  
✅ **MLP training**: Automatic weight learning on XOR  
✅ **Open question**: MSE + sigmoid for classification — is this the best we can do?

### 🤔 Quick Questions (from Session 6's "Think About")

**Q1:** Our MLP uses MSE loss for XOR classification. What might be wrong with MSE for classification?

<details>
<summary>Click to reveal answer</summary>
MSE treats all errors equally — an error of 0.1 is penalized the same whether the output is 0.5 or 0.99. But for classification, a confident **wrong** prediction (output 0.99 when target is 0) should be penalized much more harshly than an uncertain one (output 0.5). We need a loss function that understands **probabilities**.
</details>

**Q2:** If the network outputs 0.99 for a class-0 sample, how should the loss behave?

<details>
<summary>Click to reveal answer</summary>
The loss should be **very large** — almost infinite! The network is extremely confident and extremely wrong. MSE gives $(0 - 0.99)^2 = 0.98$, which is just... a number. Cross-entropy gives $-\log(1 - 0.99) = -\log(0.01) \approx 4.6$, which is much steeper. This is what we need.
</details>

**Q3:** What if we have 5 possible classes instead of 2? How should the output layer look?

<details>
<summary>Click to reveal answer</summary>
We need **5 output neurons**, one per class. Their outputs should be **probabilities that sum to 1**. This is exactly what the **softmax** function does!
</details>

---

## 2. The Classification Problem {#classification}

### Regression vs Classification

So far, we've seen two flavors of prediction:

| | Regression | Classification |
|---|---|---|
| **Output** | Continuous number | Discrete category |
| **Examples** | Price, temperature, age | Cat/dog, spam/not, digit 0-9 |
| **Output range** | Any real number | Probability per class |
| **Loss** | MSE | Cross-entropy (today!) |

### Types of Classification

| Type | Classes | Output | Example |
|---|---|---|---|
| **Binary** | 2 | 1 probability | Spam or not spam |
| **Multi-class** | $K > 2$ | $K$ probabilities | Digit recognition (0-9) |

### What We Need

A classification model should output **probabilities**:
- Output between 0 and 1
- For multi-class: outputs sum to 1
- High probability = high confidence

The **sigmoid** function (binary) and **softmax** function (multi-class) do exactly this!

---

## 3. Sigmoid as Probability: Logistic Regression {#logistic}

### The Model

Logistic regression = linear model + sigmoid activation.

$$
P(y = 1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}
$$

This is just a **single neuron** with sigmoid activation — the simplest possible classifier!

```
Input           Neuron            Output

 x₁ ─── w₁ ──┐
              │
 x₂ ─── w₂ ──┼── Σ + b ── σ ── P(y=1)
              │
 x₃ ─── w₃ ──┘
```

### Why Sigmoid Outputs Are Probabilities

The sigmoid function has exactly the right properties:

| Property | Sigmoid satisfies? |
|---|---|
| Output in $[0, 1]$ | ✅ $\sigma(z) \in (0, 1)$ for all $z$ |
| Output 0.5 when uncertain | ✅ $\sigma(0) = 0.5$ |
| Approaches 1 for large positive $z$ | ✅ Confident class 1 |
| Approaches 0 for large negative $z$ | ✅ Confident class 0 |

### The Decision Threshold

To make a hard prediction, we choose a **threshold** (usually 0.5):

$$
\hat{y} = \begin{cases} 1 & \text{if } \sigma(\mathbf{w}^T\mathbf{x} + b) \geq 0.5 \\ 0 & \text{otherwise} \end{cases}
$$

### 🤔 Think About It

**Q:** When is the sigmoid output exactly 0.5?

<details>
<summary>Answer</summary>
When $\mathbf{w}^T\mathbf{x} + b = 0$. This equation defines the **decision boundary** — the hyperplane that separates the two classes. Points on one side have $P(y=1) > 0.5$, points on the other have $P(y=1) < 0.5$.

This is exactly the same linear boundary we saw with the perceptron in Session 2!
</details>

### 💻 Code It: Sigmoid as Probability

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Visualize sigmoid as probability
z = np.linspace(-6, 6, 200)
p = sigmoid(z)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(z, p, 'b-', linewidth=3)

# Annotate regions
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

ax.fill_between(z, p, 0.5, where=(p >= 0.5), alpha=0.15, color='red', label='Predict class 1')
ax.fill_between(z, p, 0.5, where=(p < 0.5), alpha=0.15, color='blue', label='Predict class 0')

ax.annotate('Confident: class 0', xy=(-4, 0.05), fontsize=12, color='blue')
ax.annotate('Uncertain', xy=(-0.8, 0.55), fontsize=12, color='gray')
ax.annotate('Confident: class 1', xy=(2.5, 0.95), fontsize=12, color='red')

ax.set_xlabel('$z = w^T x + b$', fontsize=14)
ax.set_ylabel('$P(y = 1)$', fontsize=14)
ax.set_title('Sigmoid Output as Probability', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)
plt.show()
```

---

## 4. Why MSE Fails for Classification {#mse-fails}

### The Problem

Let's compare MSE and cross-entropy (which we'll define next) when the true label is $y = 1$:

```python
# True label: y = 1
y_hat = np.linspace(0.001, 0.999, 200)

# MSE loss
mse = (1 - y_hat) ** 2

# Cross-entropy loss (preview!)
bce = -np.log(y_hat)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# MSE
ax = axes[0]
ax.plot(y_hat, mse, 'b-', linewidth=2)
ax.set_xlabel('Predicted $\\hat{y}$', fontsize=14)
ax.set_ylabel('Loss', fontsize=14)
ax.set_title('MSE Loss (target = 1)', fontsize=16)
ax.grid(True, alpha=0.3)
ax.annotate('Gradient is small here!\n(learning is slow)', 
            xy=(0.05, 0.85), fontsize=11, color='red',
            arrowprops=dict(arrowstyle='->', color='red'),
            xytext=(0.3, 0.6))

# Cross-entropy
ax = axes[1]
ax.plot(y_hat, bce, 'r-', linewidth=2)
ax.set_xlabel('Predicted $\\hat{y}$', fontsize=14)
ax.set_ylabel('Loss', fontsize=14)
ax.set_title('Cross-Entropy Loss (target = 1)', fontsize=16)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 7)
ax.annotate('Gradient is huge!\n(learning is fast)', 
            xy=(0.05, 5.5), fontsize=11, color='red',
            arrowprops=dict(arrowstyle='->', color='red'),
            xytext=(0.3, 4.0))

plt.tight_layout()
plt.show()
```

### Two Problems with MSE for Classification

**Problem 1: Slow gradients when confidently wrong**

When the output is near 0 or 1, the sigmoid is **saturated** — its derivative is almost zero. Combined with MSE, the gradient becomes tiny:

$$
\frac{\partial L_{\text{MSE}}}{\partial z} = 2(\hat{y} - y) \cdot \underbrace{\sigma'(z)}_{\approx 0 \text{ when saturated}}
$$

The network is very wrong but can barely learn! This is called the **slow learning problem**.

**Problem 2: Non-convex loss surface**

MSE + sigmoid creates a loss surface with **flat regions**, making optimization harder. Cross-entropy + sigmoid creates a much nicer (convex) surface.

### The Solution: Cross-Entropy

Cross-entropy loss **cancels out** the sigmoid saturation:

$$
\frac{\partial L_{\text{CE}}}{\partial z} = \hat{y} - y
$$

No sigmoid derivative! The gradient is simply the **prediction error**. The more wrong the prediction, the larger the gradient — exactly what we want.

### ✏️ Exercise 4.1: Gradient Comparison

For $y = 1$ and $\hat{y} = \sigma(z) = 0.01$ (very wrong prediction, $z \approx -4.6$):

**MSE gradient w.r.t. $z$:**

$$
\frac{\partial L_{\text{MSE}}}{\partial z} = 2(\hat{y} - y) \cdot \sigma'(z) = 2(0.01 - 1) \cdot 0.01 \cdot 0.99 = \text{___}
$$

**Cross-entropy gradient w.r.t. $z$:**

$$
\frac{\partial L_{\text{CE}}}{\partial z} = \hat{y} - y = 0.01 - 1 = \text{___}
$$

<details>
<summary>Solution</summary>

**MSE:** $2 \times (-0.99) \times 0.0099 = -0.0196$ (tiny gradient!)

**Cross-entropy:** $0.01 - 1 = -0.99$ (large gradient!)

The cross-entropy gradient is **50× larger**. This means the network learns much faster from confident wrong predictions.
</details>

---

## 5. Binary Cross-Entropy Loss {#bce}

### Definition

For a single sample with true label $y \in \{0, 1\}$ and predicted probability $\hat{y} \in (0, 1)$:

$$
L_{\text{BCE}} = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]
$$

For a batch of $N$ samples:

$$
L_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

### Understanding the Formula

The formula has **two cases** (only one is active per sample):

| True label $y$ | Active term | Loss | Behavior |
|---|---|---|---|
| $y = 1$ | $-\log(\hat{y})$ | Low when $\hat{y} \to 1$ | Penalizes low confidence for class 1 |
| $y = 0$ | $-\log(1 - \hat{y})$ | Low when $\hat{y} \to 0$ | Penalizes high confidence for class 0 |

### Why Logarithm?

The logarithm provides the right penalty shape:

| $\hat{y}$ (when $y = 1$) | $-\log(\hat{y})$ | Interpretation |
|---|---|---|
| 0.99 | 0.01 | Almost right → tiny loss |
| 0.9 | 0.11 | Pretty right → small loss |
| 0.5 | 0.69 | Uncertain → moderate loss |
| 0.1 | 2.30 | Pretty wrong → large loss |
| 0.01 | 4.61 | Very wrong → **huge** loss |
| 0.001 | 6.91 | Extremely wrong → **massive** loss |

The log penalty grows **without bound** as the prediction approaches the wrong answer — exactly the steep penalty we wanted!

### ✏️ Exercise 5.1: Compute BCE by Hand

Compute the binary cross-entropy loss for these predictions:

| Sample | $y$ | $\hat{y}$ | $-y\log(\hat{y})$ | $-(1-y)\log(1-\hat{y})$ | Sample Loss |
|--------|-----|-----------|-------------------|------------------------|-------------|
| 1      | 1   | 0.9       | ___               | ___                    | ___         |
| 2      | 0   | 0.2       | ___               | ___                    | ___         |
| 3      | 1   | 0.3       | ___               | ___                    | ___         |
| 4      | 0   | 0.8       | ___               | ___                    | ___         |

**Total BCE =** ___

Which sample contributes the **most** loss? Why?

<details>
<summary>Solution</summary>

| Sample | $y$ | $\hat{y}$ | $-y\log(\hat{y})$ | $-(1-y)\log(1-\hat{y})$ | Sample Loss |
|--------|-----|-----------|-------------------|------------------------|-------------|
| 1      | 1   | 0.9       | $-\log(0.9) = 0.105$ | 0 | 0.105 |
| 2      | 0   | 0.2       | 0 | $-\log(0.8) = 0.223$ | 0.223 |
| 3      | 1   | 0.3       | $-\log(0.3) = 1.204$ | 0 | 1.204 |
| 4      | 0   | 0.8       | 0 | $-\log(0.2) = 1.609$ | 1.609 |

$$
\text{BCE} = \frac{0.105 + 0.223 + 1.204 + 1.609}{4} = \frac{3.141}{4} = 0.785
$$

**Sample 4** contributes the most loss ($1.609$) — it predicts 0.8 probability for class 1, but the true label is class 0. It's confidently wrong!
</details>

### The Gradient: Beautifully Simple

For logistic regression ($\hat{y} = \sigma(\mathbf{w}^T\mathbf{x} + b)$) with BCE loss:

$$
\frac{\partial L}{\partial w_j} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i) x_{ij}
$$

$$
\frac{\partial L}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)
$$

**This is the same form as linear regression with MSE!** The only difference is that $\hat{y}$ is now passed through sigmoid.

### 💻 Code It: Compare MSE vs BCE Loss Surfaces

```python
# 1D logistic regression: P(y=1) = sigmoid(w * x + b)
# Let's fix b = 0 and vary w to see the loss surface

np.random.seed(42)
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([0, 0, 0, 1, 1], dtype=float)

w_range = np.linspace(-3, 3, 200)

mse_losses = []
bce_losses = []

for w in w_range:
    y_hat = sigmoid(w * X)
    y_hat_clipped = np.clip(y_hat, 1e-7, 1 - 1e-7)  # Prevent log(0)
    
    mse = np.mean((y - y_hat) ** 2)
    bce = -np.mean(y * np.log(y_hat_clipped) + (1 - y) * np.log(1 - y_hat_clipped))
    
    mse_losses.append(mse)
    bce_losses.append(bce)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(w_range, mse_losses, 'b-', linewidth=2)
axes[0].set_title('MSE Loss Surface', fontsize=16)
axes[0].set_xlabel('Weight $w$', fontsize=14)
axes[0].set_ylabel('Loss', fontsize=14)
axes[0].grid(True, alpha=0.3)

axes[1].plot(w_range, bce_losses, 'r-', linewidth=2)
axes[1].set_title('BCE Loss Surface', fontsize=16)
axes[1].set_xlabel('Weight $w$', fontsize=14)
axes[1].set_ylabel('Loss', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.suptitle('MSE vs Binary Cross-Entropy for Classification', fontsize=18, y=1.02)
plt.tight_layout()
plt.show()
```

### 🤔 Think About It

**Q:** Look at the MSE loss surface. Can you see the flat regions? What would happen to gradient descent in those regions?

<details>
<summary>Answer</summary>
The MSE surface has **plateaus** where the sigmoid is saturated. Gradient descent would slow down dramatically in these regions, even though the weights are far from optimal. The BCE surface is much smoother and steeper everywhere — gradient descent converges faster and more reliably.
</details>

---

## 6. Implementing Logistic Regression from Scratch {#impl-logistic}

### The Complete Model

```python
class LogisticRegression:
    """
    Binary classifier: P(y=1|x) = sigmoid(Wx + b)
    Trained with binary cross-entropy loss.
    """
    
    def __init__(self, n_features):
        self.W = np.zeros((1, n_features))
        self.b = np.zeros((1, 1))
    
    def forward(self, X):
        """
        X shape: (n_features, N)
        Returns probabilities shape: (1, N)
        """
        self.z = self.W @ X + self.b
        self.y_hat = sigmoid(self.z)
        return self.y_hat
    
    def compute_loss(self, y_true):
        """Binary cross-entropy loss"""
        N = y_true.shape[1]
        y_hat_clipped = np.clip(self.y_hat, 1e-7, 1 - 1e-7)
        loss = -np.mean(
            y_true * np.log(y_hat_clipped) + 
            (1 - y_true) * np.log(1 - y_hat_clipped)
        )
        return loss
    
    def backward(self, X, y_true, lr):
        """Gradient descent update"""
        N = X.shape[1]
        
        # Gradient (the beautiful simple form!)
        error = self.y_hat - y_true                # (1, N)
        dW = (1 / N) * (error @ X.T)              # (1, n_features)
        db = (1 / N) * np.sum(error, keepdims=True)  # (1, 1)
        
        self.W -= lr * dW
        self.b -= lr * db
    
    def predict(self, X, threshold=0.5):
        """Hard predictions"""
        probs = self.forward(X)
        return (probs >= threshold).astype(int)
```

### 💻 Code It: Train on a 2D Dataset

**Fill in the training loop:**

```python
def generate_binary_data(n_samples=200):
    """Two Gaussian clouds"""
    np.random.seed(42)
    # Class 0: centered at (-1, -1)
    X0 = np.random.randn(2, n_samples // 2) * 0.8 + np.array([[-1], [-1]])
    # Class 1: centered at (1, 1)
    X1 = np.random.randn(2, n_samples // 2) * 0.8 + np.array([[1], [1]])
    
    X = np.hstack([X0, X1])
    y = np.hstack([np.zeros((1, n_samples // 2)), np.ones((1, n_samples // 2))])
    
    # Shuffle
    idx = np.random.permutation(n_samples)
    return X[:, idx], y[:, idx]

X_train, y_train = generate_binary_data()

# Train
model = LogisticRegression(n_features=2)
loss_history = []

for epoch in range(___):  # How many epochs?
    # TODO: Forward pass
    ___
    
    # TODO: Compute and record loss
    ___
    
    # TODO: Backward pass
    ___
    
    if epoch % 100 == 0:
        acc = np.mean(model.predict(X_train) == y_train) * 100
        print(f"Epoch {epoch:4d}: Loss = {loss:.4f}, Acc = {acc:.1f}%")
```

<details>
<summary>Solution</summary>

```python
model = LogisticRegression(n_features=2)
loss_history = []

for epoch in range(1000):
    model.forward(X_train)
    loss = model.compute_loss(y_train)
    loss_history.append(loss)
    model.backward(X_train, y_train, lr=0.5)
    
    if epoch % 100 == 0:
        acc = np.mean(model.predict(X_train) == y_train) * 100
        print(f"Epoch {epoch:4d}: Loss = {loss:.4f}, Acc = {acc:.1f}%")
```
</details>

### 💻 Code It: Visualize Decision Boundary and Probabilities

```python
def plot_logistic_result(model, X, y):
    """Visualize the decision boundary and probability field"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Create grid
    x_min, x_max = X[0].min() - 1, X[0].max() + 1
    y_min, y_max = X[1].min() - 1, X[1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                          np.linspace(y_min, y_max, 200))
    grid = np.vstack([xx.ravel(), yy.ravel()])
    probs = model.forward(grid).reshape(xx.shape)
    
    # Plot 1: Decision boundary
    ax = axes[0]
    ax.contourf(xx, yy, probs, levels=[0, 0.5, 1],
                colors=['#ADD8E6', '#FFCCCB'], alpha=0.4)
    ax.contour(xx, yy, probs, levels=[0.5], colors='black', linewidths=2)
    ax.scatter(X[0, y[0] == 0], X[1, y[0] == 0], c='blue', alpha=0.6, 
               edgecolors='black', label='Class 0')
    ax.scatter(X[0, y[0] == 1], X[1, y[0] == 1], c='red', alpha=0.6, 
               edgecolors='black', label='Class 1')
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_title('Decision Boundary', fontsize=16)
    ax.legend(fontsize=12)
    
    # Plot 2: Probability heatmap
    ax = axes[1]
    contour = ax.contourf(xx, yy, probs, levels=20, cmap='RdYlBu_r')
    plt.colorbar(contour, ax=ax, label='$P(y=1)$')
    ax.contour(xx, yy, probs, levels=[0.5], colors='black', linewidths=2)
    ax.scatter(X[0, y[0] == 0], X[1, y[0] == 0], c='blue', alpha=0.6, 
               edgecolors='black', s=20)
    ax.scatter(X[0, y[0] == 1], X[1, y[0] == 1], c='red', alpha=0.6, 
               edgecolors='black', s=20)
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_title('Predicted Probabilities', fontsize=16)
    
    plt.tight_layout()
    plt.show()

plot_logistic_result(model, X_train, y_train)
```

### The Decision Threshold

We used 0.5, but the threshold is a **tunable parameter**:

```python
def threshold_experiment(model, X, y):
    """Show effect of different thresholds"""
    thresholds = [0.3, 0.5, 0.7]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Create grid
    x_min, x_max = X[0].min() - 1, X[0].max() + 1
    y_min, y_max = X[1].min() - 1, X[1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                          np.linspace(y_min, y_max, 200))
    grid = np.vstack([xx.ravel(), yy.ravel()])
    probs = model.forward(grid).reshape(xx.shape)
    
    for ax, t in zip(axes, thresholds):
        ax.contourf(xx, yy, probs, levels=[0, t, 1],
                    colors=['#ADD8E6', '#FFCCCB'], alpha=0.4)
        ax.contour(xx, yy, probs, levels=[t], colors='black', linewidths=2)
        ax.scatter(X[0, y[0] == 0], X[1, y[0] == 0], c='blue', alpha=0.6, 
                   edgecolors='black', s=30)
        ax.scatter(X[0, y[0] == 1], X[1, y[0] == 1], c='red', alpha=0.6, 
                   edgecolors='black', s=30)
        
        preds = (model.forward(X) >= t).astype(int)
        acc = np.mean(preds == y) * 100
        ax.set_title(f'Threshold = {t} (Acc: {acc:.1f}%)', fontsize=14)
        ax.set_xlabel('$x_1$', fontsize=12)
        ax.set_ylabel('$x_2$', fontsize=12)
    
    plt.suptitle('Effect of Decision Threshold', fontsize=16)
    plt.tight_layout()
    plt.show()

threshold_experiment(model, X_train, y_train)
```

### 🤔 Think About It

**Q:** When would you use a threshold other than 0.5?

<details>
<summary>Answer</summary>
When the **cost of mistakes is asymmetric**:

- **Medical diagnosis** (cancer detection): Use a low threshold (e.g., 0.3). It's better to have false positives (unnecessary tests) than false negatives (missed cancer).
- **Spam filtering**: Use a higher threshold (e.g., 0.7). It's worse to lose a real email (false positive) than to let through some spam (false negative).

The threshold lets you trade off between **precision** and **recall** (more on this in Section 10).
</details>

---

## 7. Multi-Class Classification: Softmax {#softmax}

### The Challenge

Binary classification: 1 output neuron + sigmoid → $P(y = 1)$.

But what about $K$ classes (e.g., digit recognition with 10 classes)?

We need:
- $K$ output neurons
- All outputs between 0 and 1
- All outputs **sum to 1** (they're a probability distribution!)

### The Softmax Function

Given a vector of raw scores (logits) $\mathbf{z} = [z_1, z_2, \ldots, z_K]$:

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

**In words:** Exponentiate each score, then normalize by the total.

### Step-by-Step Example

Suppose a 3-class network outputs logits $\mathbf{z} = [2.0, 1.0, 0.5]$:

| Step | $z_1 = 2.0$ | $z_2 = 1.0$ | $z_3 = 0.5$ |
|------|------------|------------|------------|
| 1. Exponentiate | $e^{2.0} = 7.389$ | $e^{1.0} = 2.718$ | $e^{0.5} = 1.649$ |
| 2. Sum | $7.389 + 2.718 + 1.649 = 11.756$ |||
| 3. Normalize | $\frac{7.389}{11.756} = 0.629$ | $\frac{2.718}{11.756} = 0.231$ | $\frac{1.649}{11.756} = 0.140$ |

**Result:** $P = [0.629, 0.231, 0.140]$ — sums to 1.0 ✓

**Prediction:** Class 1 (highest probability).

### ✏️ Exercise 7.1: Compute Softmax by Hand

Compute softmax for $\mathbf{z} = [1.0, 2.0, 3.0]$:

| | $z_1 = 1.0$ | $z_2 = 2.0$ | $z_3 = 3.0$ |
|---|---|---|---|
| $e^{z_i}$ | ___ | ___ | ___ |
| $\sum e^{z_j}$ | ___ | | |
| $\text{softmax}(z_i)$ | ___ | ___ | ___ |

Which class is predicted? ___

<details>
<summary>Solution</summary>

| | $z_1 = 1.0$ | $z_2 = 2.0$ | $z_3 = 3.0$ |
|---|---|---|---|
| $e^{z_i}$ | $2.718$ | $7.389$ | $20.086$ |
| $\sum e^{z_j}$ | $30.193$ | | |
| $\text{softmax}(z_i)$ | $0.090$ | $0.245$ | $0.665$ |

**Predicted class: 3** (highest probability at 66.5%)

Notice how softmax amplifies differences: the logits differ by 1 each, but the probabilities range from 9% to 66%.
</details>

### Properties of Softmax

| Property | Explanation |
|---|---|
| All outputs $\in (0, 1)$ | Exponentials are always positive |
| Outputs sum to 1 | Division by total ensures normalization |
| Preserves ordering | Largest logit → largest probability |
| Amplifies differences | Exponential makes large values much larger |
| Sensitive to scale | Multiplying all logits by a constant changes the distribution |

### The Numerical Stability Trick

Computing $e^{z_i}$ for large $z$ can cause **overflow**. The fix:

$$
\text{softmax}(z_i) = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j} e^{z_j - \max(\mathbf{z})}}
$$

Subtracting the max doesn't change the result (it cancels out) but keeps numbers manageable.

### 💻 Code It: Softmax Implementation

```python
def softmax(z):
    """
    Numerically stable softmax.
    z shape: (K, N) where K = classes, N = samples
    """
    # Subtract max for numerical stability
    z_shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# Test
z = np.array([[2.0, 1.0],
              [1.0, 3.0],
              [0.5, 0.5]])  # 3 classes, 2 samples

probs = softmax(z)
print("Logits:")
print(z)
print("\nSoftmax probabilities:")
print(probs)
print("\nSum per sample:", np.sum(probs, axis=0))  # Should be [1, 1]
```

### 💻 Code It: Visualize Softmax Behavior

```python
def visualize_softmax():
    """Show how softmax transforms logits to probabilities"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Different logit patterns
    cases = [
        ([2.0, 1.0, 0.5], "Moderate confidence"),
        ([5.0, 1.0, 0.5], "High confidence"),
        ([1.0, 1.0, 1.0], "Uniform (uncertain)"),
    ]
    
    for ax, (logits, title) in zip(axes, cases):
        z = np.array(logits)
        probs = softmax(z.reshape(-1, 1)).flatten()
        classes = [f'Class {i+1}' for i in range(len(z))]
        
        # Side-by-side bars
        x = np.arange(len(z))
        width = 0.35
        ax.bar(x - width/2, z, width, label='Logits $z$', color='steelblue', alpha=0.7)
        ax.bar(x + width/2, probs, width, label='Softmax $P$', color='coral', alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Annotate probabilities
        for i, p in enumerate(probs):
            ax.annotate(f'{p:.1%}', xy=(i + width/2, p + 0.1), 
                       ha='center', fontsize=11, fontweight='bold', color='red')
    
    plt.suptitle('Softmax: From Logits to Probabilities', fontsize=16)
    plt.tight_layout()
    plt.show()

visualize_softmax()
```

### Softmax vs Sigmoid

| | Sigmoid | Softmax |
|---|---|---|
| **Use** | Binary classification | Multi-class classification |
| **Outputs** | 1 value in $(0, 1)$ | $K$ values, sum to 1 |
| **Formula** | $\frac{1}{1+e^{-z}}$ | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ |
| **Special case** | Softmax with $K = 2$ reduces to sigmoid! | |

---

## 8. Categorical Cross-Entropy Loss {#cce}

### One-Hot Encoding

For multi-class problems, targets are encoded as **one-hot vectors**:

| Class | One-hot | Meaning |
|---|---|---|
| Cat (class 0) | $[1, 0, 0]$ | 100% cat, 0% dog, 0% bird |
| Dog (class 1) | $[0, 1, 0]$ | 0% cat, 100% dog, 0% bird |
| Bird (class 2) | $[0, 0, 1]$ | 0% cat, 0% dog, 100% bird |

### Categorical Cross-Entropy (CCE)

For a single sample with one-hot target $\mathbf{y}$ and predicted probabilities $\hat{\mathbf{y}}$:

$$
L_{\text{CCE}} = -\sum_{k=1}^{K} y_k \log(\hat{y}_k)
$$

Since $\mathbf{y}$ is one-hot (only one element is 1), this simplifies to:

$$
L_{\text{CCE}} = -\log(\hat{y}_c)
$$

Where $c$ is the **true class**. We simply take $-\log$ of the predicted probability for the correct class!

### ✏️ Exercise 8.1: Compute CCE by Hand

A 3-class network predicts:

| Sample | True class | $\hat{y}_1$ | $\hat{y}_2$ | $\hat{y}_3$ | Loss $-\log(\hat{y}_c)$ |
|--------|-----------|-------------|-------------|-------------|------------------------|
| 1      | Class 2   | 0.1         | 0.7         | 0.2         | ___                    |
| 2      | Class 1   | 0.8         | 0.1         | 0.1         | ___                    |
| 3      | Class 3   | 0.2         | 0.3         | 0.5         | ___                    |

**Average CCE =** ___

<details>
<summary>Solution</summary>

| Sample | True class | Correct prob $\hat{y}_c$ | Loss $-\log(\hat{y}_c)$ |
|--------|-----------|-------------------------|------------------------|
| 1      | Class 2   | 0.7                     | $-\log(0.7) = 0.357$  |
| 2      | Class 1   | 0.8                     | $-\log(0.8) = 0.223$  |
| 3      | Class 3   | 0.5                     | $-\log(0.5) = 0.693$  |

$$
\text{Average CCE} = \frac{0.357 + 0.223 + 0.693}{3} = 0.424
$$

**Observations:**
- Sample 2 has the lowest loss (most confident and correct)
- Sample 3 has the highest loss (least confident about the correct class)
- Sample 1 is in between (fairly confident and correct)
</details>

### The Gradient: Softmax + CCE

Just like sigmoid + BCE, the combination of softmax + CCE gives a clean gradient:

$$
\frac{\partial L}{\partial z_k} = \hat{y}_k - y_k
$$

**Exactly the same beautiful form!** Predicted minus true, for each class.

### 💻 Code It: Categorical Cross-Entropy

```python
def categorical_cross_entropy(y_true, y_hat):
    """
    Compute CCE loss.
    y_true: one-hot, shape (K, N)
    y_hat: softmax probabilities, shape (K, N)
    """
    N = y_true.shape[1]
    y_hat_clipped = np.clip(y_hat, 1e-7, 1 - 1e-7)
    loss = -np.sum(y_true * np.log(y_hat_clipped)) / N
    return loss

# Test
y_true = np.array([[0, 1, 0],   # Sample 1: class 2
                    [1, 0, 0],   # Sample 2: class 1
                    [0, 0, 1]]).T  # Sample 3: class 3
# shape: (3 classes, 3 samples)

y_hat = np.array([[0.1, 0.8, 0.2],
                   [0.7, 0.1, 0.3],
                   [0.2, 0.1, 0.5]]).T

print(f"CCE Loss: {categorical_cross_entropy(y_true, y_hat):.4f}")
```

---

## 9. Complete Classification Pipeline {#pipeline}

### Multi-Class MLP Classifier

Let's build a full multi-class classifier using everything we've learned:

```python
class MultiClassMLP:
    """
    MLP for multi-class classification.
    Architecture: input → hidden (ReLU) → output (softmax)
    Loss: categorical cross-entropy
    """
    
    def __init__(self, n_input, n_hidden, n_classes):
        np.random.seed(42)
        # Xavier initialization
        self.W1 = np.random.randn(n_hidden, n_input) * np.sqrt(2.0 / n_input)
        self.b1 = np.zeros((n_hidden, 1))
        self.W2 = np.random.randn(n_classes, n_hidden) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros((n_classes, 1))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def forward(self, X):
        """Forward pass with ReLU hidden + softmax output"""
        self.X = X
        
        # Hidden layer (ReLU)
        self.z1 = self.W1 @ X + self.b1
        self.a1 = self.relu(self.z1)
        
        # Output layer (softmax)
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = softmax(self.z2)
        
        return self.a2
    
    def compute_loss(self, y_true):
        """Categorical cross-entropy"""
        return categorical_cross_entropy(y_true, self.a2)
    
    def backward(self, y_true, lr):
        """Backprop with softmax + CCE (gradient = y_hat - y_true)"""
        N = y_true.shape[1]
        
        # Output layer: the beautiful gradient
        delta2 = (self.a2 - y_true) / N           # (K, N)
        dW2 = delta2 @ self.a1.T                   # (K, n_hidden)
        db2 = np.sum(delta2, axis=1, keepdims=True)
        
        # Hidden layer
        delta1 = (self.W2.T @ delta2) * self.relu_derivative(self.z1)
        dW1 = delta1 @ self.X.T
        db1 = np.sum(delta1, axis=1, keepdims=True)
        
        # Update
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
    
    def predict(self, X):
        """Return predicted class indices"""
        probs = self.forward(X)
        return np.argmax(probs, axis=0)
```

### 💻 Code It: The Spiral Dataset (3 Classes)

```python
def generate_spiral_data(n_per_class=100, n_classes=3, noise=0.3):
    """
    Generate a spiral dataset with K classes.
    Classic benchmark for non-linear classifiers.
    """
    np.random.seed(42)
    N = n_per_class * n_classes
    X = np.zeros((2, N))
    y = np.zeros(N, dtype=int)
    
    for k in range(n_classes):
        start = k * n_per_class
        end = start + n_per_class
        
        r = np.linspace(0.2, 1.0, n_per_class)
        theta = np.linspace(k * 4.0, (k + 1) * 4.0, n_per_class) + np.random.randn(n_per_class) * noise
        
        X[0, start:end] = r * np.cos(theta)
        X[1, start:end] = r * np.sin(theta)
        y[start:end] = k
    
    # One-hot encode
    y_onehot = np.zeros((n_classes, N))
    y_onehot[y, np.arange(N)] = 1
    
    # Shuffle
    idx = np.random.permutation(N)
    return X[:, idx], y_onehot[:, idx], y[idx]

X_spiral, y_spiral_oh, y_spiral = generate_spiral_data()

# Visualize
plt.figure(figsize=(8, 8))
colors = ['blue', 'red', 'green']
for k in range(3):
    mask = y_spiral == k
    plt.scatter(X_spiral[0, mask], X_spiral[1, mask], 
                c=colors[k], alpha=0.6, edgecolors='black', s=30,
                label=f'Class {k}')
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Spiral Dataset (3 Classes)', fontsize=16)
plt.legend(fontsize=12)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.show()
```

### 💻 Code It: Train and Visualize

```python
# Train
mlp = MultiClassMLP(n_input=2, n_hidden=50, n_classes=3)
loss_history = []

n_epochs = 5000
lr = 1.0

for epoch in range(n_epochs):
    mlp.forward(X_spiral)
    loss = mlp.compute_loss(y_spiral_oh)
    loss_history.append(loss)
    mlp.backward(y_spiral_oh, lr)
    
    if epoch % 1000 == 0:
        preds = mlp.predict(X_spiral)
        acc = np.mean(preds == y_spiral) * 100
        print(f"Epoch {epoch:5d}: Loss = {loss:.4f}, Acc = {acc:.1f}%")

# Final accuracy
preds = mlp.predict(X_spiral)
print(f"\nFinal accuracy: {np.mean(preds == y_spiral) * 100:.1f}%")
```

```python
# Visualize decision regions
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Loss curve
axes[0].plot(loss_history, 'b-', linewidth=1)
axes[0].set_xlabel('Epoch', fontsize=14)
axes[0].set_ylabel('CCE Loss', fontsize=14)
axes[0].set_title('Training Loss', fontsize=16)
axes[0].grid(True, alpha=0.3)

# Plot 2: Decision regions
ax = axes[1]
x_min, x_max = X_spiral[0].min() - 0.5, X_spiral[0].max() + 0.5
y_min, y_max = X_spiral[1].min() - 0.5, X_spiral[1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                      np.linspace(y_min, y_max, 200))
grid = np.vstack([xx.ravel(), yy.ravel()])
Z = mlp.predict(grid).reshape(xx.shape)

ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5], 
            colors=['#ADD8E6', '#FFCCCB', '#90EE90'], alpha=0.4)

colors = ['blue', 'red', 'green']
for k in range(3):
    mask = y_spiral == k
    ax.scatter(X_spiral[0, mask], X_spiral[1, mask],
               c=colors[k], alpha=0.6, edgecolors='black', s=30,
               label=f'Class {k}')

ax.set_xlabel('$x_1$', fontsize=14)
ax.set_ylabel('$x_2$', fontsize=14)
ax.set_title('Learned Decision Regions', fontsize=16)
ax.legend(fontsize=12)

plt.tight_layout()
plt.show()
```

---

## 10. Evaluation Metrics {#metrics}

### Beyond Accuracy

Accuracy = fraction of correct predictions. Simple, but sometimes **misleading**.

**Example:** A disease affects 1% of patients. A model that always predicts "healthy" gets 99% accuracy — but it's completely useless!

### The Confusion Matrix

For binary classification, there are 4 types of outcomes:

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actually Positive** | True Positive (TP) | False Negative (FN) |
| **Actually Negative** | False Positive (FP) | True Negative (TN) |

### Key Metrics

| Metric | Formula | Question it answers |
|---|---|---|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall, how often are we right? |
| **Precision** | $\frac{TP}{TP + FP}$ | When we predict positive, how often are we right? |
| **Recall** | $\frac{TP}{TP + FN}$ | Of all positives, how many did we catch? |
| **F1 Score** | $2 \cdot \frac{P \cdot R}{P + R}$ | Balance between precision and recall |

### ✏️ Exercise 10.1: Compute Metrics

A spam classifier produces these results on 100 emails:

|  | Predicted Spam | Predicted Not Spam |
|---|---|---|
| **Actually Spam** | 40 | 10 |
| **Actually Not Spam** | 5 | 45 |

Compute:
- Accuracy = ___
- Precision = ___
- Recall = ___
- F1 Score = ___

<details>
<summary>Solution</summary>

$TP = 40, FP = 5, FN = 10, TN = 45$

- **Accuracy** $= \frac{40 + 45}{100} = 0.85 = 85\%$

- **Precision** $= \frac{40}{40 + 5} = \frac{40}{45} = 0.889 = 88.9\%$

- **Recall** $= \frac{40}{40 + 10} = \frac{40}{50} = 0.80 = 80\%$

- **F1** $= 2 \times \frac{0.889 \times 0.80}{0.889 + 0.80} = 2 \times \frac{0.711}{1.689} = 0.842 = 84.2\%$

**Interpretation:** The classifier is better at precision (when it says "spam", it's usually right) than recall (it misses 20% of actual spam).
</details>

### 💻 Code It: Confusion Matrix

```python
def confusion_matrix(y_true, y_pred, n_classes):
    """
    Compute confusion matrix.
    y_true, y_pred: arrays of class indices, shape (N,)
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm

def plot_confusion_matrix(cm, class_names=None):
    """Visualize confusion matrix"""
    n = cm.shape[0]
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n)]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    
    # Add text annotations
    for i in range(n):
        for j in range(n):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', 
                    color=color, fontsize=14, fontweight='bold')
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('True', fontsize=14)
    ax.set_title('Confusion Matrix', fontsize=16)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

# Compute and display for spiral dataset
preds = mlp.predict(X_spiral)
cm = confusion_matrix(y_spiral, preds, n_classes=3)
plot_confusion_matrix(cm, class_names=['Class 0', 'Class 1', 'Class 2'])
print("\nConfusion matrix:")
print(cm)
print(f"\nPer-class accuracy:")
for k in range(3):
    class_acc = cm[k, k] / cm[k].sum() * 100
    print(f"  Class {k}: {class_acc:.1f}%")
```

### Multi-Class Metrics

For $K > 2$ classes, we compute precision and recall **per class** and then average:

```python
def classification_report(y_true, y_pred, n_classes):
    """Compute per-class and overall metrics"""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    
    print(f"{'Class':>8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 50)
    
    precisions, recalls, f1s, supports = [], [], [], []
    
    for k in range(n_classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        support = cm[k, :].sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{'Class '+str(k):>8} {precision:10.3f} {recall:10.3f} {f1:10.3f} {support:10d}")
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
    
    # Weighted average
    total = sum(supports)
    w_prec = sum(p * s for p, s in zip(precisions, supports)) / total
    w_rec = sum(r * s for r, s in zip(recalls, supports)) / total
    w_f1 = sum(f * s for f, s in zip(f1s, supports)) / total
    
    print("-" * 50)
    print(f"{'Avg':>8} {w_prec:10.3f} {w_rec:10.3f} {w_f1:10.3f} {total:10d}")

classification_report(y_spiral, preds, n_classes=3)
```

---

## 11. Final Exercises {#exercises}

### 📝 Exercise 11.1: Binary Classification from Scratch (Easy)

Train a logistic regression model on this dataset and compute the evaluation metrics:

```python
def exercise_binary():
    """
    TODO:
    1. Generate a linearly separable binary dataset
    2. Train LogisticRegression for 500 epochs
    3. Plot the decision boundary
    4. Compute accuracy, precision, recall, F1
    """
    np.random.seed(0)
    X0 = np.random.randn(2, 50) + np.array([[-2], [0]])
    X1 = np.random.randn(2, 50) + np.array([[2], [0]])
    X = np.hstack([X0, X1])
    y = np.hstack([np.zeros((1, 50)), np.ones((1, 50))])
    
    # TODO: Train and evaluate
    pass

# exercise_binary()
```

<details>
<summary>Solution</summary>

```python
def exercise_binary():
    np.random.seed(0)
    X0 = np.random.randn(2, 50) + np.array([[-2], [0]])
    X1 = np.random.randn(2, 50) + np.array([[2], [0]])
    X = np.hstack([X0, X1])
    y = np.hstack([np.zeros((1, 50)), np.ones((1, 50))])
    
    model = LogisticRegression(n_features=2)
    for epoch in range(500):
        model.forward(X)
        loss = model.compute_loss(y)
        model.backward(X, y, lr=0.5)
    
    preds = model.predict(X)
    acc = np.mean(preds == y) * 100
    
    tp = np.sum((preds == 1) & (y == 1))
    fp = np.sum((preds == 1) & (y == 0))
    fn = np.sum((preds == 0) & (y == 1))
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    
    print(f"Accuracy:  {acc:.1f}%")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    
    plot_logistic_result(model, X, y)

exercise_binary()
```
</details>

---

### 📝 Exercise 11.2: Softmax by Hand (Easy)

**a)** Compute softmax for $\mathbf{z} = [0, 0, 0]$. What do you get?

**b)** Compute softmax for $\mathbf{z} = [10, 0, 0]$. What happens?

**c)** Compute softmax for $\mathbf{z} = [100, 0, 0]$. Why is the numerical stability trick needed?

<details>
<summary>Solution</summary>

**a)** $e^0 = 1$ for all, sum = 3, so softmax = $[\frac{1}{3}, \frac{1}{3}, \frac{1}{3}]$.

Equal logits → **uniform distribution** (maximum uncertainty).

**b)** $e^{10} \approx 22026$, $e^0 = 1$, sum $\approx 22028$.

Softmax $\approx [0.9999, 0.00005, 0.00005]$ — almost all probability on class 1.

**c)** $e^{100} \approx 2.69 \times 10^{43}$. This is still representable in float64, but for larger values (e.g., $z = 1000$), we get **overflow** ($e^{1000} = \infty$).

The stability trick: subtract $\max(\mathbf{z}) = 100$, so we compute $e^{0}, e^{-100}, e^{-100}$ instead. The result is the same but no overflow occurs.
</details>

---

### 📝 Exercise 11.3: Multi-Class Classifier on Concentric Circles (Medium)

Classify points into 3 rings (inner, middle, outer):

```python
def generate_rings(n_per_class=100):
    """Three concentric rings"""
    np.random.seed(42)
    N = n_per_class * 3
    X = np.zeros((2, N))
    y = np.zeros(N, dtype=int)
    
    for k, (r_min, r_max) in enumerate([(0.0, 0.4), (0.5, 0.9), (1.0, 1.4)]):
        start = k * n_per_class
        end = start + n_per_class
        r = np.random.uniform(r_min, r_max, n_per_class)
        theta = np.random.uniform(0, 2 * np.pi, n_per_class)
        X[0, start:end] = r * np.cos(theta)
        X[1, start:end] = r * np.sin(theta)
        y[start:end] = k
    
    y_oh = np.zeros((3, N))
    y_oh[y, np.arange(N)] = 1
    idx = np.random.permutation(N)
    return X[:, idx], y_oh[:, idx], y[idx]

# TODO:
# 1. Generate the dataset and visualize it
# 2. Create a MultiClassMLP (experiment with hidden layer size)
# 3. Train and plot loss curve
# 4. Visualize decision regions
# 5. Print confusion matrix and classification report
```

<details>
<summary>Hints</summary>

- Concentric rings need a **non-linear** boundary — logistic regression won't work!
- Try 20-50 hidden neurons
- Learning rate around 0.5-1.0
- Train for 3000-5000 epochs
</details>

<details>
<summary>Solution</summary>

```python
X_rings, y_rings_oh, y_rings = generate_rings()

# Train
np.random.seed(42)
mlp_rings = MultiClassMLP(n_input=2, n_hidden=30, n_classes=3)
losses = []

for epoch in range(5000):
    mlp_rings.forward(X_rings)
    loss = mlp_rings.compute_loss(y_rings_oh)
    losses.append(loss)
    mlp_rings.backward(y_rings_oh, lr=0.8)

preds = mlp_rings.predict(X_rings)
print(f"Accuracy: {np.mean(preds == y_rings) * 100:.1f}%")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(losses)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('CCE Loss')
axes[0].set_title('Training Loss')
axes[0].grid(True, alpha=0.3)

ax = axes[1]
xx, yy = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-2, 2, 200))
grid = np.vstack([xx.ravel(), yy.ravel()])
Z = mlp_rings.predict(grid).reshape(xx.shape)
ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5],
            colors=['#ADD8E6', '#FFCCCB', '#90EE90'], alpha=0.4)
for k, c in enumerate(['blue', 'red', 'green']):
    mask = y_rings == k
    ax.scatter(X_rings[0, mask], X_rings[1, mask], c=c, alpha=0.6,
               edgecolors='black', s=20, label=f'Ring {k}')
ax.set_title('Decision Regions')
ax.legend()
ax.axis('equal')

plt.tight_layout()
plt.show()

classification_report(y_rings, preds, n_classes=3)
```
</details>

---

### 📝 Exercise 11.4: Compare MSE vs Cross-Entropy (Hard)

Train the **same MLP architecture** on the spiral dataset with two different loss functions and compare convergence:

```python
def loss_comparison():
    """
    TODO:
    1. Implement an MLP variant that uses MSE loss instead of CCE
       (Hint: use sigmoid output instead of softmax, with MSE)
    2. Train both versions on the spiral dataset for 5000 epochs
    3. Plot both loss curves on the same graph
    4. Compare final accuracy
    5. Explain the difference
    """
    pass
```

<details>
<summary>Discussion</summary>

You should observe:
- **Cross-entropy** converges faster, especially in early epochs
- **MSE** can get stuck in flat regions (sigmoid saturation problem)
- **Cross-entropy** typically achieves higher final accuracy

This demonstrates **why cross-entropy is the standard loss for classification** — it provides stronger gradients when the model is wrong, which is exactly when we need them most.
</details>

---

### 📝 Exercise 11.5: Decision Threshold Optimization (Hard)

For a binary classifier, find the optimal threshold by evaluating F1 score across a range of thresholds:

```python
def find_optimal_threshold(model, X, y):
    """
    TODO:
    1. Get predicted probabilities from the model
    2. For thresholds in [0.1, 0.2, ..., 0.9]:
       a. Compute predictions at this threshold
       b. Compute precision, recall, F1
    3. Plot precision, recall, and F1 vs threshold
    4. Return the threshold that maximizes F1
    """
    pass

# Test on the binary dataset
```

<details>
<summary>Solution</summary>

```python
def find_optimal_threshold(model, X, y):
    probs = model.forward(X)
    thresholds = np.arange(0.1, 0.95, 0.05)
    
    precisions, recalls, f1s = [], [], []
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = np.sum((preds == 1) & (y == 1))
        fp = np.sum((preds == 1) & (y == 0))
        fn = np.sum((preds == 0) & (y == 1))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, 'b-o', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, 'r-o', label='Recall', linewidth=2)
    plt.plot(thresholds, f1s, 'g-o', label='F1 Score', linewidth=2)
    
    best_idx = np.argmax(f1s)
    plt.axvline(x=thresholds[best_idx], color='gray', linestyle='--', 
                label=f'Best F1 at t={thresholds[best_idx]:.2f}')
    
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Precision, Recall, F1 vs Decision Threshold', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Optimal threshold: {thresholds[best_idx]:.2f}")
    print(f"Best F1: {f1s[best_idx]:.3f}")
    
    return thresholds[best_idx]
```
</details>

---

## Summary

### What We Learned

✅ **Sigmoid for probabilities**: Output of logistic regression is $P(y=1|\mathbf{x})$  
✅ **MSE fails for classification**: Slow gradients when sigmoid is saturated  
✅ **Binary cross-entropy**: $-[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ — steep penalty for wrong predictions  
✅ **Softmax**: Turns $K$ logits into a probability distribution that sums to 1  
✅ **Categorical cross-entropy**: $-\log(\hat{y}_c)$ — penalizes low probability for the correct class  
✅ **Evaluation metrics**: Accuracy, precision, recall, F1, confusion matrix  
✅ **Decision threshold**: Tunable trade-off between precision and recall

### Key Insights

1. **Match loss to task:**
   - Regression → MSE
   - Binary classification → sigmoid + BCE
   - Multi-class classification → softmax + CCE

2. **The gradient tells the story:**
   - MSE + sigmoid: $\frac{\partial L}{\partial z} = 2(\hat{y} - y) \cdot \sigma'(z)$ — saturates!
   - BCE + sigmoid: $\frac{\partial L}{\partial z} = \hat{y} - y$ — clean and strong
   - CCE + softmax: $\frac{\partial L}{\partial z_k} = \hat{y}_k - y_k$ — same beautiful form

3. **Accuracy is not enough:**
   - Precision matters when false positives are costly
   - Recall matters when false negatives are costly
   - F1 balances both
   - The confusion matrix shows the full picture

### What's Next?

**Session 8: Generalization & Regularization**

In the next session, we'll learn:
- **Train/validation/test splits**: How to evaluate honestly
- **Overfitting**: When a model memorizes instead of learning
- **Regularization**: L1, L2, dropout — tools to prevent overfitting
- **Modern optimizers**: Momentum, Adam — beyond basic SGD

**The goal:** Build models that work on **new, unseen** data, not just the training set!

### Before Next Session

**Think about:**
1. Our spiral classifier gets 95% accuracy on training data. Does that mean it will work well on new spirals?
2. What if we increased the hidden layer to 500 neurons? Would that help or hurt?
3. How would you know if your model is **too simple** vs **too complex**?

**Optional reading:**
- Chapter 7 of Goodfellow et al., "Deep Learning" (Regularization)
- Andrew Ng's "Advice for applying ML" lecture

---

**End of Session 7** 🎓

**You now understand:**
- ✅ How to build proper classification models with the right loss functions
- ✅ How softmax and cross-entropy work together
- ✅ How to evaluate classifiers beyond simple accuracy

**Next up:** Making models generalize to new data! 🚀
