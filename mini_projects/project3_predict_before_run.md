# Project 3 — Predict Before You Run

**Neural Networks — Practical Project**  
Duration: ~2 hours | group 2-3

---

## Overview

You are given a complete, working MLP. You will run 5 experiments on it — but there is a strict rule:

> **Before running each experiment, you must write your prediction.**

Your prediction must include:
- What you expect to happen (loss curve behaviour, final accuracy, decision boundary)
- A justification using equations or reasoning from the course

After running, you record the actual result and explain any gap between prediction and reality.

A wrong prediction that is well-explained is worth more than a correct prediction with no reasoning.

---

## Setup

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

class MLP:
    def __init__(self, n_input, n_hidden, n_output, seed=42,
                 use_bias=True, zero_init=False):
        self.use_bias = use_bias
        if zero_init:
            self.W1 = np.zeros((n_hidden, n_input))
            self.W2 = np.zeros((n_output, n_hidden))
        else:
            np.random.seed(seed)
            self.W1 = np.random.randn(n_hidden, n_input) * 0.5
            self.W2 = np.random.randn(n_output, n_hidden) * 0.5

        self.b1 = np.zeros((n_hidden, 1))
        self.b2 = np.zeros((n_output, 1))

    def forward(self, X):
        self.z1 = self.W1 @ X + (self.b1 if self.use_bias else 0)
        self.a1 = sigmoid(self.z1)
        self.z2 = self.W2 @ self.a1 + (self.b2 if self.use_bias else 0)
        self.a2 = sigmoid(self.z2)
        return self.a2

    def compute_loss(self, y):
        return np.mean((self.a2 - y) ** 2)

    def backward(self, X, y, lr):
        N = X.shape[1]
        dL_da2 = 2 * (self.a2 - y) / N
        delta2 = dL_da2 * sigmoid_derivative(self.z2)
        dW2 = delta2 @ self.a1.T
        db2 = np.sum(delta2, axis=1, keepdims=True)
        delta1 = (self.W2.T @ delta2) * sigmoid_derivative(self.z1)
        dW1 = delta1 @ X.T
        db1 = np.sum(delta1, axis=1, keepdims=True)
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, X, y, lr, n_epochs):
        losses = []
        for _ in range(n_epochs):
            self.forward(X)
            losses.append(self.compute_loss(y))
            self.backward(X, y, lr)
        return losses


def plot_boundary(mlp, X, y, title=""):
    x_min, x_max = X[0].min() - 0.5, X[0].max() + 0.5
    y_min, y_max = X[1].min() - 0.5, X[1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.vstack([xx.ravel(), yy.ravel()])
    Z = mlp.forward(grid).reshape(xx.shape)
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1],
                 colors=["#ADD8E6", "#FFCCCB"], alpha=0.4)
    plt.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)
    mask = y[0] == 0
    plt.scatter(X[0, mask], X[1, mask], c="blue", s=20, alpha=0.6, label="Class 0")
    plt.scatter(X[0, ~mask], X[1, ~mask], c="red", s=20, alpha=0.6, label="Class 1")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.show()


# Reference setup: XOR dataset, 2-4-1 MLP, lr=0.5, 5000 epochs
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]], dtype=float)
y = np.array([[0, 1, 1, 0]], dtype=float)
```

---

## The 5 Experiments

For each experiment, you must complete the **prediction card before running any code**.

---

### Experiment 1 — Very high learning rate

**Setup:**
```python
mlp = MLP(n_input=2, n_hidden=4, n_output=1)
losses = mlp.train(X, y, lr=10.0, n_epochs=5000)
```

**Prediction card** *(fill in before running)*

| Field | Your answer |
|-------|-------------|
| Will the network converge? | |
| Describe the shape of the loss curve | |
| What will the final loss be (order of magnitude)? | |
| Justify using the gradient descent update rule $w \leftarrow w - \eta \nabla L$ | |

**After running:**

| Field | Your answer |
|-------|-------------|
| Actual loss curve shape | |
| Final loss | |
| Was your prediction correct? | |
| Explanation of any gap | |

---

### Experiment 2 — Zero initialization

**Setup:**
```python
mlp = MLP(n_input=2, n_hidden=4, n_output=1, zero_init=True)
losses = mlp.train(X, y, lr=0.5, n_epochs=5000)
```

**Prediction card** *(fill in before running)*

| Field | Your answer |
|-------|-------------|
| Will the network learn anything? | |
| What will all hidden neurons compute after 1 epoch? | |
| Write $\delta^{(1)}_1$ and $\delta^{(1)}_2$ symbolically — are they equal? Show your work. | |
| What does the decision boundary look like after training? | |

**After running:**

| Field | Your answer |
|-------|-------------|
| Actual final loss | |
| Decision boundary (plot it) | |
| Was your prediction correct? | |
| Explanation of any gap | |

---

### Experiment 3 — Deeper network

**Setup:**
```python
# Add a second hidden layer manually (or use the architecture below)
# 2 inputs → 4 hidden → 2 hidden → 1 output
# Implement this as two stacked MLP blocks, or extend the MLP class

# Alternatively, compare:
mlp_shallow = MLP(n_input=2, n_hidden=4, n_output=1)
mlp_deep    = MLP(n_input=2, n_hidden=2, n_output=1)  # fewer neurons, same depth

losses_shallow = mlp_shallow.train(X, y, lr=0.5, n_epochs=5000)
losses_deep    = mlp_deep.train(X, y, lr=0.5, n_epochs=5000)
```

**Prediction card** *(fill in before running)*

| Field | Your answer |
|-------|-------------|
| Which will converge faster — 4 hidden neurons or 2? | |
| Which will reach a lower final loss? | |
| For XOR specifically, is more capacity always better? Why or why not? | |
| What is the minimum number of hidden neurons that can theoretically solve XOR? Justify. | |

**After running:**

| Field | Your answer |
|-------|-------------|
| Final loss — 4 neurons | |
| Final loss — 2 neurons | |
| Was your prediction correct? | |
| Explanation of any gap | |

---

### Experiment 4 — Training on only 2 of the 4 XOR points

**Setup:**
```python
# Train on only (0,1)→1 and (1,0)→1  (the two positive examples)
X_partial = np.array([[0, 1],
                      [1, 0]], dtype=float)
y_partial  = np.array([[1, 1]], dtype=float)

mlp = MLP(n_input=2, n_hidden=4, n_output=1)
losses = mlp.train(X_partial, y_partial, lr=0.5, n_epochs=5000)

# Then evaluate on all 4 points
print(mlp.forward(X).round(3))
```

**Prediction card** *(fill in before running)*

| Field | Your answer |
|-------|-------------|
| Will training loss go to 0? | |
| What will the network predict for (0,0) and (1,1)? | |
| Is the learned model correct for XOR? | |
| What does this tell you about generalisation vs memorisation? | |

**After running:**

| Field | Your answer |
|-------|-------------|
| Training loss after 5000 epochs | |
| Predictions on all 4 XOR points | |
| Was your prediction correct? | |
| Explanation of any gap | |

---

### Experiment 5 — No biases

**Setup:**
```python
mlp = MLP(n_input=2, n_hidden=4, n_output=1, use_bias=False)
losses = mlp.train(X, y, lr=0.5, n_epochs=5000)
plot_boundary(mlp, X, y, title="No biases")
```

**Prediction card** *(fill in before running)*

| Field | Your answer |
|-------|-------------|
| Can XOR be solved without biases? | |
| Without bias, what constraint does every hidden neuron's decision boundary satisfy? | |
| Sketch (on paper) the decision boundaries that hidden neurons can draw when $b=0$ | |
| What will the decision boundary look like after training? | |

**After running:**

| Field | Your answer |
|-------|-------------|
| Final loss | |
| Decision boundary (plot it) | |
| Was your prediction correct? | |
| Explanation of any gap | |

---

## Final Reflection

After completing all 5 experiments, answer these questions (5–8 sentences total):

**Which experiment surprised you most?** Describe what you expected, what happened, and what it taught you about how neural networks work.

**Across the 5 experiments, which factor had the biggest impact on whether the network learned correctly** — learning rate, initialization, architecture, or data? Justify with reference to your results.

---

## Deliverables

- **Before the session:** submit your 5 prediction cards (the top half of each table, with no code run yet) — these will be collected separately and compared to your final results
- **After the session:** the complete notebook with all runs, plots, and filled-in result tables
- The final reflection

---

## What to bring to the oral

Be ready to:
- Explain experiment 2 (zero initialization) by writing $\delta^{(1)}_1$ and $\delta^{(1)}_2$ on a whiteboard
- Sketch, without code, the decision boundary produced by a no-bias network on XOR
- Receive a 6th experiment on the spot and write a prediction card live
