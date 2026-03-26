# Project 2 — Justify Your Architecture

**Neural Networks — Practical Project**  
Duration: ~2 hours | group 2-3

---

## Overview

You are given three 2D synthetic datasets and a working MLP class. For each dataset, you must:

1. Train an MLP that achieves good classification
2. Plot the learned decision boundary
3. Write an **Architecture Card** justifying every hyperparameter choice

The goal is not just to make something work — it is to understand *why* your choices work.

---

## Setup — MLP and dataset generators

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

class MLP:
    def __init__(self, n_input, n_hidden, n_output, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(n_hidden, n_input) * 0.5
        self.b1 = np.zeros((n_hidden, 1))
        self.W2 = np.random.randn(n_output, n_hidden) * 0.5
        self.b2 = np.zeros((n_output, 1))

    def forward(self, X):
        self.z1 = self.W1 @ X + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.W2 @ self.a1 + self.b2
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


# ── Dataset generators ─────────────────────────────────────────────────────

def make_xor(n=200, noise=0.1, seed=0):
    np.random.seed(seed)
    X = np.random.randn(2, n)
    y = ((X[0] * X[1]) > 0).astype(float).reshape(1, -1)
    X += np.random.randn(2, n) * noise
    return X, y

def make_spiral(n=300, noise=0.2, seed=0):
    np.random.seed(seed)
    N = n // 2
    theta = np.linspace(0, 4 * np.pi, N)
    r = np.linspace(0.1, 1.0, N)
    X0 = np.vstack([r * np.cos(theta), r * np.sin(theta)])
    X1 = np.vstack([r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)])
    X = np.hstack([X0, X1]) + np.random.randn(2, n) * noise
    y = np.hstack([np.zeros(N), np.ones(N)]).reshape(1, -1)
    return X, y

def make_gaussians(n=300, sep=1.5, seed=0):
    np.random.seed(seed)
    N = n // 2
    X0 = np.random.randn(2, N) + np.array([[-sep], [0]])
    X1 = np.random.randn(2, N) + np.array([[sep], [0]])
    X = np.hstack([X0, X1])
    y = np.hstack([np.zeros(N), np.ones(N)]).reshape(1, -1)
    return X, y


# ── Decision boundary helper ────────────────────────────────────────────────

def plot_boundary(mlp, X, y, title=""):
    x_min, x_max = X[0].min() - 0.5, X[0].max() + 0.5
    y_min, y_max = X[1].min() - 0.5, X[1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.vstack([xx.ravel(), yy.ravel()])
    Z = mlp.forward(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1],
                 colors=["#ADD8E6", "#FFCCCB"], alpha=0.4)
    plt.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)
    mask = y[0] == 0
    plt.scatter(X[0, mask], X[1, mask], c="blue", alpha=0.5, label="Class 0", s=20)
    plt.scatter(X[0, ~mask], X[1, ~mask], c="red", alpha=0.5, label="Class 1", s=20)
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.show()
```

---

## Your Tasks

### Task 1 — Train a model on each dataset

For each of the three datasets, train an MLP and produce:

- A **loss curve** (epochs vs MSE, log scale)
- A **decision boundary plot** using `plot_boundary`
- Final training accuracy (percentage of correctly classified points)

```python
X_xor,     y_xor     = make_xor()
X_spiral,  y_spiral  = make_spiral()
X_gauss,   y_gauss   = make_gaussians()
```

You are free to choose any architecture and hyperparameters. Start simple and adjust based on what you observe.

---

### Task 2 — Write an Architecture Card for each dataset

For each dataset, fill in the following card. Every field requires a justification — not just the value.

---

**Architecture Card — Dataset: XOR / Spiral / Gaussians** *(one card per dataset)*

| Hyperparameter | Value chosen | Justification |
|----------------|-------------|---------------|
| Number of hidden neurons | | |
| Activation function | | |
| Learning rate | | |
| Number of epochs | | |

**What is the geometric shape of the decision boundary your model needs to learn?**  
*(e.g. "two parallel lines", "a closed curve", "a single line")*

**How does each hidden neuron contribute to building that boundary?**  
*(explain in terms of what a single sigmoid neuron computes geometrically)*

**What happened when you tried a configuration that did NOT work?**  
*(describe one failed attempt: what you tried, what you observed, why it failed)*

---

### Task 3 — Cross-dataset comparison

Answer the following questions in writing (3–5 sentences each):

**Q1.** Your three models likely use different numbers of hidden neurons. Rank the three datasets by how many hidden neurons they need, from fewest to most. Justify the ranking geometrically — why does one dataset require more capacity than another?

**Q2.** The learning rate controls the step size in gradient descent. Did you use the same learning rate for all three datasets? If not, what guided your choice? If yes, do you think it is optimal for all three — why or why not?

**Q3.** Look at your three loss curves. Which dataset converges fastest? Explain why in terms of the loss landscape — what property of the data makes optimization easier or harder?

---

### Task 4 — Stress test (required)

Pick your best model on the spiral dataset. Retrain it with **each of the following modifications**, one at a time, keeping everything else fixed. For each, report what happens to the final loss and decision boundary.

| Modification | What you observe | Your explanation |
|---|---|---|
| Reduce hidden neurons to 1 | | |
| Set learning rate to 5.0 | | |
| Set learning rate to 0.001 | | |
| Train for only 100 epochs | | |

---

## Deliverables

- A `.py` or `.ipynb` file with all training code, loss curves, and boundary plots
- The three Architecture Cards (Task 2) and written answers (Task 3) as a separate document
- The stress-test table (Task 4)

---

## What to bring to the oral

Be ready to:
- Point to any line of your loss curve and explain what is happening at that moment
- Answer "what would happen if you removed one hidden neuron?" for any of your models
- Explain the difference in capacity needed between two datasets using a sketch on the board
