# Project 2 — Justify Your Architecture

**Neural Networks — Practical Project**  
Duration: ~2 hours | Individual or pair

---

## Overview

You are given three 2D synthetic datasets and a working MLP. For each dataset you must train a model, plot the decision boundary, and **justify every choice in writing using equations** — not just intuition.

The goal is not to find the best model. It is to understand precisely why your model works, why a simpler one would not, and to predict outcomes before observing them.

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


# ── Dataset generators ──────────────────────────────────────────────────────

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

X_xor,    y_xor    = make_xor()
X_spiral, y_spiral = make_spiral()
X_gauss,  y_gauss  = make_gaussians()
```

---

## Task 1 — Train a model on each dataset

For each of the three datasets, train an MLP and produce:

- A **loss curve** (MSE vs epochs, log scale)
- A **decision boundary plot** using `plot_boundary`
- Final training accuracy (percentage of correctly classified points)

You are free to choose any architecture and hyperparameters. Start with the simplest architecture that could plausibly work, then increase complexity only if needed.

---

## Task 2 — Architecture Cards

For each dataset, fill in the card below completely. The justification column must contain either an equation, a geometric argument, or a reference to a specific observation from your loss curve. Answers like "I tried it and it worked" are not accepted.

---

**Architecture Card — Dataset: _______________**  
*(fill one card per dataset)*

| Hyperparameter | Value chosen | Justification — equation or geometric argument |
|----------------|-------------|------------------------------------------------|
| Number of hidden neurons | | |
| Activation function | | |
| Learning rate | | |
| Number of epochs | | |

**Minimum neuron count:**  
What is the minimum number of hidden neurons that could theoretically solve this dataset? Argue from geometry — what is the simplest set of half-planes whose intersection or union produces the required boundary?

**Decision boundary geometry:**  
Describe the shape of the boundary your trained model must produce. Sketch it on paper and include the sketch (photo is fine).

**One failed attempt:**  
Describe one architecture or hyperparameter choice that did not work. Report the symptom (loss curve shape, boundary shape) and explain the failure with reference to gradient descent or network capacity.

---

## Task 3 — Hidden neuron geometry by hand

This task must be completed **without modifying the trained model** — only read its weights.

After training your model on the **XOR dataset**, extract the weights of hidden neuron number 1 (first row of $W^{(1)}$, first element of $b^{(1)}$):

```python
w1 = mlp_xor.W1[0, :]   # shape (2,)
b1 = mlp_xor.b1[0, 0]   # scalar
print("w1:", w1, "  b1:", b1)
```

**3a.** A sigmoid neuron is close to 0.5 — its most uncertain point — when its pre-activation is zero:

$$w_{1,1} \cdot x_1 + w_{1,2} \cdot x_2 + b_1 = 0$$

This is the equation of a line in input space. Rearrange it into the form $x_2 = \alpha x_1 + \beta$ and compute $\alpha$ and $\beta$ numerically using your trained weights.

**3b.** Draw this line on your XOR boundary plot. Does it pass through a region where the decision boundary changes? Explain why this neuron's line is positioned where it is.

**3c.** Repeat for hidden neuron number 2. Do the two lines together explain the shape of the full decision boundary? Explain in 3–4 sentences, referring to how the output neuron combines $a^{(1)}_1$ and $a^{(1)}_2$.

---

## Task 4 — Stress test with predictions

Pick your trained model on the **spiral dataset**. Before running each modification, write your prediction. Submit the prediction column **before running any code** — it will be collected separately.

For each row: fill the prediction first, then run, then fill the result and explanation.

| Modification | Prediction (before running) | Actual result | Explanation of gap |
|---|---|---|---|
| Reduce hidden neurons to 1 | | | |
| Set learning rate to 5.0 | | | |
| Set learning rate to 0.001 | | | |
| Train for only 100 epochs | | | |
| Remove all biases (set $b^{(1)} = b^{(2)} = 0$ permanently) | | | |

**Prediction format:** for each row, write at minimum: expected final loss (order of magnitude), expected boundary shape (sketch or description), and one sentence of justification using gradient descent or geometry.

---

## Task 5 — Boundary-to-architecture matching

Below are four decision boundary descriptions. For each one, identify which dataset it most likely came from, what architecture (number of hidden neurons) most likely produced it, and justify your answer.

You may not run any code for this task — reason from what you know about what each architecture can represent.

---

**Boundary A:** A single smooth S-shaped curve running diagonally across the input space, cleanly separating left from right with no loops or islands.

- Dataset: _______________
- Architecture (hidden neurons): _______________
- Justification:

---

**Boundary B:** Two roughly diagonal bands of class 1 separated by class 0 regions in the corners — the boundary consists of two nearly parallel curves.

- Dataset: _______________
- Architecture (hidden neurons): _______________
- Justification:

---

**Boundary C:** A highly irregular boundary with several alternating class regions, roughly following a spiral path outward from the center. The boundary crosses itself multiple times.

- Dataset: _______________
- Architecture (hidden neurons): _______________
- Justification:

---

**Boundary D:** A smooth boundary similar to C but coarser — it captures the general spiral direction but misses several points near the center, leaving a visible misclassified cluster.

- Dataset: _______________
- Architecture (hidden neurons): _______________
- Justification (why fewer neurons produces this coarser result):

---

## Task 6 — Design from a symptom

You are given three loss curve descriptions from models trained by someone else. For each, diagnose the problem and prescribe the minimal fix. Your fix must be expressed as a concrete change (new neuron count, new learning rate value, etc.) with a justification that uses at least one equation from the course.

---

**Symptom 1:** The loss decreases smoothly for 500 epochs then completely plateaus at MSE ≈ 0.25 on the spiral dataset. The decision boundary is a single diagonal line.

- Diagnosis:
- Prescribed fix (concrete values):
- Justification with equation:

---

**Symptom 2:** The loss drops rapidly for the first 50 epochs on the gaussian dataset, then begins oscillating — alternating between 0.05 and 0.20 every few epochs — and never stabilises.

- Diagnosis:
- Prescribed fix (concrete values):
- Justification with equation (refer to the gradient descent update rule):

---

**Symptom 3:** The model trains correctly on XOR (loss reaches 0.005) but when you add 20 more hidden neurons and retrain, the final loss is 0.008 — slightly worse — and the boundary has jagged, irregular edges in empty regions of input space.

- Diagnosis:
- Prescribed fix (concrete values):
- Justification — why does adding neurons sometimes hurt?

---

## Task 7 — Cross-dataset comparison

Answer the following questions. Each answer must contain at least one equation or numerical reference to your results.

**Q1.** Rank the three datasets by the minimum number of hidden neurons required to solve them. Justify each position in the ranking using a geometric argument about what the hidden neurons must collectively represent.

**Q2.** For the gaussian dataset, a single hidden neuron is sufficient. Write the equation of the decision line it learns, using symbolic weights $w_1, w_2, b$. Explain why this works for gaussians but not for XOR.

**Q3.** The spiral dataset requires significantly more neurons than the other two. Express this in terms of the number of sign changes the output function must make along any radial line from the origin — and relate this to the minimum number of hidden neurons needed.

---

## Deliverables

- A `.py` or `.ipynb` file with all training code, loss curves, and boundary plots
- Architecture Cards (Task 2), hand computation (Task 3), and written answers (Tasks 5, 6, 7)
- The **prediction column of the stress-test table** submitted separately before running code

---

## What to bring to the oral

Be ready to:
- Point to any neuron in your XOR model and draw its decision line on the board using its weights
- Be given a loss curve on the spot and diagnose it without running code
- Justify why the spiral needs more neurons than XOR using a geometric argument drawn live
- Answer: *"Your gaussian model uses $n$ hidden neurons. I claim 1 is enough — prove or disprove this, and write the equation of the boundary that single neuron would learn"*
