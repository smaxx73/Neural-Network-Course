# Session 9: PyTorch Introduction
## From Scratch to Framework

**Course: Neural Networks for Engineers**  
**Duration: 2 hours**

---

## Table of Contents

### Part I — PyTorch Fundamentals (≈ 50 min)
1. [Recap & Motivation](#recap)
2. [Tensors: The Building Block](#tensors)
3. [Autograd: Automatic Differentiation](#autograd)
4. [nn.Module: Building Networks](#nn-module)
5. [The PyTorch Training Loop](#training-loop)

### Part II — Mini-Projects (≈ 70 min)
6. [Mini-Project A: From Perceptron to MLP in PyTorch](#project-a)
7. [Mini-Project B: The Full Pipeline — Spiral Classifier](#project-b)
8. [Mini-Project C: MNIST — Your First Real Dataset](#project-c)

---

# Part I — PyTorch Fundamentals

---

## 1. Recap & Motivation {#recap}

### What We've Built So Far (by Hand)

Over Sessions 2–8, we implemented **everything** from scratch in NumPy:

| Component | Session | Lines of code |
|---|---|---|
| Forward propagation | 4 | ~20 |
| Backpropagation | 6 | ~30 |
| Cross-entropy + softmax | 7 | ~15 |
| Early stopping | 8 | ~30 |
| L2 regularization | 8 | 3 lines added |
| Dropout | 8 | 2 lines added |
| SGD, Momentum, Adam | 8 | ~40 |
| **Total** | | **~170 lines** |

And we still only support **2-layer networks** with **one activation function**.

### The Problem

To add a third hidden layer, we'd need to:
- Write new forward pass code for the extra layer
- Write new backward pass code (more $\delta$ computations)
- Update the weight-saving logic in early stopping
- Update the gradient computation in FlexMLP
- Update every optimizer's velocity/moment lists

This doesn't scale. For a 50-layer ResNet, manual backprop would be **thousands of lines** of error-prone code.

### The Solution: PyTorch

PyTorch gives us:

| Our manual code | PyTorch equivalent |
|---|---|
| `model.forward()` + stored `z`, `a` | `model(x)` — automatic |
| `model.backward()` with chain rule | `loss.backward()` — **automatic for any graph** |
| `SGD`, `Adam` classes | `torch.optim.SGD`, `torch.optim.Adam` |
| `categorical_cross_entropy()` | `nn.CrossEntropyLoss()` |
| `RegularizedMLP` with manual L2 | `weight_decay` parameter in optimizer |
| Manual dropout mask + scaling | `nn.Dropout(p)` |
| NumPy arrays on CPU only | Tensors on **CPU or GPU** |

### 🤔 Quick Questions (from Session 8's "Think About")

**Q1:** What parts of our manual implementation were tedious and error-prone?

<details>
<summary>Click to reveal answer</summary>
Backpropagation: computing $\delta$ for each layer, getting the transposes right, remembering to store $z$ during forward pass, dividing by $N$, not regularizing biases. Gradient checking helped, but it was slow.
</details>

**Q2:** If a library handles gradients automatically, what do **you** still need to decide?

<details>
<summary>Click to reveal answer</summary>
The architecture (how many layers, how many neurons, which activations), the loss function, the optimizer and its hyperparameters, the training/validation split, when to stop, and how to evaluate the model. The **engineering decisions** remain yours — PyTorch automates the **calculus**.
</details>

---

## 2. Tensors: The Building Block {#tensors}

### What is a Tensor?

A tensor is PyTorch's version of a NumPy array — a multi-dimensional array of numbers. The key difference: tensors can **track gradients** and run on **GPUs**.

```python
import torch
import torch.nn as nn
import torch.optim as optim

print(f"PyTorch version: {torch.__version__}")
```

### Creating Tensors

```python
# From Python lists
a = torch.tensor([1.0, 2.0, 3.0])
print(f"From list:   {a}, shape: {a.shape}, dtype: {a.dtype}")

# From NumPy (zero-copy when possible!)
import numpy as np
np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
b = torch.from_numpy(np_array)
print(f"From numpy:  {b}, shape: {b.shape}")

# Common constructors (just like NumPy)
zeros = torch.zeros(3, 4)       # 3×4 of zeros
ones = torch.ones(2, 3)         # 2×3 of ones
rand = torch.randn(2, 3)       # 2×3 from N(0,1)
eye = torch.eye(3)              # 3×3 identity

# Range
r = torch.arange(0, 10, 2)      # [0, 2, 4, 6, 8]
l = torch.linspace(0, 1, 5)     # [0, 0.25, 0.5, 0.75, 1]

print(f"\nzeros: {zeros.shape}")
print(f"randn: {rand}")
print(f"arange: {r}")
```

### Tensor Operations

Almost everything works like NumPy:

```python
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# Element-wise operations
print(f"x + y = {x + y}")
print(f"x * y = {x * y}")       # Element-wise (NOT matrix multiply)
print(f"x ** 2 = {x ** 2}")

# Matrix operations
A = torch.randn(3, 4)
B = torch.randn(4, 2)
C = A @ B                        # Matrix multiply (same as NumPy!)
print(f"A @ B shape: {C.shape}")  # (3, 2)

# Reductions
print(f"sum:  {x.sum()}")
print(f"mean: {x.mean()}")
print(f"max:  {x.max()}")

# Reshaping
M = torch.arange(12).reshape(3, 4)
print(f"Reshaped:\n{M}")
```

### NumPy ↔ PyTorch Cheat Sheet

| NumPy | PyTorch | Notes |
|---|---|---|
| `np.array([1,2,3])` | `torch.tensor([1,2,3])` | |
| `np.zeros((3,4))` | `torch.zeros(3, 4)` | No tuple needed |
| `np.random.randn(3,4)` | `torch.randn(3, 4)` | |
| `A @ B` or `np.dot(A,B)` | `A @ B` or `torch.matmul(A,B)` | Same `@` syntax! |
| `A * B` | `A * B` | Element-wise |
| `np.sum(A, axis=0)` | `A.sum(dim=0)` | `axis` → `dim` |
| `np.max(A, axis=1)` | `A.max(dim=1)` | Returns `(values, indices)` |
| `A.reshape(3, -1)` | `A.reshape(3, -1)` or `A.view(3, -1)` | `view` is faster |
| `np.concatenate` | `torch.cat` | |
| `A.T` | `A.T` or `A.t()` | |

### Key Difference: Data Types

PyTorch defaults to `float32` (NumPy defaults to `float64`). This matters for GPU performance.

```python
x = torch.tensor([1, 2, 3])          # int64 by default
y = torch.tensor([1.0, 2.0, 3.0])    # float32 by default

print(f"int tensor dtype: {x.dtype}")   # torch.int64
print(f"float tensor dtype: {y.dtype}") # torch.float32

# Convert explicitly
x_float = x.float()   # int → float32
y_double = y.double()  # float32 → float64
```

---

## 3. Autograd: Automatic Differentiation {#autograd}

### The Magic of `requires_grad`

This is the feature that makes PyTorch revolutionary. Set `requires_grad=True` on a tensor, and PyTorch will **track every operation** to compute gradients automatically.

```python
# Create a tensor that tracks gradients
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)

# Forward pass: y = w * x + 1
y = w * x + 1
print(f"y = {y}")           # tensor(7., grad_fn=<AddBackward0>)

# Notice "grad_fn" — PyTorch recorded the computation!

# Backward pass: compute dy/dw
y.backward()
print(f"dy/dw = {w.grad}")  # tensor(3.) — which is x, as expected!
```

### How It Works

When `requires_grad=True`, PyTorch builds a **computational graph** — exactly like the ones we drew in Session 6:

```
w (requires_grad=True)
 │
 ├── [*] ── (w * x) ── [+] ── y
 │                       │
 x                       1

backward(): dy/dw = x = 3.0 ✓
```

### Autograd Replaces Our Manual Backprop

**Session 6 (manual):**
```python
# We had to compute every derivative by hand:
delta2 = (self.a2 - y_true) / N
dW2 = delta2 @ self.a1.T
delta1 = (self.W2.T @ delta2) * (self.z1 > 0).astype(float)
dW1 = delta1 @ self.X.T
# ... and pray we didn't make a mistake
```

**PyTorch (automatic):**
```python
loss = criterion(output, target)
loss.backward()   # ALL gradients computed automatically!
# w.grad now contains ∂loss/∂w for every parameter w
```

### Example: Linear Regression in 5 Lines

Remember the linear regression from Session 5? Here's the entire gradient computation in PyTorch:

```python
# Data
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y_true = torch.tensor([2.2, 3.8, 6.1, 7.9, 10.1])  # y ≈ 2x

# Parameters (trainable)
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# Forward + loss + backward
y_pred = w * x + b
loss = ((y_pred - y_true) ** 2).mean()   # MSE
loss.backward()                           # Compute ALL gradients

print(f"dL/dw = {w.grad:.4f}")
print(f"dL/db = {b.grad:.4f}")
```

### Important: `zero_grad()` and `no_grad()`

**Gradients accumulate** by default — you must zero them before each backward pass:

```python
w = torch.tensor(1.0, requires_grad=True)

# First backward
y1 = (w * 3) ** 2
y1.backward()
print(f"After first backward:  w.grad = {w.grad}")   # 18.0

# Second backward WITHOUT zeroing — WRONG!
y2 = (w * 3) ** 2
y2.backward()
print(f"Without zero_grad:     w.grad = {w.grad}")   # 36.0 (accumulated!)

# Correct: zero before each backward
w.grad.zero_()
y3 = (w * 3) ** 2
y3.backward()
print(f"After zero_grad:       w.grad = {w.grad}")   # 18.0 ✓
```

**Disable gradient tracking** for validation/inference:

```python
with torch.no_grad():
    # No graph built → faster, uses less memory
    val_output = model(val_data)
    val_loss = criterion(val_output, val_target)
```

### 🤔 Think About It

**Q:** In our manual code, we had to **store** all intermediate values ($z^{(l)}$, $a^{(l)}$) during the forward pass for use in the backward pass. Does PyTorch need us to do this?

<details>
<summary>Answer</summary>
**No!** When you compute operations on tensors with `requires_grad=True`, PyTorch automatically builds a graph that stores everything it needs. The `loss.backward()` call walks this graph in reverse. That's why it's called **automatic** differentiation — you just write the forward pass, and backward is free.
</details>

---

## 4. nn.Module: Building Networks {#nn-module}

### The nn.Module Pattern

Every PyTorch network inherits from `nn.Module` and implements two things:
1. `__init__`: define the layers
2. `forward`: define the computation

```python
class MyFirstNetwork(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()   # Always call this first!
        self.hidden = nn.Linear(n_input, n_hidden)   # Weights + bias, auto-initialized
        self.output = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.hidden(x))    # Hidden layer + ReLU
        x = self.output(x)               # Output layer (raw logits)
        return x

# Create and inspect
net = MyFirstNetwork(n_input=2, n_hidden=4, n_output=3)
print(net)
print(f"\nTotal parameters: {sum(p.numel() for p in net.parameters())}")
```

### What `nn.Linear` Does

`nn.Linear(in_features, out_features)` is exactly our manual $z = Wx + b$:

```python
layer = nn.Linear(3, 2)  # 3 inputs → 2 outputs

# It has weights and bias
print(f"Weight shape: {layer.weight.shape}")  # (2, 3)
print(f"Bias shape:   {layer.bias.shape}")    # (2,)
print(f"Weight:\n{layer.weight.data}")
print(f"Bias: {layer.bias.data}")
```

**Convention difference:** In PyTorch, input shape is `(N, features)` — batch first. In our NumPy code, we used `(features, N)`. PyTorch's convention is more common in practice.

| | Our NumPy code | PyTorch |
|---|---|---|
| Input shape | `(n_features, N)` | `(N, n_features)` |
| Weight shape | `(n_output, n_input)` | `(n_output, n_input)` |
| Operation | `W @ X + b` | `F.linear(x, W, b)` or `layer(x)` |

### nn.Sequential: Quick Network Building

For simple networks, `nn.Sequential` avoids writing a class:

```python
# These two are equivalent:

# Method 1: nn.Sequential
net_seq = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 3),
)

# Method 2: Custom class
class NetCustom(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(4, 3)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

net_custom = NetCustom()

# Both produce the same structure
print("Sequential:", net_seq)
print("\nCustom:", net_custom)
```

**When to use which:**

| Use `nn.Sequential` | Use custom `nn.Module` |
|---|---|
| Layers are a simple chain | Skip connections, multiple inputs/outputs |
| No conditional logic | Different behavior during train/eval |
| Quick prototyping | Complex architectures (ResNet, U-Net) |

### Common Layers and Activations

| Layer | Our NumPy | PyTorch |
|---|---|---|
| Fully connected | `W @ x + b` | `nn.Linear(in, out)` |
| ReLU | `np.maximum(0, z)` | `nn.ReLU()` |
| Sigmoid | `1 / (1 + np.exp(-z))` | `nn.Sigmoid()` |
| Dropout | Manual mask + scaling | `nn.Dropout(p)` |
| Batch normalization | (not covered) | `nn.BatchNorm1d(features)` |
| Softmax | Manual exp + normalize | `nn.Softmax(dim=1)` |

**Important:** For classification, PyTorch's `nn.CrossEntropyLoss` applies softmax internally. You do **not** add a softmax layer at the end of your network — just output raw logits.

---

## 5. The PyTorch Training Loop {#training-loop}

### The Standard Pattern

Every PyTorch training loop follows the same 5-step pattern:

```python
model = MyNetwork(...)
criterion = nn.CrossEntropyLoss()        # Loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer

for epoch in range(n_epochs):
    # ① Forward pass
    outputs = model(X_train)
    
    # ② Compute loss
    loss = criterion(outputs, y_train)
    
    # ③ Zero gradients (BEFORE backward!)
    optimizer.zero_grad()
    
    # ④ Backward pass (compute all gradients)
    loss.backward()
    
    # ⑤ Update weights
    optimizer.step()
```

**That's it.** Five lines inside the loop. Compare with our 20+ line manual loop!

### Side-by-Side: Manual vs PyTorch

| Step | Our Manual Code (Session 6–8) | PyTorch |
|---|---|---|
| Forward | `model.forward(X)` | `outputs = model(X)` |
| Loss | `model.loss(y)` | `loss = criterion(outputs, y)` |
| Zero grads | *(not needed — we computed fresh each time)* | `optimizer.zero_grad()` |
| Backward | `model.backward(X, y, lr)` | `loss.backward()` |
| Update | *(inside backward)* | `optimizer.step()` |
| Regularization | Manual `+ lambda * W` in gradient | `weight_decay=0.01` in optimizer |
| Dropout | Manual mask in `forward()` | `nn.Dropout(p)` — auto handles train/eval |

### Train vs Eval Mode

Dropout and batch normalization behave differently during training and inference. Toggle with:

```python
model.train()   # Enable dropout, batch norm in training mode
model.eval()    # Disable dropout, batch norm uses running statistics
```

### Complete Training Loop with Validation

```python
def train_model(model, X_train, y_train, X_val, y_val,
                criterion, optimizer, n_epochs, patience=50):
    """
    The complete PyTorch training loop with early stopping.
    """
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = model.state_dict().copy()   # Save best weights (PyTorch way!)
    
    for epoch in range(n_epochs):
        # ── Train ──
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())  # .item() → Python float
        
        # ── Validate ──
        model.eval()
        with torch.no_grad():             # No gradient needed for validation!
            val_out = model(X_val)
            val_loss = criterion(val_out, y_val)
            val_losses.append(val_loss.item())
        
        # ── Early stopping ──
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch} (best: {best_epoch})")
            break
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:5d}: Train {loss.item():.4f} | Val {val_loss.item():.4f}")
    
    model.load_state_dict(best_state)
    return train_losses, val_losses, best_epoch
```

### 🤔 Think About It

**Q:** Our manual early stopping required saving 4 arrays (W1, b1, W2, b2) with `.copy()`. For a 50-layer network, that would be 100+ arrays. How does PyTorch solve this?

<details>
<summary>Answer</summary>
`model.state_dict()` returns a dictionary of **all** parameters, regardless of how many layers there are. And `model.load_state_dict(state)` restores them all in one call. This is why frameworks scale — the code doesn't change when the architecture changes.
</details>

---

# Part II — Mini-Projects

---

## 6. Mini-Project A: From Perceptron to MLP in PyTorch {#project-a}

### 🎯 Goal

Rebuild the Perceptron (Session 2) and XOR MLP (Session 6) in PyTorch, seeing how autograd eliminates manual backprop.

**Skills reused:** Perceptron logic (Session 2), XOR problem (Session 4), backprop training (Session 6).

---

### Phase 1 — Linear Regression with Autograd

Before networks, let's see autograd in action on the simplest case: learning $y = 2x + 1$ with raw tensors.

**Task:** Complete the training loop using only tensors and autograd — no `nn.Module` yet.

```python
# Data: y = 2x + 1 + noise
torch.manual_seed(42)
x = torch.linspace(0, 5, 50)
y_true = 2 * x + 1 + torch.randn(50) * 0.5

# Learnable parameters
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

lr = 0.01
losses = []

for epoch in range(500):
    # TODO: Forward pass — compute y_pred = w * x + b
    y_pred = ___
    
    # TODO: Compute MSE loss
    loss = ___
    
    # TODO: Backward pass — compute gradients
    ___
    
    # TODO: Update weights manually (inside torch.no_grad() block!)
    # Why no_grad? Because we don't want the update to be tracked in the graph.
    with torch.no_grad():
        w -= ___
        b -= ___
    
    # TODO: Zero the gradients for next iteration
    w.grad.___
    b.grad.___
    
    losses.append(loss.item())

print(f"Learned: y = {w.item():.3f}x + {b.item():.3f}")
print(f"True:    y = 2.000x + 1.000")
```

<details>
<summary>Solution</summary>

```python
for epoch in range(500):
    y_pred = w * x + b
    loss = ((y_pred - y_true) ** 2).mean()
    loss.backward()
    
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    
    w.grad.zero_()
    b.grad.zero_()
    
    losses.append(loss.item())
```

Note how we **never computed a derivative by hand** — `loss.backward()` did it all!
</details>

### Phase 2 — Perceptron as nn.Module

**Task:** Implement a Perceptron classifier for the AND gate using `nn.Module`. Use `nn.Linear` and `nn.Sigmoid`.

```python
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Define a single linear layer: 2 inputs → 1 output
        self.linear = ___
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # TODO: Linear → Sigmoid
        return ___

# AND gate data (PyTorch convention: batch first → shape (N, features))
X_and = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y_and = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)

# TODO: Create model, loss function (BCELoss for binary), and optimizer (SGD, lr=1.0)
model = ___
criterion = ___
optimizer = ___

# TODO: Train for 1000 epochs (5-line loop!)
for epoch in range(1000):
    ___

# Test
with torch.no_grad():
    preds = model(X_and)
    print("AND gate results:")
    for i in range(4):
        x1, x2 = X_and[i]
        p = preds[i].item()
        print(f"  ({x1:.0f}, {x2:.0f}) → {p:.3f} → {int(p > 0.5)}")
```

<details>
<summary>Solution</summary>

```python
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = Perceptron()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=1.0)

for epoch in range(1000):
    outputs = model(X_and)
    loss = criterion(outputs, y_and)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
</details>

### Phase 3 — XOR MLP: The Moment of Truth

**Task:** Build and train an MLP to solve XOR. In Session 4, we did this manually. In Session 6, we implemented backprop. Now: just define the architecture and let PyTorch handle everything.

```python
class XOR_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Define a 2-layer MLP: 2 → 4 → 1
        # Hidden layer with ReLU, output with Sigmoid
        # Hint: Use nn.Sequential or define layers individually
        ___
    
    def forward(self, x):
        ___

# XOR data
X_xor = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y_xor = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# TODO: Create model, BCELoss, Adam optimizer (lr=0.01)
model = ___
criterion = ___
optimizer = ___

# TODO: Train for 5000 epochs, print loss every 1000
losses = []
for epoch in range(5000):
    ___
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Test
model.eval()
with torch.no_grad():
    preds = model(X_xor)
    print("\nXOR results:")
    for i in range(4):
        x1, x2 = X_xor[i]
        p = preds[i].item()
        print(f"  ({x1:.0f}, {x2:.0f}) → {p:.3f} → {int(p > 0.5)} "
              f"(true: {int(y_xor[i].item())})")
```

<details>
<summary>Solution</summary>

```python
class XOR_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.net(x)

model = XOR_MLP()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

losses = []
for epoch in range(5000):
    outputs = model(X_xor)
    loss = criterion(outputs, y_xor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```
</details>

### Phase 4 — Compare: Manual vs PyTorch

**Task:** Plot the XOR loss curves from Session 6 (manual backprop) and this session (PyTorch) on the same graph.

```python
import matplotlib.pyplot as plt

# Session 6 manual version (simplified re-run)
def sigmoid_np(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

class ManualMLP:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(4, 2) * np.sqrt(2.0 / 2)
        self.b1 = np.zeros((4, 1))
        self.W2 = np.random.randn(1, 4) * np.sqrt(2.0 / 4)
        self.b2 = np.zeros((1, 1))
    
    def train_epoch(self, X, y, lr):
        self.z1 = self.W1 @ X + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = sigmoid_np(self.z2)
        loss = np.mean((y - self.a2) ** 2)
        
        N = X.shape[1]
        da2 = -2 * (y - self.a2) / N
        dz2 = da2 * self.a2 * (1 - self.a2)
        self.W2 -= lr * (dz2 @ self.a1.T)
        self.b2 -= lr * np.sum(dz2, axis=1, keepdims=True)
        dz1 = (self.W2.T @ dz2) * (self.z1 > 0).astype(float)
        self.W1 -= lr * (dz1 @ X.T)
        self.b1 -= lr * np.sum(dz1, axis=1, keepdims=True)
        return loss

X_np = np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=float)
y_np = np.array([[0, 1, 1, 0]], dtype=float)

manual_model = ManualMLP(seed=42)
manual_losses = [manual_model.train_epoch(X_np, y_np, lr=2.0) for _ in range(5000)]

# TODO: Plot both loss curves on the same axes
# Blue line for manual, orange line for PyTorch
# Add xlabel, ylabel, title, legend, grid, log scale on y-axis
fig, ax = plt.subplots(figsize=(10, 6))
___

plt.show()
```

<details>
<summary>Solution</summary>

```python
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(manual_losses, label='Manual (NumPy, lr=2.0)', linewidth=2, alpha=0.7)
ax.plot(losses, label='PyTorch (Adam, lr=0.01)', linewidth=2, alpha=0.7)
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Loss', fontsize=14)
ax.set_title('XOR Training: Manual vs PyTorch', fontsize=16)
ax.set_yscale('log')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.show()
```
</details>

---

## 7. Mini-Project B: The Full Pipeline — Spiral Classifier {#project-b}

### 🎯 Goal

Rebuild the spiral classifier from Session 7–8 entirely in PyTorch: data preparation, model, training loop with early stopping, evaluation, and decision boundary visualization.

**Skills reused:** Spiral dataset (Session 7), train/val/test split (Session 8), learning curves (Session 8), evaluation metrics (Session 7).

---

### Phase 1 — Prepare Data as Tensors

**Task:** Generate the spiral dataset, split it, and convert everything to PyTorch tensors. Watch out for the shape convention change!

```python
# Generate data (NumPy)
from session8_toolkit import generate_spiral, train_val_test_split  # or copy from Session 8

np.random.seed(42)
N_per_class = 150
N_classes = 3
N = N_per_class * N_classes
X_np = np.zeros((2, N))
y_np = np.zeros(N, dtype=int)
for k in range(N_classes):
    s, e = k * N_per_class, (k + 1) * N_per_class
    r = np.linspace(0.2, 1.0, N_per_class)
    theta = np.linspace(k * 4.0, (k + 1) * 4.0, N_per_class) + np.random.randn(N_per_class) * 0.25
    X_np[0, s:e] = r * np.cos(theta)
    X_np[1, s:e] = r * np.sin(theta)
    y_np[s:e] = k

# Shuffle
idx = np.random.permutation(N)
X_np, y_np = X_np[:, idx], y_np[idx]

# Split (70/15/15)
n_test = int(N * 0.15)
n_val = int(N * 0.15)
n_train = N - n_val - n_test

X_train_np, y_train_np = X_np[:, :n_train], y_np[:n_train]
X_val_np, y_val_np = X_np[:, n_train:n_train+n_val], y_np[n_train:n_train+n_val]
X_test_np, y_test_np = X_np[:, n_train+n_val:], y_np[n_train+n_val:]

# TODO: Convert to PyTorch tensors
# IMPORTANT: Transpose X from (2, N) to (N, 2) — PyTorch uses batch-first!
# IMPORTANT: y should be LongTensor for CrossEntropyLoss (class indices, not one-hot)
X_train = torch.tensor(___, dtype=torch.float32)   # shape: (n_train, 2)
y_train = torch.tensor(___, dtype=torch.long)       # shape: (n_train,)
X_val = torch.tensor(___, dtype=torch.float32)
y_val = torch.tensor(___, dtype=torch.long)
X_test = torch.tensor(___, dtype=torch.float32)
y_test = torch.tensor(___, dtype=torch.long)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
# Expected: X_train: torch.Size([315, 2]), y_train: torch.Size([315])
```

<details>
<summary>Solution</summary>

```python
X_train = torch.tensor(X_train_np.T, dtype=torch.float32)  # (N, 2)
y_train = torch.tensor(y_train_np, dtype=torch.long)        # (N,)
X_val = torch.tensor(X_val_np.T, dtype=torch.float32)
y_val = torch.tensor(y_val_np, dtype=torch.long)
X_test = torch.tensor(X_test_np.T, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.long)
```

Key points:
- `.T` transposes from `(features, N)` to `(N, features)` — PyTorch convention
- `dtype=torch.long` for class labels — `CrossEntropyLoss` requires integer indices, not one-hot
</details>

### Phase 2 — Build the Model

**Task:** Define a multi-class MLP with regularization. Use `nn.Sequential` with:
- Linear(2 → 100) → ReLU → Dropout(0.2) → Linear(100 → 3)

**No softmax at the end** — `CrossEntropyLoss` includes it.

```python
class SpiralNet(nn.Module):
    def __init__(self, n_hidden=100, dropout=0.2):
        super().__init__()
        # TODO: Define the network using nn.Sequential
        # Layers: Linear(2, n_hidden) → ReLU → Dropout(dropout) → Linear(n_hidden, 3)
        self.net = nn.Sequential(
            ___
        )
    
    def forward(self, x):
        return self.net(x)

model = SpiralNet(n_hidden=100, dropout=0.2)
print(model)
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
# Expected: 2*100 + 100 + 100*3 + 3 = 603 parameters
```

<details>
<summary>Solution</summary>

```python
class SpiralNet(nn.Module):
    def __init__(self, n_hidden=100, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, 3),
        )
    
    def forward(self, x):
        return self.net(x)
```
</details>

### Phase 3 — Train with Early Stopping

**Task:** Write the full training loop. Use `nn.CrossEntropyLoss` and `Adam` with `weight_decay` for L2 regularization.

```python
# TODO: Create model, loss, optimizer
# Use weight_decay=0.005 for L2 regularization (replaces our manual lambda!)
model = SpiralNet(n_hidden=100, dropout=0.2)
criterion = ___
optimizer = ___

train_losses, val_losses = [], []
best_val_loss = float('inf')
best_epoch = 0
best_state = None
patience = 200
n_epochs = 5000

for epoch in range(n_epochs):
    # ── Train ──
    model.train()     # Enables dropout
    
    # TODO: The 5-step training pattern
    outputs = ___
    loss = ___
    ___              # zero_grad
    ___              # backward
    ___              # step
    
    train_losses.append(loss.item())
    
    # ── Validate ──
    model.eval()      # Disables dropout
    with torch.no_grad():
        val_out = model(X_val)
        val_loss = criterion(val_out, y_val)
        val_losses.append(val_loss.item())
    
    # ── Early stopping ──
    # TODO: Track best val loss and save best model state
    if ___:
        best_val_loss = val_loss.item()
        best_epoch = epoch
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    if epoch - best_epoch >= patience:
        print(f"Early stopping at epoch {epoch} (best: {best_epoch})")
        break
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d}: Train {loss.item():.4f} | Val {val_loss.item():.4f}")

# TODO: Restore best weights
model.load_state_dict(___)
print(f"\nBest epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
```

<details>
<summary>Solution</summary>

```python
model = SpiralNet(n_hidden=100, dropout=0.2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.005)

train_losses, val_losses = [], []
best_val_loss = float('inf')
best_epoch = 0
best_state = None
patience = 200
n_epochs = 5000

for epoch in range(n_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    model.eval()
    with torch.no_grad():
        val_out = model(X_val)
        val_loss = criterion(val_out, y_val)
        val_losses.append(val_loss.item())
    
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        best_epoch = epoch
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    if epoch - best_epoch >= patience:
        print(f"Early stopping at epoch {epoch} (best: {best_epoch})")
        break
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d}: Train {loss.item():.4f} | Val {val_loss.item():.4f}")

model.load_state_dict(best_state)
```
</details>

### Phase 4 — Evaluate and Visualize

**Task:** Compute test accuracy, plot learning curves, and draw the decision boundary.

```python
# TODO: Compute test accuracy
model.eval()
with torch.no_grad():
    test_out = model(X_test)
    # Hint: torch.argmax(test_out, dim=1) gives predicted classes
    test_preds = ___
    test_acc = ___
    print(f"Test accuracy: {test_acc:.1f}%")

# TODO: Create a 1×2 figure
# Left: Learning curves with vertical line at best_epoch
# Right: Decision boundary on test data
#   - Create meshgrid, convert to tensor, forward through model
#   - Use torch.argmax on model output to get class predictions
#   - Convert back to numpy for matplotlib

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: learning curves
ax = axes[0]
___

# Right: decision boundary
ax = axes[1]
# Hint for meshgrid → tensor → predict → numpy:
#   grid_tensor = torch.tensor(np.vstack([xx.ravel(), yy.ravel()]).T, dtype=torch.float32)
#   with torch.no_grad(): Z = torch.argmax(model(grid_tensor), dim=1).numpy()
___

plt.tight_layout()
plt.show()
```

<details>
<summary>Solution</summary>

```python
model.eval()
with torch.no_grad():
    test_out = model(X_test)
    test_preds = torch.argmax(test_out, dim=1)
    test_acc = (test_preds == y_test).float().mean().item() * 100
    print(f"Test accuracy: {test_acc:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Learning curves
ax = axes[0]
ax.plot(train_losses, label='Train', linewidth=1.5)
ax.plot(val_losses, label='Val', linewidth=1.5)
ax.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best @ {best_epoch}')
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Loss', fontsize=14)
ax.set_title(f'Learning Curves (Test Acc: {test_acc:.1f}%)', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Decision boundary
ax = axes[1]
xx, yy = np.meshgrid(
    np.linspace(X_np[0].min()-0.3, X_np[0].max()+0.3, 200),
    np.linspace(X_np[1].min()-0.3, X_np[1].max()+0.3, 200))
grid_tensor = torch.tensor(np.vstack([xx.ravel(), yy.ravel()]).T, dtype=torch.float32)

model.eval()
with torch.no_grad():
    Z = torch.argmax(model(grid_tensor), dim=1).numpy().reshape(xx.shape)

ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5],
            colors=['#ADD8E6', '#FFCCCB', '#90EE90'], alpha=0.4)
test_np = X_test.numpy()
test_y_np = y_test.numpy()
for k, c in enumerate(['blue', 'red', 'green']):
    mask = test_y_np == k
    ax.scatter(test_np[mask, 0], test_np[mask, 1], c=c, edgecolors='black', s=30, alpha=0.8)
ax.set_xlabel('$x_1$', fontsize=14)
ax.set_ylabel('$x_2$', fontsize=14)
ax.set_title('Decision Boundary (Test Set)', fontsize=16)

plt.tight_layout()
plt.show()
```
</details>

### Phase 5 — Line Count Comparison

Count the lines you wrote for the PyTorch spiral classifier vs the Session 7–8 manual version:

| Component | Manual (Sessions 7–8) | PyTorch (this session) |
|---|---|---|
| Model definition | ~40 lines (class with forward + backward) | ~10 lines |
| Loss function | ~5 lines | 1 line (`nn.CrossEntropyLoss()`) |
| Optimizer | ~20 lines (Adam class) | 1 line (`optim.Adam(...)`) |
| Training loop | ~15 lines | ~10 lines |
| Dropout | ~5 lines in forward + training flag | 1 line (`nn.Dropout(p)`) |
| L2 regularization | 3 lines in backward | 0 lines (`weight_decay=`) |
| **Total** | **~88 lines** | **~23 lines** |

And the PyTorch version supports **any number of layers** with zero additional code.

---

## 8. Mini-Project C: MNIST — Your First Real Dataset {#project-c}

### 🎯 Goal

Train an MLP on handwritten digit recognition (MNIST) — 10 classes, 28×28 pixel images, 60,000 training samples. This is the standard "Hello World" of deep learning.

**Skills reused:** Multi-class classification (Session 7), full training pipeline (Project B), evaluation metrics (Session 7).

---

### Phase 1 — Load MNIST

PyTorch provides MNIST through `torchvision`. Each image is 28×28 grayscale, which we **flatten** to a vector of 784 features.

```python
from torchvision import datasets, transforms

# Download and load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),            # Convert to tensor, scale to [0, 1]
    transforms.Lambda(lambda x: x.view(-1))  # Flatten 28×28 → 784
])

train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples:     {len(test_dataset)}")
print(f"Image shape (flattened): {train_dataset[0][0].shape}")
print(f"Classes: {train_dataset.classes}")
```

### Working with a Subset

For speed (and to see overfitting effects), we'll use a **subset** of 5,000 training samples:

```python
# Use a subset for manageable training times
n_train = 5000
n_val = 1000

# Load into tensors
all_train_X = torch.stack([train_dataset[i][0] for i in range(n_train + n_val)])
all_train_y = torch.tensor([train_dataset[i][1] for i in range(n_train + n_val)])

X_train_mnist = all_train_X[:n_train]
y_train_mnist = all_train_y[:n_train]
X_val_mnist = all_train_X[n_train:]
y_val_mnist = all_train_y[n_train:]

# Full test set
X_test_mnist = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
y_test_mnist = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

print(f"Train: {X_train_mnist.shape}")   # (5000, 784)
print(f"Val:   {X_val_mnist.shape}")     # (1000, 784)
print(f"Test:  {X_test_mnist.shape}")    # (10000, 784)
```

### Visualize Some Samples

```python
fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for i, ax in enumerate(axes.flatten()):
    img = X_train_mnist[i].reshape(28, 28)
    ax.imshow(img, cmap='gray')
    ax.set_title(str(y_train_mnist[i].item()), fontsize=10)
    ax.axis('off')
plt.suptitle('MNIST Samples', fontsize=14)
plt.tight_layout()
plt.show()
```

### Phase 2 — Build the MNIST Classifier

**Task:** Define an MLP for 10-class digit recognition. Architecture:
- 784 → 256 (ReLU, Dropout 0.2) → 128 (ReLU, Dropout 0.2) → 10

This is your **first 3-layer network** — easy in PyTorch, would have been painful manually!

```python
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Define a 3-layer MLP
        # 784 → 256 → ReLU → Dropout(0.2) → 128 → ReLU → Dropout(0.2) → 10
        self.net = nn.Sequential(
            ___
        )
    
    def forward(self, x):
        return self.net(x)

model = MNISTNet()
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")
# Expected: 784*256 + 256 + 256*128 + 128 + 128*10 + 10 = 234,506
```

<details>
<summary>Solution</summary>

```python
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )
    
    def forward(self, x):
        return self.net(x)
```
</details>

### 🤔 Think About It

**Q:** Why is building a 3-layer network trivial in PyTorch but painful in our manual code?

<details>
<summary>Answer</summary>
In our manual MLP, the backward pass was hardcoded for exactly 2 layers — adding a third would require writing new delta propagation code. In PyTorch, `loss.backward()` walks the computational graph **regardless of depth**. The forward pass defines the graph, the backward pass is automatic.
</details>

### Phase 3 — Train

**Task:** Train the model with early stopping. This is the same loop as Project B — practice makes permanent!

```python
model = MNISTNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_loss = float('inf')
best_epoch = 0
best_state = None
patience = 30
n_epochs = 200

for epoch in range(n_epochs):
    # ── Train ──
    model.train()
    
    # TODO: Forward, loss, zero_grad, backward, step
    ___
    
    train_losses.append(loss.item())
    
    # TODO: Compute training accuracy (with no_grad)
    with torch.no_grad():
        train_preds = ___
        train_acc = ___
        train_accs.append(train_acc)
    
    # ── Validate ──
    model.eval()
    with torch.no_grad():
        val_out = model(X_val_mnist)
        val_loss = criterion(val_out, y_val_mnist)
        val_losses.append(val_loss.item())
        
        val_preds = torch.argmax(val_out, dim=1)
        val_acc = (val_preds == y_val_mnist).float().mean().item() * 100
        val_accs.append(val_acc)
    
    # ── Early stopping ──
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        best_epoch = epoch
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    if epoch - best_epoch >= patience:
        print(f"Early stopping at epoch {epoch} (best: {best_epoch})")
        break
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}: Train {loss.item():.4f} ({train_acc:.1f}%) | "
              f"Val {val_loss.item():.4f} ({val_acc:.1f}%)")

model.load_state_dict(best_state)
```

<details>
<summary>Solution — training step</summary>

```python
    # Train
    model.train()
    outputs = model(X_train_mnist)
    loss = criterion(outputs, y_train_mnist)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    with torch.no_grad():
        train_preds = torch.argmax(outputs, dim=1)
        train_acc = (train_preds == y_train_mnist).float().mean().item() * 100
        train_accs.append(train_acc)
```
</details>

### Phase 4 — Evaluate on Full Test Set

**Task:** Compute test accuracy and build a confusion matrix on the 10,000 test images.

```python
# Test accuracy
model.eval()
with torch.no_grad():
    test_out = model(X_test_mnist)
    test_preds = torch.argmax(test_out, dim=1)
    test_acc = (test_preds == y_test_mnist).float().mean().item() * 100
    print(f"Test accuracy on 10,000 images: {test_acc:.1f}%")

# TODO: Build and display confusion matrix (10×10)
# Reuse the confusion_matrix function from Session 7, or write it with PyTorch
cm = np.zeros((10, 10), dtype=int)
for true, pred in zip(y_test_mnist.numpy(), test_preds.numpy()):
    cm[true, pred] += 1

# TODO: Plot the confusion matrix as a heatmap
fig, ax = plt.subplots(figsize=(10, 8))
___

plt.show()

# Print per-digit accuracy
print("\nPer-digit accuracy:")
for d in range(10):
    digit_acc = cm[d, d] / cm[d].sum() * 100
    print(f"  Digit {d}: {digit_acc:.1f}% ({cm[d, d]}/{cm[d].sum()})")
```

<details>
<summary>Solution — confusion matrix plot</summary>

```python
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, cmap='Blues')
for i in range(10):
    for j in range(10):
        color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color=color, fontsize=9)

ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xlabel('Predicted', fontsize=14)
ax.set_ylabel('True', fontsize=14)
ax.set_title(f'MNIST Confusion Matrix (Test Acc: {test_acc:.1f}%)', fontsize=16)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()
```
</details>

### Phase 5 — Visualize Predictions

**Task:** Show some correctly classified and misclassified digits.

```python
# Find correct and incorrect predictions
correct_mask = (test_preds == y_test_mnist).numpy()
incorrect_mask = ~correct_mask

correct_idx = np.where(correct_mask)[0][:10]
incorrect_idx = np.where(incorrect_mask)[0][:10]

fig, axes = plt.subplots(2, 10, figsize=(16, 4))

# Row 1: Correct predictions
for i, idx in enumerate(correct_idx):
    ax = axes[0, i]
    img = X_test_mnist[idx].reshape(28, 28)
    ax.imshow(img, cmap='gray')
    ax.set_title(f'{test_preds[idx].item()}', color='green', fontsize=12, fontweight='bold')
    ax.axis('off')
axes[0, 0].set_ylabel('Correct', fontsize=12)

# Row 2: Incorrect predictions
for i, idx in enumerate(incorrect_idx):
    ax = axes[1, i]
    img = X_test_mnist[idx].reshape(28, 28)
    ax.imshow(img, cmap='gray')
    ax.set_title(f'{test_preds[idx].item()} (true: {y_test_mnist[idx].item()})',
                 color='red', fontsize=10, fontweight='bold')
    ax.axis('off')
axes[1, 0].set_ylabel('Wrong', fontsize=12)

plt.suptitle('MNIST Predictions', fontsize=14)
plt.tight_layout()
plt.show()
```

### Phase 6 — Architecture Experiment

**Task:** Compare three architectures on MNIST. Train each, record validation accuracy. Which is best?

```python
architectures = {
    "Small (784→32→10)": nn.Sequential(
        nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 10)
    ),
    "Medium (784→128→10)": nn.Sequential(
        nn.Linear(784, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 10)
    ),
    "Deep (784→256→128→10)": nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 10)
    ),
}

# TODO: For each architecture:
# 1. Create model, criterion, optimizer (Adam, lr=0.001, weight_decay=1e-4)
# 2. Train for 100 epochs (no early stopping — keep it simple here)
# 3. Record val accuracy history
# 4. Print final test accuracy

arch_results = {}

for name, net_seq in architectures.items():
    # TODO: Wrap in nn.Module or use directly, train, evaluate
    model = type('Net', (nn.Module,), {
        '__init__': lambda self, n=net_seq: (super(type(self), self).__init__(), setattr(self, 'net', n)),
        'forward': lambda self, x: self.net(x)
    })()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    val_hist = []
    for epoch in range(100):
        ___  # train + record val accuracy
    
    model.eval()
    with torch.no_grad():
        test_preds = torch.argmax(model(X_test_mnist), dim=1)
        test_acc = (test_preds == y_test_mnist).float().mean().item() * 100
    
    arch_results[name] = {"val_hist": val_hist, "test_acc": test_acc}
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name:>30s}: Test {test_acc:.1f}% | Params: {n_params:,}")
```

<details>
<summary>Solution — training loop per architecture</summary>

```python
    val_hist = []
    for epoch in range(100):
        model.train()
        out = model(X_train_mnist)
        loss = criterion(out, y_train_mnist)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_preds = torch.argmax(model(X_val_mnist), dim=1)
            val_acc = (val_preds == y_val_mnist).float().mean().item() * 100
            val_hist.append(val_acc)
```
</details>

**Task:** Plot validation accuracy curves for all three architectures on the same axes.

```python
fig, ax = plt.subplots(figsize=(10, 6))

# TODO: Plot val accuracy curves and add a legend with test accuracy
for name, res in arch_results.items():
    ax.plot(res["val_hist"], label=f'{name} (Test: {res["test_acc"]:.1f}%)', linewidth=2)

ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Validation Accuracy (%)', fontsize=14)
ax.set_title('Architecture Comparison on MNIST', fontsize=16)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.show()
```

### Phase 7 — Reflection

Answer in your notebook:

1. How many lines of code did it take to go from our 2-layer NumPy MLP to a 3-layer PyTorch model on 784-dimensional data?
2. Which MNIST digits are most often confused? (Look at the off-diagonal elements of the confusion matrix.)
3. The deep model has 234K parameters but only 5K training samples. Is it overfitting? How can you tell?
4. What accuracy would you expect from a random classifier on 10 classes? How much better is our model?

---

## Summary

### What We Learned

✅ **Tensors**: PyTorch arrays — like NumPy but with gradient tracking and GPU support  
✅ **Autograd**: `loss.backward()` computes all gradients automatically  
✅ **nn.Module**: Define networks with layers and a forward method  
✅ **nn.Sequential**: Quick network building for simple architectures  
✅ **Training loop**: forward → loss → zero_grad → backward → step  
✅ **train()/eval()**: Toggle dropout and batch norm behavior  
✅ **state_dict()**: Save and restore model weights

### Key Insights

1. **PyTorch automates the calculus, not the engineering:**
   - Gradients are free → focus on architecture and hyperparameters
   - The 5-step training loop is always the same
   - But choosing the right architecture, loss, optimizer, and regularization is still up to you

2. **The transition from NumPy to PyTorch is small:**
   - Nearly identical array syntax (`@`, `+`, `.reshape()`)
   - Main differences: `axis` → `dim`, `float64` → `float32`, batch-first convention
   - Your understanding of backprop, loss functions, and regularization carries over completely

3. **Framework advantages compound with complexity:**
   - 2-layer MLP: PyTorch saves some effort
   - 3-layer MLP: PyTorch saves a lot of effort
   - 50-layer ResNet: PyTorch makes it possible at all

### What's Next?

**Session 10: The Convolution Operation**

In the next session, we'll learn:
- **Why MLPs fail for images**: The curse of dimensionality (MNIST's 784 inputs are tiny — real images have millions of pixels!)
- **Convolutions**: Sliding filters that detect local patterns
- **Feature detectors**: How kernels find edges, textures, and shapes
- **Parameter efficiency**: A 3×3 kernel has 9 parameters regardless of image size

**The goal:** Understand the building block of Convolutional Neural Networks (CNNs)!

### Before Next Session

**Think about:**
1. Our MNIST MLP flattens the 28×28 image into a 784-dimensional vector. What spatial information is lost?
2. If we wanted to classify 224×224 RGB images, the input would have $224 \times 224 \times 3 = 150{,}528$ features. How many parameters would the first hidden layer need?
3. When you recognize a digit, do you look at every pixel equally, or do you focus on **local patterns** (curves, lines, intersections)?

**Optional reading:**
- PyTorch official tutorials: https://pytorch.org/tutorials/
- Stanford CS231n: "Convolutional Neural Networks for Visual Recognition"

---

**End of Session 9** 🎓

**You now understand:**
- ✅ How to build and train neural networks with PyTorch
- ✅ How autograd replaces manual backpropagation
- ✅ How to tackle real datasets like MNIST

**Next up:** Convolutions — learning to see! 🚀
