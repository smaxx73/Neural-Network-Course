# Session 6: Backpropagation
## Teaching Neural Networks to Learn

**Course: Neural Networks for Engineers**  
**Duration: 2 hours**

---

## Table of Contents

1. [Recap: What We Know So Far](#recap)
2. [The Chain Rule: Derivatives Through Composition](#chain-rule)
3. [Computational Graphs: Visualizing the Chain Rule](#comp-graphs)
4. [Backpropagation: The Algorithm](#backprop)
5. [Step-by-Step: Backprop on a 2-2-1 Network](#step-by-step)
6. [Implementing Backpropagation from Scratch](#implementation)
7. [Training the XOR Network (Finally!)](#xor-training)
8. [Gradient Checking: Verifying Your Gradients](#grad-check)
9. [Final Exercises](#exercises)

---

## 1. Recap: What We Know So Far {#recap}

### What We've Learned

✅ **Loss functions**: MSE measures "how wrong" our predictions are  
✅ **Gradient descent**: Update rule $w \leftarrow w - \eta \nabla L$  
✅ **Derivatives**: The slope tells us which direction to move  
✅ **Linear regression**: Trained a single neuron with GD  
✅ **The missing piece**: How to compute gradients for **hidden layers**

### 🤔 Quick Questions

**Q1:** In gradient descent, we update weights using $w \leftarrow w - \eta \frac{\partial L}{\partial w}$. Why the **minus** sign?

<details>
<summary>Click to reveal answer</summary>
The gradient points in the direction of steepest **ascent** (increasing loss). We want to **decrease** the loss, so we move in the **opposite** direction.
</details>

**Q2:** For a single neuron $\hat{y} = wx + b$, computing $\frac{\partial L}{\partial w}$ was straightforward. Why is it harder for hidden layer weights?

<details>
<summary>Click to reveal answer</summary>
Hidden layer weights don't affect the loss **directly**. Changing $W^{(1)}$ changes the hidden activations $h$, which changes the output $\hat{y}$, which changes the loss. We need to trace the effect through **multiple layers** — this requires the **chain rule**.
</details>

**Q3:** What were the three "think about" questions from last session?

<details>
<summary>Click to reveal answer</summary>

1. For $f(g(x))$, we use the chain rule: $\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$
2. If $\Delta h = +0.1$ causes $\Delta L = +0.3$, then $\frac{\partial L}{\partial h} \approx 3.0$
3. We "pass" the error backward by multiplying by the chain of derivatives — this is backpropagation!
</details>

---

## 2. The Chain Rule: Derivatives Through Composition {#chain-rule}

### The Problem

Consider a 2-layer network:

$$
\hat{y} = \sigma(w_2 \cdot \sigma(w_1 \cdot x + b_1) + b_2)
$$

This is a **composition** of functions: the output of one feeds into the next.

How do we compute $\frac{\partial L}{\partial w_1}$?

### The Chain Rule (Single Variable)

If $y = f(g(x))$, then:

$$
\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}
$$

**In words:** "The rate of change of $y$ with respect to $x$ equals the rate of change of $y$ with respect to $g$, times the rate of change of $g$ with respect to $x$."

### Building Intuition: A Temperature Example

Suppose:
- Temperature in °C depends on altitude: $C = 20 - 6h$ (drops 6°C per km)
- Temperature in °F depends on °C: $F = 1.8C + 32$

**Question:** How fast does °F change with altitude?

$$
\frac{dF}{dh} = \frac{dF}{dC} \cdot \frac{dC}{dh} = 1.8 \times (-6) = -10.8 \text{ °F/km}
$$

Each intermediate variable **multiplies** its effect through the chain.

### ✏️ Exercise 2.1: Chain Rule Practice

Compute $\frac{dy}{dx}$ for each composition:

**a)** $y = (3x + 1)^2$

Hint: let $u = 3x + 1$, so $y = u^2$

$\frac{dy}{dx} =$ ___

**b)** $y = \text{sigmoid}(2x - 1)$

Hint: recall $\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))$

$\frac{dy}{dx} =$ ___

**c)** $y = (5 - \sigma(x))^2$ (this is like a loss function!)

$\frac{dy}{dx} =$ ___

<details>
<summary>Solutions</summary>

**a)** Let $u = 3x + 1$:

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = 2u \cdot 3 = 6(3x + 1)
$$

**b)** Let $z = 2x - 1$:

$$
\frac{dy}{dx} = \frac{d\sigma}{dz} \cdot \frac{dz}{dx} = \sigma(z)(1 - \sigma(z)) \cdot 2
$$

**c)** Let $a = \sigma(x)$, then $y = (5 - a)^2$:

$$
\frac{dy}{dx} = \frac{dy}{da} \cdot \frac{da}{dx} = -2(5 - \sigma(x)) \cdot \sigma(x)(1 - \sigma(x))
$$

This is exactly the kind of calculation backpropagation does!
</details>

### The Chain Rule with Multiple Variables

In neural networks, each weight affects the loss through a **path** of intermediate variables. The chain rule extends:

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

For a longer path (deeper network):

$$
\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h} \cdot \frac{\partial h}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1}
$$

**Key insight:** Each layer contributes one **multiplication factor** in the chain.

---

## 3. Computational Graphs: Visualizing the Chain Rule {#comp-graphs}

### What is a Computational Graph?

A computational graph breaks a complex function into simple steps, showing how data flows forward and gradients flow backward.

### Example: Single Neuron

For $L = (\hat{y} - y)^2$ where $\hat{y} = \sigma(wx + b)$:

```
Forward pass (left to right):
═══════════════════════════════

   x ──┐
       ├── [×] ── z₁ ──┐
   w ──┘                ├── [+] ── z ── [σ] ── ŷ ── [-] ── e ── [²] ── L
                   b ──┘                        y ──┘
```

```
Backward pass (right to left):
═══════════════════════════════

  ∂L    ∂L   ∂L     ∂L      ∂L     ∂L     ∂L
  ── ← ── ← ── ←  ── ←   ── ←   ── ←   ── = 1
  ∂w   ∂z₁  ∂z    ∂ŷ      ∂e     ∂L     ∂L
```

### Reading the Graph

**Forward pass** (compute the output):
1. $z_1 = w \cdot x$
2. $z = z_1 + b$
3. $\hat{y} = \sigma(z)$
4. $e = \hat{y} - y$
5. $L = e^2$

**Backward pass** (compute the gradients):
1. $\frac{\partial L}{\partial L} = 1$ (starting point)
2. $\frac{\partial L}{\partial e} = 2e$
3. $\frac{\partial L}{\partial \hat{y}} = \frac{\partial L}{\partial e} \cdot 1 = 2e$
4. $\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \sigma'(z) = 2e \cdot \sigma(z)(1-\sigma(z))$
5. $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot x$
6. $\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \cdot 1$

### 🤔 Think About It

**Q:** Notice that $\frac{\partial L}{\partial z}$ is used to compute **both** $\frac{\partial L}{\partial w}$ and $\frac{\partial L}{\partial b}$. Why is this efficient?

<details>
<summary>Answer</summary>
Because $w$ and $b$ both feed into the same node $z$. Once we know how the loss changes with respect to $z$, we can quickly find how it changes with respect to anything that contributes to $z$. This **reuse of intermediate gradients** is what makes backpropagation efficient — we compute each intermediate gradient only once!
</details>

### ✏️ Exercise 3.1: Trace the Backward Pass

Given: $x = 2$, $w = 0.5$, $b = -0.5$, $y = 1$

**Forward pass** (fill in the values):

| Step | Computation | Value |
|------|------------|-------|
| 1    | $z_1 = wx$ | ___ |
| 2    | $z = z_1 + b$ | ___ |
| 3    | $\hat{y} = \sigma(z)$ | ___ |
| 4    | $e = \hat{y} - y$ | ___ |
| 5    | $L = e^2$ | ___ |

**Backward pass** (fill in the gradients):

| Step | Gradient | Value |
|------|----------|-------|
| 1    | $\frac{\partial L}{\partial e} = 2e$ | ___ |
| 2    | $\frac{\partial L}{\partial \hat{y}} = \frac{\partial L}{\partial e} \cdot 1$ | ___ |
| 3    | $\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \sigma(z)(1 - \sigma(z))$ | ___ |
| 4    | $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot x$ | ___ |
| 5    | $\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \cdot 1$ | ___ |

<details>
<summary>Solution</summary>

**Forward pass:**

| Step | Computation | Value |
|------|------------|-------|
| 1    | $z_1 = 0.5 \times 2$ | $1.0$ |
| 2    | $z = 1.0 + (-0.5)$ | $0.5$ |
| 3    | $\hat{y} = \sigma(0.5) = \frac{1}{1+e^{-0.5}}$ | $\approx 0.6225$ |
| 4    | $e = 0.6225 - 1$ | $\approx -0.3775$ |
| 5    | $L = (-0.3775)^2$ | $\approx 0.1425$ |

**Backward pass:**

| Step | Gradient | Value |
|------|----------|-------|
| 1    | $\frac{\partial L}{\partial e} = 2(-0.3775)$ | $\approx -0.7550$ |
| 2    | $\frac{\partial L}{\partial \hat{y}} = -0.7550$ | $\approx -0.7550$ |
| 3    | $\frac{\partial L}{\partial z} = -0.7550 \times 0.6225 \times (1 - 0.6225)$ | $\approx -0.1774$ |
| 4    | $\frac{\partial L}{\partial w} = -0.1774 \times 2$ | $\approx -0.3549$ |
| 5    | $\frac{\partial L}{\partial b} = -0.1774 \times 1$ | $\approx -0.1774$ |

Both gradients are **negative** → the loss decreases when we increase $w$ and $b$ → gradient descent will increase both.
</details>

---

## 4. Backpropagation: The Algorithm {#backprop}

### Overview

Backpropagation = **"backward propagation of errors"**

It is not a new concept — it's just the chain rule applied systematically through a network. But the key insight is **how** we organize the computation:

1. **Forward pass**: Compute and **store** all intermediate values
2. **Backward pass**: Compute gradients from output to input, **reusing** stored values

### The General Setup

For a network with $L$ layers:

**Forward pass** — for each layer $l = 1, 2, \ldots, L$:

$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
$$
$$
a^{(l)} = f^{(l)}(z^{(l)})
$$

Where $a^{(0)} = \mathbf{x}$ (the input) and $f^{(l)}$ is the activation function for layer $l$.

**Loss computation:**

$$
L = \text{MSE}(y, a^{(L)}) = (y - a^{(L)})^2
$$

### The Key Quantity: $\delta^{(l)}$

We define the **error signal** for layer $l$:

$$
\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}}
$$

This tells us: "How does the loss change when we change the weighted sum at layer $l$?"

Once we have $\delta^{(l)}$, the gradients for that layer are easy:

$$
\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} \cdot (a^{(l-1)})^T
$$

$$
\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}
$$

### Backward Pass — Computing $\delta$

**Output layer** (layer $L$):

$$
\delta^{(L)} = \frac{\partial L}{\partial a^{(L)}} \odot f'^{(L)}(z^{(L)})
$$

For MSE loss: $\frac{\partial L}{\partial a^{(L)}} = -2(y - a^{(L)})$

**Hidden layers** (layer $l < L$) — the magic formula:

$$
\delta^{(l)} = \left( (W^{(l+1)})^T \delta^{(l+1)} \right) \odot f'^{(l)}(z^{(l)})
$$

**In words:** The error signal at layer $l$ =  
(error signal from the next layer, **pulled back** through the weights) × (slope of the activation function)

The symbol $\odot$ means element-wise multiplication.

### 🤔 Think About It

**Q:** Why do we multiply by the **transpose** of $W^{(l+1)}$?

<details>
<summary>Answer</summary>
In the forward pass, $W^{(l+1)}$ maps activations **forward** (from layer $l$ to $l+1$). In the backward pass, we need to map errors **backward** (from layer $l+1$ to $l$). The transpose does exactly this — it "reverses" the connections, distributing each output error back to the hidden neurons that contributed to it.
</details>

### The Complete Algorithm

```
BACKPROPAGATION ALGORITHM
─────────────────────────

Input: training sample (x, y), network parameters {W, b}

FORWARD PASS:
  a⁽⁰⁾ = x
  for l = 1 to L:
      z⁽ˡ⁾ = W⁽ˡ⁾ a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
      a⁽ˡ⁾ = f(z⁽ˡ⁾)
      STORE z⁽ˡ⁾ and a⁽ˡ⁾        ← Important!

LOSS:
  L = loss(y, a⁽ᴸ⁾)

BACKWARD PASS:
  δ⁽ᴸ⁾ = ∂L/∂a⁽ᴸ⁾ ⊙ f'(z⁽ᴸ⁾)    ← output layer error
  for l = L-1 down to 1:
      δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀ δ⁽ˡ⁺¹⁾ ⊙ f'(z⁽ˡ⁾)   ← propagate backward

GRADIENTS:
  for l = 1 to L:
      ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾ (a⁽ˡ⁻¹⁾)ᵀ
      ∂L/∂b⁽ˡ⁾ = δ⁽ˡ⁾

UPDATE:
  for l = 1 to L:
      W⁽ˡ⁾ ← W⁽ˡ⁾ - η ∂L/∂W⁽ˡ⁾
      b⁽ˡ⁾ ← b⁽ˡ⁾ - η ∂L/∂b⁽ˡ⁾
```

---

## 5. Step-by-Step: Backprop on a 2-2-1 Network {#step-by-step}

### Network Setup

Let's trace backpropagation through our familiar XOR network architecture:

```
Input (2)    Hidden (2)    Output (1)
  x₁ ─────── h₁ ──────┐
       ╲  ╱             ├── ŷ
       ╱  ╲             │
  x₂ ─────── h₂ ──────┘
```

**Parameters:**
- $W^{(1)} \in \mathbb{R}^{2 \times 2}$, $b^{(1)} \in \mathbb{R}^{2}$ (input → hidden)
- $W^{(2)} \in \mathbb{R}^{1 \times 2}$, $b^{(2)} \in \mathbb{R}^{1}$ (hidden → output)
- Activation: sigmoid everywhere
- Loss: MSE

### Concrete Example

**Given:**

$$
W^{(1)} = \begin{bmatrix} 0.5 & 0.3 \\ -0.2 & 0.8 \end{bmatrix}, \quad b^{(1)} = \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix}
$$

$$
W^{(2)} = \begin{bmatrix} 0.6 & -0.4 \end{bmatrix}, \quad b^{(2)} = 0.2
$$

**Input:** $x = [1, 0]^T$, **Target:** $y = 1$

### Step 1: Forward Pass

**Hidden layer:**

$$
z^{(1)} = W^{(1)} x + b^{(1)} = \begin{bmatrix} 0.5 & 0.3 \\ -0.2 & 0.8 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix} = \begin{bmatrix} 0.5 + 0.1 \\ -0.2 - 0.1 \end{bmatrix} = \begin{bmatrix} 0.6 \\ -0.3 \end{bmatrix}
$$

$$
a^{(1)} = \sigma(z^{(1)}) = \begin{bmatrix} \sigma(0.6) \\ \sigma(-0.3) \end{bmatrix} \approx \begin{bmatrix} 0.6457 \\ 0.4256 \end{bmatrix}
$$

**Output layer:**

$$
z^{(2)} = W^{(2)} a^{(1)} + b^{(2)} = \begin{bmatrix} 0.6 & -0.4 \end{bmatrix} \begin{bmatrix} 0.6457 \\ 0.4256 \end{bmatrix} + 0.2
$$

$$
z^{(2)} = 0.6 \times 0.6457 + (-0.4) \times 0.4256 + 0.2 = 0.3874 - 0.1702 + 0.2 = 0.4172
$$

$$
\hat{y} = a^{(2)} = \sigma(0.4172) \approx 0.6028
$$

**Loss:**

$$
L = (y - \hat{y})^2 = (1 - 0.6028)^2 = (0.3972)^2 \approx 0.1578
$$

### Step 2: Backward Pass — Output Layer

We need $\delta^{(2)}$:

$$
\delta^{(2)} = \frac{\partial L}{\partial z^{(2)}} = \frac{\partial L}{\partial a^{(2)}} \cdot \sigma'(z^{(2)})
$$

**Loss gradient:**

$$
\frac{\partial L}{\partial a^{(2)}} = -2(y - \hat{y}) = -2(1 - 0.6028) = -0.7944
$$

**Sigmoid derivative:** $\sigma'(z) = \sigma(z)(1 - \sigma(z)) = 0.6028 \times 0.3972 = 0.2395$

$$
\delta^{(2)} = -0.7944 \times 0.2395 \approx -0.1903
$$

### ✏️ Exercise 5.1: Compute Output Layer Gradients

Using $\delta^{(2)} = -0.1903$ and $a^{(1)} = [0.6457, 0.4256]^T$:

$\frac{\partial L}{\partial W^{(2)}} = \delta^{(2)} \cdot (a^{(1)})^T =$ ___

$\frac{\partial L}{\partial b^{(2)}} = \delta^{(2)} =$ ___

<details>
<summary>Solution</summary>

$$
\frac{\partial L}{\partial W^{(2)}} = -0.1903 \times \begin{bmatrix} 0.6457 & 0.4256 \end{bmatrix} = \begin{bmatrix} -0.1229 & -0.0810 \end{bmatrix}
$$

$$
\frac{\partial L}{\partial b^{(2)}} = -0.1903
$$

Both gradients are **negative** → gradient descent will **increase** these values → pushing the output closer to 1 ✓
</details>

### Step 3: Backward Pass — Hidden Layer

Now the magic: propagating the error **backward** to the hidden layer.

$$
\delta^{(1)} = \left( (W^{(2)})^T \delta^{(2)} \right) \odot \sigma'(z^{(1)})
$$

**Pull error back through weights:**

$$
(W^{(2)})^T \delta^{(2)} = \begin{bmatrix} 0.6 \\ -0.4 \end{bmatrix} \times (-0.1903) = \begin{bmatrix} -0.1142 \\ 0.0761 \end{bmatrix}
$$

**Multiply by sigmoid derivative:**

$$
\sigma'(z^{(1)}) = \begin{bmatrix} \sigma(0.6)(1 - \sigma(0.6)) \\ \sigma(-0.3)(1 - \sigma(-0.3)) \end{bmatrix} = \begin{bmatrix} 0.6457 \times 0.3543 \\ 0.4256 \times 0.5744 \end{bmatrix} \approx \begin{bmatrix} 0.2288 \\ 0.2445 \end{bmatrix}
$$

$$
\delta^{(1)} = \begin{bmatrix} -0.1142 \\ 0.0761 \end{bmatrix} \odot \begin{bmatrix} 0.2288 \\ 0.2445 \end{bmatrix} = \begin{bmatrix} -0.0261 \\ 0.0186 \end{bmatrix}
$$

### ✏️ Exercise 5.2: Compute Hidden Layer Gradients

Using $\delta^{(1)} = [-0.0261, 0.0186]^T$ and $a^{(0)} = x = [1, 0]^T$:

$\frac{\partial L}{\partial W^{(1)}} = \delta^{(1)} \cdot (a^{(0)})^T =$ ___

$\frac{\partial L}{\partial b^{(1)}} = \delta^{(1)} =$ ___

<details>
<summary>Solution</summary>

$$
\frac{\partial L}{\partial W^{(1)}} = \begin{bmatrix} -0.0261 \\ 0.0186 \end{bmatrix} \begin{bmatrix} 1 & 0 \end{bmatrix} = \begin{bmatrix} -0.0261 & 0 \\ 0.0186 & 0 \end{bmatrix}
$$

$$
\frac{\partial L}{\partial b^{(1)}} = \begin{bmatrix} -0.0261 \\ 0.0186 \end{bmatrix}
$$

Note: the second column of $\frac{\partial L}{\partial W^{(1)}}$ is zero because $x_2 = 0$ — a weight has no gradient contribution from an input that is zero!
</details>

### Step 4: Update All Weights

With learning rate $\eta = 1.0$ (large for demonstration):

$$
W^{(2)}_{\text{new}} = W^{(2)} - \eta \frac{\partial L}{\partial W^{(2)}} = \begin{bmatrix} 0.6 & -0.4 \end{bmatrix} - 1.0 \times \begin{bmatrix} -0.1229 & -0.0810 \end{bmatrix}
$$

$$
= \begin{bmatrix} 0.7229 & -0.3190 \end{bmatrix}
$$

**Interpretation:** $w_{21}$ increased (stronger connection from h₁) and $w_{22}$ increased (weaker negative connection from h₂), both pushing the output closer to 1.

---

## 6. Implementing Backpropagation from Scratch {#implementation}

### Activation Functions and Their Derivatives

Before implementing backprop, we need the sigmoid derivative:

```python
import numpy as np

def sigmoid(z):
    """Sigmoid activation"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    """Derivative of sigmoid: σ(z)(1 - σ(z))"""
    s = sigmoid(z)
    return s * (1 - s)
```

### 🤔 Quick Check

**Q:** What is the maximum value of $\sigma'(z)$? At what value of $z$?

<details>
<summary>Answer</summary>
The maximum is $\sigma'(0) = 0.5 \times 0.5 = 0.25$, occurring at $z = 0$.

This means the sigmoid "passes through" at most **25%** of the gradient at each layer. For deep networks, this causes the **vanishing gradient problem** — gradients shrink exponentially as they propagate backward through many sigmoid layers. This is one reason ReLU became popular (its derivative is either 0 or 1).
</details>

### The MLP Class

**Fill in the blanks for the backward pass:**

```python
class MLP:
    """
    A 2-layer MLP (input → hidden → output) trained with backpropagation.
    """
    
    def __init__(self, n_input, n_hidden, n_output):
        """Initialize with random weights"""
        np.random.seed(42)
        # Xavier initialization (better than pure random)
        self.W1 = np.random.randn(n_hidden, n_input) * np.sqrt(2.0 / n_input)
        self.b1 = np.zeros((n_hidden, 1))
        self.W2 = np.random.randn(n_output, n_hidden) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros((n_output, 1))
    
    def forward(self, X):
        """
        Forward pass. X shape: (n_input, N) where N = number of samples.
        
        IMPORTANT: Store intermediate values for backprop!
        """
        # Hidden layer
        self.z1 = self.W1 @ X + self.b1          # (n_hidden, N)
        self.a1 = sigmoid(self.z1)                # (n_hidden, N)
        
        # Output layer
        self.z2 = self.W2 @ self.a1 + self.b2    # (n_output, N)
        self.a2 = sigmoid(self.z2)                # (n_output, N)
        
        return self.a2
    
    def compute_loss(self, y_true):
        """MSE loss"""
        N = y_true.shape[1]
        return np.mean((y_true - self.a2) ** 2)
    
    def backward(self, X, y_true, lr):
        """
        Backward pass + weight update.
        
        X shape: (n_input, N)
        y_true shape: (n_output, N)
        """
        N = X.shape[1]  # Number of samples
        
        # --- Output layer gradients ---
        
        # dL/da2 = -2(y - a2) / N
        dL_da2 = -2 * (y_true - self.a2) / N
        
        # delta2 = dL/dz2 = dL/da2 * sigmoid'(z2)
        delta2 = dL_da2 * sigmoid_derivative(self.z2)     # (n_output, N)
        
        # Gradients for W2 and b2
        dW2 = delta2 @ self.a1.T                           # (n_output, n_hidden)
        db2 = np.sum(delta2, axis=1, keepdims=True)        # (n_output, 1)
        
        # --- Hidden layer gradients ---
        
        # TODO: Propagate error backward through W2
        delta1 = (___.T @ ___) * sigmoid_derivative(___)   # Fill in!
        
        # TODO: Gradients for W1 and b1
        dW1 = ___ @ ___.T                                  # Fill in!
        db1 = np.sum(___, axis=1, keepdims=True)            # Fill in!
        
        # --- Update weights ---
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        
        return dW1, db1, dW2, db2
```

<details>
<summary>Solution for blanks</summary>

```python
# Propagate error backward through W2
delta1 = (self.W2.T @ delta2) * sigmoid_derivative(self.z1)

# Gradients for W1 and b1
dW1 = delta1 @ X.T
db1 = np.sum(delta1, axis=1, keepdims=True)
```

**Why these formulas?**

- `self.W2.T @ delta2`: pulls the output error back through the weights (transpose reverses the direction)
- `* sigmoid_derivative(self.z1)`: scales by how much each hidden neuron's activation changes
- `delta1 @ X.T`: the gradient for $W^{(1)}$ is the error signal times the input (just like in linear regression!)
</details>

### 💻 Code It: Verify the Shapes

Understanding matrix dimensions is crucial for debugging. Let's check:

```python
# Create a small network: 2 inputs, 3 hidden, 1 output
mlp = MLP(n_input=2, n_hidden=3, n_output=1)

# Dummy data: 4 samples
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]], dtype=float)  # (2, 4)
y = np.array([[0, 1, 1, 0]], dtype=float)   # (1, 4) — XOR!

# Forward pass
output = mlp.forward(X)

print("Shape check:")
print(f"  X:     {X.shape}")           # (2, 4)
print(f"  W1:    {mlp.W1.shape}")      # (3, 2)
print(f"  z1:    {mlp.z1.shape}")      # (3, 4)
print(f"  a1:    {mlp.a1.shape}")      # (3, 4)
print(f"  W2:    {mlp.W2.shape}")      # (1, 3)
print(f"  z2:    {mlp.z2.shape}")      # (1, 4)
print(f"  a2:    {mlp.a2.shape}")      # (1, 4)
print(f"  y:     {y.shape}")           # (1, 4)
print(f"  output: {output.shape}")     # (1, 4)
```

### Shape Rule of Thumb

| Quantity | Shape | Why |
|---|---|---|
| $W^{(l)}$ | (neurons in $l$, neurons in $l{-}1$) | Maps from layer $l{-}1$ to $l$ |
| $b^{(l)}$ | (neurons in $l$, 1) | One bias per neuron |
| $z^{(l)}, a^{(l)}$ | (neurons in $l$, $N$) | One column per sample |
| $\delta^{(l)}$ | (neurons in $l$, $N$) | Same shape as $z^{(l)}$ |
| $\frac{\partial L}{\partial W^{(l)}}$ | Same as $W^{(l)}$ | One gradient per weight |

---

## 7. Training the XOR Network (Finally!) {#xor-training}

### The Moment We've Been Waiting For

Since Session 4, we've been trying to solve XOR. We built the network, did forward propagation, and tried manual weight tuning. Now we can **train it automatically**!

### 💻 Code It: Train on XOR

```python
import matplotlib.pyplot as plt

# XOR dataset
X_xor = np.array([[0, 0, 1, 1],
                   [0, 1, 0, 1]], dtype=float)
y_xor = np.array([[0, 1, 1, 0]], dtype=float)

# Create network: 2 inputs, 4 hidden neurons, 1 output
mlp = MLP(n_input=2, n_hidden=4, n_output=1)

# Training loop
n_epochs = 10000
lr = 2.0  # Sigmoid networks often need larger learning rates
loss_history = []

for epoch in range(n_epochs):
    # Forward pass
    output = mlp.forward(X_xor)
    
    # Compute loss
    loss = mlp.compute_loss(y_xor)
    loss_history.append(loss)
    
    # Backward pass + update
    mlp.backward(X_xor, y_xor, lr)
    
    # Print progress
    if epoch % 2000 == 0:
        predictions = (output > 0.5).astype(int)
        accuracy = np.mean(predictions == y_xor) * 100
        print(f"Epoch {epoch:5d}: Loss = {loss:.6f}, Accuracy = {accuracy:.0f}%")

# Final results
print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)
output = mlp.forward(X_xor)
for i in range(4):
    x1, x2 = X_xor[0, i], X_xor[1, i]
    pred = output[0, i]
    true = y_xor[0, i]
    status = "✓" if (pred > 0.5) == true else "✗"
    print(f"{status} Input: ({x1:.0f}, {x2:.0f}) → Output: {pred:.4f} → "
          f"Predicted: {int(pred > 0.5)} (True: {int(true)})")
```

### 💻 Code It: Visualize Training

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Loss curve
ax = axes[0]
ax.plot(loss_history, 'b-', linewidth=1)
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('MSE Loss', fontsize=14)
ax.set_title('XOR Training: Loss Over Time', fontsize=16)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# Plot 2: Decision boundary
ax = axes[1]
x_range = np.linspace(-0.5, 1.5, 200)
y_range = np.linspace(-0.5, 1.5, 200)
X_grid, Y_grid = np.meshgrid(x_range, y_range)
grid_input = np.vstack([X_grid.ravel(), Y_grid.ravel()])  # (2, 40000)

Z_grid = mlp.forward(grid_input).reshape(X_grid.shape)

ax.contourf(X_grid, Y_grid, Z_grid, levels=[0, 0.5, 1], 
            colors=['#ADD8E6', '#FFCCCB'], alpha=0.5)
ax.contour(X_grid, Y_grid, Z_grid, levels=[0.5], colors='black', linewidths=2)

# Plot XOR points
ax.scatter([0, 1], [0, 1], s=300, c='blue', marker='o', 
           edgecolors='black', linewidth=3, label='Class 0', zorder=5)
ax.scatter([0, 1], [1, 0], s=300, c='red', marker='s', 
           edgecolors='black', linewidth=3, label='Class 1', zorder=5)

ax.set_xlabel('$x_1$', fontsize=14)
ax.set_ylabel('$x_2$', fontsize=14)
ax.set_title('Learned Decision Boundary', fontsize=16)
ax.legend(fontsize=12)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 🎉 What Just Happened?

The network **learned** to solve XOR by itself! Compare:

| Approach | Session | Method | Result |
|---|---|---|---|
| Manual weights | Session 4 | Trial and error | Found a solution (painfully) |
| Auto-trained | Session 6 | Backprop + GD | Found a solution automatically! |

The network discovered its own way to decompose XOR into sub-problems — and it might have found a **different** solution than our manual one!

### 💻 Code It: What Did the Hidden Neurons Learn?

```python
def visualize_hidden_neurons(mlp):
    """Visualize what each hidden neuron responds to"""
    x_range = np.linspace(-0.5, 1.5, 200)
    y_range = np.linspace(-0.5, 1.5, 200)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    grid_input = np.vstack([X_grid.ravel(), Y_grid.ravel()])
    
    # Get hidden activations
    z1 = mlp.W1 @ grid_input + mlp.b1
    a1 = sigmoid(z1)
    
    n_hidden = a1.shape[0]
    fig, axes = plt.subplots(1, n_hidden, figsize=(5 * n_hidden, 4))
    if n_hidden == 1:
        axes = [axes]
    
    for idx in range(n_hidden):
        ax = axes[idx]
        Z = a1[idx].reshape(X_grid.shape)
        
        contour = ax.contourf(X_grid, Y_grid, Z, levels=20, cmap='RdYlBu_r')
        plt.colorbar(contour, ax=ax)
        
        ax.scatter([0, 1], [0, 1], s=150, c='blue', marker='o', 
                   edgecolors='black', linewidth=2)
        ax.scatter([0, 1], [1, 0], s=150, c='red', marker='s', 
                   edgecolors='black', linewidth=2)
        
        ax.set_xlabel('$x_1$', fontsize=11)
        ax.set_ylabel('$x_2$', fontsize=11)
        ax.set_title(f'Hidden Neuron h{idx+1}', fontsize=13)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
    
    plt.suptitle('What Each Hidden Neuron Learned', fontsize=16)
    plt.tight_layout()
    plt.show()

visualize_hidden_neurons(mlp)
```

---

## 8. Gradient Checking: Verifying Your Gradients {#grad-check}

### Why Gradient Checking?

Backpropagation involves a lot of matrix operations. It's easy to make mistakes (wrong transpose, missing factor of 2, etc.). **Gradient checking** compares your analytical gradients to numerical approximations.

### The Method

For each weight $w_{ij}$:

$$
\frac{\partial L}{\partial w_{ij}} \approx \frac{L(w_{ij} + \epsilon) - L(w_{ij} - \epsilon)}{2\epsilon}
$$

If the analytical and numerical gradients agree (relative difference < $10^{-5}$), your implementation is likely correct!

### 💻 Code It: Gradient Checker

```python
def gradient_check(mlp, X, y, epsilon=1e-7):
    """
    Verify backprop gradients against numerical gradients.
    
    Returns the maximum relative error.
    """
    # Get analytical gradients from backprop
    mlp.forward(X)
    dW1, db1, dW2, db2 = mlp.backward(X, y, lr=0.0)  # lr=0 so weights don't change
    
    max_error = 0
    
    # Check each parameter set
    for name, param, grad in [('W1', mlp.W1, dW1), ('b1', mlp.b1, db1),
                               ('W2', mlp.W2, dW2), ('b2', mlp.b2, db2)]:
        print(f"\nChecking {name} (shape {param.shape}):")
        
        # Numerical gradient for each element
        num_grad = np.zeros_like(param)
        
        for idx in np.ndindex(param.shape):
            # Save original
            original = param[idx]
            
            # L(w + ε)
            param[idx] = original + epsilon
            mlp.forward(X)
            loss_plus = mlp.compute_loss(y)
            
            # L(w - ε)
            param[idx] = original - epsilon
            mlp.forward(X)
            loss_minus = mlp.compute_loss(y)
            
            # Numerical gradient
            num_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            
            # Restore
            param[idx] = original
        
        # Compare
        diff = np.abs(grad - num_grad)
        denom = np.maximum(np.abs(grad) + np.abs(num_grad), 1e-8)
        relative_error = np.max(diff / denom)
        max_error = max(max_error, relative_error)
        
        print(f"  Max absolute diff:  {np.max(diff):.2e}")
        print(f"  Max relative error: {relative_error:.2e}")
        
        if relative_error < 1e-5:
            print(f"  ✓ PASSED")
        else:
            print(f"  ✗ FAILED — check your backprop implementation!")
    
    return max_error

# Run gradient check
mlp_check = MLP(n_input=2, n_hidden=3, n_output=1)
X_check = np.array([[1.0], [0.5]])
y_check = np.array([[1.0]])

max_err = gradient_check(mlp_check, X_check, y_check)
print(f"\n{'='*50}")
print(f"Overall max relative error: {max_err:.2e}")
if max_err < 1e-5:
    print("✓ All gradients verified!")
else:
    print("✗ Some gradients may be incorrect!")
```

### Common Backprop Bugs That Gradient Checking Catches

| Bug | Symptom |
|---|---|
| Missing factor of 2 in MSE gradient | Gradients are off by factor of 2 |
| Wrong transpose on $W^T$ | Gradient shapes mismatch |
| Forgot to divide by $N$ (batch size) | Gradients scale with batch size |
| Used $a^{(l)}$ instead of $z^{(l)}$ in $\sigma'$ | Gradients are wrong for non-zero inputs |
| Didn't store $z$ during forward pass | Using stale values from previous forward pass |

### ✏️ Exercise 8.1: Spot the Bug

This backprop implementation has **one bug**. Can you find it?

```python
def buggy_backward(self, X, y_true, lr):
    N = X.shape[1]
    
    dL_da2 = -2 * (y_true - self.a2) / N
    delta2 = dL_da2 * sigmoid_derivative(self.a2)   # BUG HERE?
    
    dW2 = delta2 @ self.a1.T
    db2 = np.sum(delta2, axis=1, keepdims=True)
    
    delta1 = (self.W2.T @ delta2) * sigmoid_derivative(self.z1)
    
    dW1 = delta1 @ X.T
    db1 = np.sum(delta1, axis=1, keepdims=True)
    
    self.W2 -= lr * dW2
    self.b2 -= lr * db2
    self.W1 -= lr * dW1
    self.b1 -= lr * db1
```

<details>
<summary>The bug</summary>

**Line 4:** `sigmoid_derivative(self.a2)` should be `sigmoid_derivative(self.z2)`.

The sigmoid derivative is computed at the **weighted sum** $z$, not the **activation** $a$.

This is a very common bug! Remember: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, which is different from $\sigma'(\sigma(z))$.

If you pass $a = \sigma(z)$ to `sigmoid_derivative`, you'd compute $a(1-a)$ which equals $\sigma(z)(1 - \sigma(z))$ — wait, that actually gives the **same result** in this case! This is a special property of sigmoid: $\sigma'(z) = \sigma(z)(1 - \sigma(z)) = a(1 - a)$.

So for sigmoid specifically, both work. But for other activations (like ReLU), using $a$ instead of $z$ would give **wrong results**. Always use $z$ — it's the correct and general approach.
</details>

---

## 9. Final Exercises {#exercises}

### 📝 Exercise 9.1: Backprop by Hand (Easy)

Given a tiny network with **1 input, 1 hidden neuron, 1 output** (all sigmoid):

- $w_1 = 0.5$, $b_1 = 0$, $w_2 = -1.0$, $b_2 = 0.5$
- Input: $x = 1.0$, Target: $y = 0$

**Compute:**
1. Forward pass: $z_1$, $a_1$, $z_2$, $\hat{y}$, $L$
2. $\delta^{(2)}$ (output error signal)
3. $\frac{\partial L}{\partial w_2}$ and $\frac{\partial L}{\partial b_2}$
4. $\delta^{(1)}$ (hidden error signal)
5. $\frac{\partial L}{\partial w_1}$ and $\frac{\partial L}{\partial b_1}$
6. Updated weights after one step with $\eta = 1.0$

<details>
<summary>Solution</summary>

**1. Forward pass:**
- $z_1 = w_1 \cdot x + b_1 = 0.5 \times 1.0 + 0 = 0.5$
- $a_1 = \sigma(0.5) \approx 0.6225$
- $z_2 = w_2 \cdot a_1 + b_2 = -1.0 \times 0.6225 + 0.5 = -0.1225$
- $\hat{y} = \sigma(-0.1225) \approx 0.4694$
- $L = (0 - 0.4694)^2 = 0.2203$

**2. Output error signal:**
- $\frac{\partial L}{\partial \hat{y}} = -2(y - \hat{y}) = -2(0 - 0.4694) = 0.9388$
- $\sigma'(z_2) = 0.4694 \times 0.5306 = 0.2490$
- $\delta^{(2)} = 0.9388 \times 0.2490 = 0.2338$

**3. Output layer gradients:**
- $\frac{\partial L}{\partial w_2} = \delta^{(2)} \cdot a_1 = 0.2338 \times 0.6225 = 0.1455$
- $\frac{\partial L}{\partial b_2} = \delta^{(2)} = 0.2338$

**4. Hidden error signal:**
- $(w_2)^T \cdot \delta^{(2)} = -1.0 \times 0.2338 = -0.2338$
- $\sigma'(z_1) = 0.6225 \times 0.3775 = 0.2350$
- $\delta^{(1)} = -0.2338 \times 0.2350 = -0.0549$

**5. Hidden layer gradients:**
- $\frac{\partial L}{\partial w_1} = \delta^{(1)} \cdot x = -0.0549 \times 1.0 = -0.0549$
- $\frac{\partial L}{\partial b_1} = \delta^{(1)} = -0.0549$

**6. Updated weights** ($\eta = 1.0$):
- $w_2 = -1.0 - 1.0 \times 0.1455 = -1.1455$ (more negative → pushes output lower ✓)
- $b_2 = 0.5 - 1.0 \times 0.2338 = 0.2662$ (lower bias → pushes output lower ✓)
- $w_1 = 0.5 - 1.0 \times (-0.0549) = 0.5549$ (increases slightly)
- $b_1 = 0.0 - 1.0 \times (-0.0549) = 0.0549$ (increases slightly)

All updates push the output toward 0 (the target). ✓
</details>

---

### 📝 Exercise 9.2: The Full Training Loop (Medium)

Write a complete training function that trains an MLP and returns the loss history:

```python
def train_mlp(mlp, X, y, lr, n_epochs, print_every=1000):
    """
    Complete training loop for an MLP.
    
    Parameters:
    -----------
    mlp : MLP
        Network to train
    X : array, shape (n_input, N)
        Training inputs
    y : array, shape (n_output, N)
        Training targets
    lr : float
        Learning rate
    n_epochs : int
        Number of epochs
    
    Returns:
    --------
    loss_history : list
        Loss at each epoch
    """
    loss_history = []
    
    for epoch in range(n_epochs):
        # TODO: Forward pass
        ___
        
        # TODO: Compute and record loss
        ___
        
        # TODO: Backward pass + update
        ___
        
        if epoch % print_every == 0:
            pred = (mlp.a2 > 0.5).astype(int)
            acc = np.mean(pred == y) * 100
            print(f"Epoch {epoch:5d}: Loss = {loss:.6f}, Acc = {acc:.0f}%")
    
    return loss_history

# Test it
mlp = MLP(n_input=2, n_hidden=4, n_output=1)
losses = train_mlp(mlp, X_xor, y_xor, lr=2.0, n_epochs=10000)
```

<details>
<summary>Solution</summary>

```python
def train_mlp(mlp, X, y, lr, n_epochs, print_every=1000):
    loss_history = []
    
    for epoch in range(n_epochs):
        # Forward pass
        output = mlp.forward(X)
        
        # Compute and record loss
        loss = mlp.compute_loss(y)
        loss_history.append(loss)
        
        # Backward pass + update
        mlp.backward(X, y, lr)
        
        if epoch % print_every == 0:
            pred = (mlp.a2 > 0.5).astype(int)
            acc = np.mean(pred == y) * 100
            print(f"Epoch {epoch:5d}: Loss = {loss:.6f}, Acc = {acc:.0f}%")
    
    return loss_history
```
</details>

---

### 📝 Exercise 9.3: Experimenting with Architecture (Medium)

Train MLPs with different numbers of hidden neurons on XOR. Compare convergence:

```python
def architecture_experiment():
    """
    Compare hidden layer sizes: [1, 2, 4, 8] neurons
    
    TODO:
    1. For each size, create an MLP and train for 10000 epochs
    2. Plot loss curves on the same graph
    3. Answer: What's the minimum number of hidden neurons needed for XOR?
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for n_hidden in [1, 2, 4, 8]:
        np.random.seed(42)
        mlp = MLP(n_input=2, n_hidden=n_hidden, n_output=1)
        losses = train_mlp(mlp, X_xor, y_xor, lr=2.0, n_epochs=10000, print_every=20000)
        ax.plot(losses, label=f'{n_hidden} hidden neurons', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('MSE Loss', fontsize=14)
    ax.set_title('Effect of Hidden Layer Size on XOR', fontsize=16)
    ax.legend(fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.show()

# architecture_experiment()
```

**Questions:**
1. Can 1 hidden neuron solve XOR?
2. What's the minimum number that works reliably?
3. Do more neurons always help?

<details>
<summary>Discussion</summary>

1. **No!** 1 hidden neuron is equivalent to a single perceptron with a sigmoid — it can only create a linear boundary. XOR requires at least 2 hidden neurons.

2. **2 hidden neurons** is the theoretical minimum. In practice, 2 neurons *can* work but training may fail to converge depending on initialization. 4 neurons is more reliable.

3. **More neurons help convergence** (more paths to a solution), but at the cost of more parameters. For a simple problem like XOR, 4 neurons is plenty — 8 is overkill but converges faster.
</details>

---

### 📝 Exercise 9.4: Beyond XOR — Circle Dataset (Hard)

Train an MLP to classify points inside vs outside a circle:

```python
def generate_circle_data(n_samples=200, noise=0.1):
    """
    Generate 2D data: class 1 if inside circle of radius 0.5, class 0 otherwise.
    """
    np.random.seed(42)
    X = np.random.randn(2, n_samples) * 0.7
    y = ((X[0] ** 2 + X[1] ** 2) < 0.5).astype(float).reshape(1, -1)
    X += np.random.randn(2, n_samples) * noise
    return X, y

# TODO:
# 1. Generate the circle dataset
# 2. Create an MLP with an appropriate architecture
# 3. Train it (experiment with learning rate and hidden size)
# 4. Visualize the decision boundary
# 5. How many hidden neurons do you need?

X_circle, y_circle = generate_circle_data()

# Visualize the data
plt.figure(figsize=(8, 8))
plt.scatter(X_circle[0, y_circle[0] == 0], X_circle[1, y_circle[0] == 0], 
            c='blue', alpha=0.5, label='Outside')
plt.scatter(X_circle[0, y_circle[0] == 1], X_circle[1, y_circle[0] == 1], 
            c='red', alpha=0.5, label='Inside')
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Circle Classification Dataset', fontsize=16)
plt.legend(fontsize=12)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.show()
```

<details>
<summary>Hints</summary>

- A circle boundary is more complex than XOR — you'll need more hidden neurons (try 8-16)
- Learning rate around 1.0-2.0 works for sigmoid networks
- Train for 5000-20000 epochs
- The decision boundary should approximate a circle!
</details>

<details>
<summary>Solution</summary>

```python
# Create and train
np.random.seed(42)
mlp_circle = MLP(n_input=2, n_hidden=10, n_output=1)
losses = train_mlp(mlp_circle, X_circle, y_circle, lr=1.5, n_epochs=15000, print_every=3000)

# Visualize decision boundary
x_range = np.linspace(-2, 2, 200)
y_range = np.linspace(-2, 2, 200)
X_grid, Y_grid = np.meshgrid(x_range, y_range)
grid_input = np.vstack([X_grid.ravel(), Y_grid.ravel()])
Z_grid = mlp_circle.forward(grid_input).reshape(X_grid.shape)

plt.figure(figsize=(8, 8))
plt.contourf(X_grid, Y_grid, Z_grid, levels=[0, 0.5, 1], 
            colors=['#ADD8E6', '#FFCCCB'], alpha=0.4)
plt.contour(X_grid, Y_grid, Z_grid, levels=[0.5], colors='black', linewidths=2)
plt.scatter(X_circle[0, y_circle[0] == 0], X_circle[1, y_circle[0] == 0], 
            c='blue', alpha=0.5, label='Outside')
plt.scatter(X_circle[0, y_circle[0] == 1], X_circle[1, y_circle[0] == 1], 
            c='red', alpha=0.5, label='Inside')
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Learned Decision Boundary (Circle)', fontsize=16)
plt.legend(fontsize=12)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.show()
```

The network approximates a circular boundary using a combination of linear boundaries from the hidden neurons!
</details>

---

### 📝 Exercise 9.5: Gradient Checking Your Implementation (Hard)

Run gradient checking on your MLP implementation with different network sizes:

```python
def full_gradient_check():
    """
    Test gradient correctness for:
    a) 2-2-1 network
    b) 3-4-1 network
    c) 2-3-2 network (2 outputs!)
    
    For each:
    1. Create the network
    2. Create a small random dataset (1-2 samples)
    3. Run gradient_check()
    4. Verify all pass with relative error < 1e-5
    """
    configs = [
        (2, 2, 1, "2-2-1"),
        (3, 4, 1, "3-4-1"),
        (2, 3, 2, "2-3-2"),
    ]
    
    for n_in, n_hid, n_out, name in configs:
        print(f"\n{'='*50}")
        print(f"Testing {name} network")
        print(f"{'='*50}")
        
        np.random.seed(42)
        mlp = MLP(n_in, n_hid, n_out)
        X = np.random.randn(n_in, 2)   # 2 samples
        y = np.random.rand(n_out, 2)   # random targets
        
        max_err = gradient_check(mlp, X, y)
        print(f"\nResult for {name}: max error = {max_err:.2e} → "
              f"{'✓ PASS' if max_err < 1e-5 else '✗ FAIL'}")

# full_gradient_check()
```

---

## Summary

### What We Learned

✅ **Chain Rule**: Compute derivatives through function compositions  
✅ **Computational Graphs**: Visualize forward and backward data flow  
✅ **Backpropagation Algorithm**: Forward pass → store values → backward pass → update  
✅ **Error Signal $\delta^{(l)}$**: Propagated backward through transposed weights  
✅ **XOR Training**: Network learned to solve XOR automatically!  
✅ **Gradient Checking**: Numerical verification of analytical gradients

### Key Insights

1. **Backprop is just the chain rule**, applied systematically:
   - Forward pass computes and stores intermediate values
   - Backward pass multiplies local gradients along each path
   - Each layer reuses the error signal from the layer above

2. **The error signal flows backward:**
   - Output layer: error comes directly from the loss
   - Hidden layers: error is "distributed" through the transposed weight matrix
   - Activation derivative acts as a "gate" at each layer

3. **Always verify with gradient checking:**
   - Numerical gradients are slow but simple and correct
   - Compare before trusting a new backprop implementation
   - Small relative error ($< 10^{-5}$) means your code is likely correct

### What's Next?

**Session 7: Logistic Regression & Softmax**

In the next session, we'll learn:
- **Sigmoid for probabilities**: Interpreting outputs as class probabilities
- **Cross-entropy loss**: A better loss function for classification
- **Softmax**: Extending to multiple classes
- **Complete classification pipeline**: From data to evaluated predictions

**The goal:** Build proper classifiers that output probabilities, not just 0/1!

### Before Next Session

**Think about:**
1. Our MLP uses MSE loss, but for classification we want to predict probabilities. What's wrong with MSE for probabilities?
2. If the network outputs 0.99 for a class-0 sample, how should the loss penalize this? Should it be proportional to $(0 - 0.99)^2 = 0.98$ or something steeper?
3. What if we have 5 possible classes instead of 2? How should the output layer look?

**Optional reading:**
- 3Blue1Brown: "Backpropagation calculus" (YouTube)
- Chapter 6.5 of Goodfellow et al., "Deep Learning"

---

**End of Session 6** 🎓

**You now understand:**
- ✅ How the chain rule enables gradient computation through layers
- ✅ How backpropagation trains multi-layer networks
- ✅ How to verify your implementation with gradient checking

**Next up:** Classification with cross-entropy and softmax! 🚀
