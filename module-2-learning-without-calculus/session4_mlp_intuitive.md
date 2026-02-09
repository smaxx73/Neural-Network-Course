# Session 4: Multi-Layer Networks (Intuitive)
## Building Neural Networks Without Calculus

**Course: Neural Networks for Engineers**  
**Duration: 2 hours**

---

## Table of Contents

1. [Recap: What We Know So Far](#recap)
2. [The XOR Problem: A Perceptron's Nemesis](#xor)
3. [The Solution: Hidden Layers](#solution)
4. [Multi-Layer Perceptron Architecture](#mlp)
5. [Forward Propagation: Computing Outputs](#forward)
6. [Understanding What Hidden Neurons Learn](#understanding)
7. [Manual Weight Tuning Challenge](#manual)
8. [Why We Need Something Better](#motivation)
9. [Final Exercises](#exercises)

---

## 1. Recap: What We Know So Far {#recap}

### What We've Learned

✅ **Perceptron basics**: Weighted sum + activation  
✅ **Learning rule**: `w ← w + η(y - ŷ)x`  
✅ **Convergence**: Works perfectly on linearly separable data  
✅ **Limitation**: Fails on XOR and other non-linear problems

### 🤔 Quick Question

Before we continue, let's check your understanding:

**Q1:** A single perceptron creates a __________ decision boundary in 2D space.

<details>
<summary>Click to reveal answer</summary>
A **line** (or more generally, a hyperplane in higher dimensions)
</details>

**Q2:** The perceptron learning rule will converge if and only if the data is __________ separable.

<details>
<summary>Click to reveal answer</summary>
**Linearly** separable
</details>

---

## 2. The XOR Problem: A Perceptron's Nemesis {#xor}

### What is XOR?

XOR (eXclusive OR) returns true when inputs are **different**.

**Truth Table:**
| $x_1$ | $x_2$ | XOR |
|-------|-------|-----|
| 0     | 0     | 0   |
| 0     | 1     | 1   |
| 1     | 0     | 1   |
| 1     | 1     | 0   |

### Why Can't a Perceptron Solve This?

```python
import numpy as np
import matplotlib.pyplot as plt

# XOR data
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Visualize
plt.figure(figsize=(8, 8))
plt.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1], 
            s=300, c='blue', marker='o', 
            edgecolors='black', linewidth=3, label='Class 0')
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], 
            s=300, c='red', marker='s', 
            edgecolors='black', linewidth=3, label='Class 1')

# Try to draw a line
# Can you separate red squares from blue circles with ONE straight line?
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('XOR Problem: Can You Draw ONE Line to Separate Them?', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.show()
```

### ✏️ Exercise 2.1: Try to Separate XOR

**Task:** Draw (on paper or in your mind) a single straight line that separates:
- Blue circles (0,0) and (1,1) on one side
- Red squares (0,1) and (1,0) on the other side

**Can you do it?** _______

<details>
<summary>Click for answer</summary>
**NO!** It's impossible with a single straight line. The data points are arranged diagonally - you'd need a more complex boundary (like an X or two lines).
</details>

### 🤔 Think About It

**Question:** If we can't use ONE line, what if we used TWO lines?

Think about this before moving forward...

### The Geometric Insight

XOR needs a **non-linear** decision boundary. Look at this visualization:

```python
# Show what boundary we actually need
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Failed linear attempt
ax = axes[0]
ax.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1], 
           s=300, c='blue', marker='o', edgecolors='black', linewidth=3)
ax.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], 
           s=300, c='red', marker='s', edgecolors='black', linewidth=3)

# Try a line (will fail)
x_line = np.linspace(-0.5, 1.5, 100)
ax.plot([0.5, 0.5], [-0.5, 1.5], 'g--', linewidth=2, label='Line 1 (separates left/right)')
ax.plot([-0.5, 1.5], [0.5, 0.5], 'purple', linestyle='--', linewidth=2, label='Line 2 (separates top/bottom)')

ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_xlabel('$x_1$', fontsize=12)
ax.set_ylabel('$x_2$', fontsize=12)
ax.set_title('Need TWO Lines!', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: What we actually need
ax = axes[1]
ax.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1], 
           s=300, c='blue', marker='o', edgecolors='black', linewidth=3)
ax.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], 
           s=300, c='red', marker='s', edgecolors='black', linewidth=3)

# Show the regions
from matplotlib.patches import Rectangle
# Blue region (corners)
ax.add_patch(Rectangle((-0.5, -0.5), 0.8, 0.8, alpha=0.2, color='blue'))
ax.add_patch(Rectangle((0.7, 0.7), 0.8, 0.8, alpha=0.2, color='blue'))
# Red region (sides)
ax.add_patch(Rectangle((-0.5, 0.7), 0.8, 0.8, alpha=0.2, color='red'))
ax.add_patch(Rectangle((0.7, -0.5), 0.8, 0.8, alpha=0.2, color='red'))

ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_xlabel('$x_1$', fontsize=12)
ax.set_ylabel('$x_2$', fontsize=12)
ax.set_title('Non-Linear Boundary Needed', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Key Insight:** We need to **combine** multiple linear boundaries to create non-linear boundaries!

---

## 3. The Solution: Hidden Layers {#solution}

### The Big Idea

What if we:
1. Use **multiple perceptrons** in parallel (hidden layer)
2. Each creates its own line
3. **Combine** their outputs with another perceptron (output layer)

This is a **Multi-Layer Perceptron (MLP)**!

### Architecture for XOR

```
Input Layer    Hidden Layer    Output Layer
   (2)            (2)              (1)

    x₁ ──┐      ┌── h₁ ──┐
         ├──────┤         ├───── ŷ
    x₂ ──┘      └── h₂ ──┘
```

### Breaking Down the Solution

The hidden layer neurons can learn:
- **h₁**: "Is the input in the top-left or bottom-right?" → detects one diagonal
- **h₂**: "Is the input in the top-right or bottom-left?" → detects other diagonal
- **Output**: Combines h₁ and h₂ to make final decision

### 💻 Code It: Visualize the Architecture

**Fill in the blanks:**

```python
def visualize_mlp_architecture():
    """Visualize a 2-2-1 MLP for XOR"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Layer positions
    input_x = 1
    hidden_x = 3
    output_x = 5
    
    # Neuron positions (y-coordinates)
    input_positions = [2, 1]  # x1 and x2
    hidden_positions = [2, 1]  # h1 and h2
    output_position = 1.5      # y_hat
    
    # Draw neurons
    for i, y in enumerate(input_positions):
        circle = plt.Circle((input_x, y), 0.2, color='lightblue', ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(input_x, y, f'$x_{i+1}$', ha='center', va='center', fontsize=14, fontweight='bold')
    
    for i, y in enumerate(hidden_positions):
        circle = plt.Circle((hidden_x, y), 0.2, color='lightgreen', ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(hidden_x, y, f'$h_{i+1}$', ha='center', va='center', fontsize=14, fontweight='bold')
    
    circle = plt.Circle((output_x, output_position), 0.2, color='lightcoral', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(output_x, output_position, r'$\hat{y}$', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Draw connections (complete this!)
    # TODO: Draw lines from input to hidden layer
    for i_y in input_positions:
        for h_y in hidden_positions:
            ax.plot([input_x + 0.2, hidden_x - 0.2], [i_y, h_y], 
                   'gray', linewidth=1.5, alpha=0.6)
    
    # TODO: Draw lines from hidden to output layer
    for h_y in hidden_positions:
        ax.plot([___ + 0.2, ___ - 0.2], [h_y, output_position],  # Fill in the blanks!
               'gray', linewidth=1.5, alpha=0.6)
    
    # Labels
    ax.text(input_x, 3, 'Input Layer', ha='center', fontsize=12, fontweight='bold')
    ax.text(hidden_x, 3, 'Hidden Layer', ha='center', fontsize=12, fontweight='bold')
    ax.text(output_x, 3, 'Output Layer', ha='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 3.5)
    ax.axis('off')
    plt.title('Multi-Layer Perceptron: 2-2-1 Architecture', fontsize=16)
    plt.show()

# visualize_mlp_architecture()
```

<details>
<summary>Solution for blanks</summary>

```python
ax.plot([hidden_x + 0.2, output_x - 0.2], [h_y, output_position],
       'gray', linewidth=1.5, alpha=0.6)
```
</details>

---

## 4. Multi-Layer Perceptron Architecture {#mlp}

### General MLP Structure

An MLP consists of:
1. **Input layer**: Receives features (not "neurons", just input values)
2. **Hidden layer(s)**: One or more layers of neurons
3. **Output layer**: Produces final prediction

### Notation

For a network with:
- **L** layers (numbered 0 to L, where 0 is input)
- Layer **l** has **n^(l)** neurons

**Weights:**
- $W^{(l)}$: weight matrix connecting layer $l-1$ to layer $l$
- $W^{(l)}_{ij}$: weight from neuron $j$ in layer $l-1$ to neuron $i$ in layer $l$

**Activations:**
- $a^{(l)}_i$: activation (output) of neuron $i$ in layer $l$
- $a^{(0)} = \mathbf{x}$: input layer activations are just the input features

### ✏️ Exercise 4.1: Architecture Counting

Given this architecture: **4-8-8-3**

**Answer these questions:**

1. How many input features? _______
2. How many hidden layers? _______
3. How many neurons in the first hidden layer? _______
4. How many output classes? _______
5. How many total neurons (excluding input)? _______

<details>
<summary>Answers</summary>

1. **4** input features
2. **2** hidden layers (8 neurons each)
3. **8** neurons in first hidden layer
4. **3** output classes
5. **8 + 8 + 3 = 19** total neurons
</details>

### 🤔 Think About It

**Q:** How many weights are in the connection between:
- Input (4 neurons) → First hidden layer (8 neurons)?

Remember: Each hidden neuron connects to ALL input neurons!

<details>
<summary>Answer</summary>
$4 \times 8 = 32$ weights (plus 8 biases for the hidden neurons)
</details>

---

## 5. Forward Propagation: Computing Outputs {#forward}

### The Process

**Forward propagation** = computing the output layer by layer.

For each layer $l$:

1. **Compute weighted sum** for each neuron:
   $$
   z^{(l)}_i = \sum_{j} W^{(l)}_{ij} a^{(l-1)}_j + b^{(l)}_i
   $$

2. **Apply activation function**:
   $$
   a^{(l)}_i = f(z^{(l)}_i)
   $$

3. **Repeat** for next layer

### Manual Example: XOR with 2-2-1 Network

Let's solve XOR step by step with **given weights**!

#### Network Architecture
- Input: $x_1, x_2$
- Hidden layer: $h_1, h_2$ (using sigmoid activation)
- Output: $\hat{y}$ (using sigmoid activation)

#### Given Weights

**Layer 1 (Input → Hidden):**
```
W1 = [[20,  20],    # weights to h1
      [20,  20]]    # weights to h2

b1 = [-10, -30]     # biases for h1, h2
```

**Layer 2 (Hidden → Output):**
```
W2 = [[20],         # weight from h1
      [-20]]        # weight from h2

b2 = [-10]          # bias for output
```

### Step-by-Step Calculation: Input (0, 1)

Let's calculate the output for $(x_1, x_2) = (0, 1)$ (should give output ≈ 1)

#### Step 1: Input Layer
$$
a^{(0)} = [x_1, x_2] = [0, 1]
$$

#### Step 2: Hidden Layer - Weighted Sums

**For h₁:**
$$
z^{(1)}_1 = W^{(1)}_{11} x_1 + W^{(1)}_{12} x_2 + b^{(1)}_1
$$

💻 **Calculate this:** (fill in the values)

$$
z^{(1)}_1 = 20 \times ___ + 20 \times ___ + (___) = ___
$$

<details>
<summary>Solution</summary>
$$
z^{(1)}_1 = 20 \times 0 + 20 \times 1 + (-10) = 10
$$
</details>

**For h₂:**
$$
z^{(1)}_2 = 20 \times 0 + 20 \times 1 + (-30) = -10
$$

#### Step 3: Hidden Layer - Activations

We use the **sigmoid** activation:
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**For h₁:**
$$
a^{(1)}_1 = \sigma(10) = \frac{1}{1 + e^{-10}} \approx 0.9999
$$

**For h₂:**
$$
a^{(1)}_2 = \sigma(-10) = \frac{1}{1 + e^{10}} \approx 0.0000
$$

**Interpretation:** 
- h₁ is **strongly activated** (≈1)
- h₂ is **not activated** (≈0)

#### Step 4: Output Layer - Weighted Sum

$$
z^{(2)} = W^{(2)}_{11} a^{(1)}_1 + W^{(2)}_{21} a^{(1)}_2 + b^{(2)}
$$

💻 **Your turn:** Calculate this!

$$
z^{(2)} = 20 \times ___ + (-20) \times ___ + (___) = ___
$$

<details>
<summary>Solution</summary>
$$
z^{(2)} = 20 \times 0.9999 + (-20) \times 0.0000 + (-10) \approx 10
$$
</details>

#### Step 5: Output Layer - Activation

$$
\hat{y} = \sigma(10) \approx 0.9999 \approx 1
$$

**Result:** Input (0,1) → Output ≈ 1 ✅ (Correct for XOR!)

### 💻 Code It: Complete Forward Propagation

Now let's implement this in code. **Fill in the missing parts:**

```python
def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

def forward_propagation_xor(x1, x2, W1, b1, W2, b2):
    """
    Forward propagation for 2-2-1 network
    
    Parameters:
    -----------
    x1, x2 : float
        Input values
    W1 : array, shape (2, 2)
        Weights from input to hidden layer
    b1 : array, shape (2,)
        Biases for hidden layer
    W2 : array, shape (2, 1)
        Weights from hidden to output layer
    b2 : float
        Bias for output layer
    
    Returns:
    --------
    output : float
        Network output
    h : array, shape (2,)
        Hidden layer activations
    """
    # Input layer
    x = np.array([x1, x2])
    
    # Hidden layer
    # TODO: Calculate z1 (weighted sum for hidden layer)
    z1 = W1 @ ___ + ___  # Fill in the blanks!
    
    # TODO: Apply sigmoid activation
    h = ___(___)  # Fill in!
    
    # Output layer
    # TODO: Calculate z2 (weighted sum for output)
    z2 = W2.T @ ___ + ___  # Fill in the blanks!
    
    # TODO: Apply sigmoid activation
    output = sigmoid(___)  # Fill in!
    
    return output, h

# Define weights (given)
W1 = np.array([[20, 20],
               [20, 20]])
b1 = np.array([-10, -30])

W2 = np.array([[20],
               [-20]])
b2 = np.array([-10])

# Test on all XOR inputs
print("XOR Network Test:")
print("="*50)
for x1, x2, y_true in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
    output, h = forward_propagation_xor(x1, x2, W1, b1, W2, b2)
    predicted = 1 if output > 0.5 else 0
    status = "✓" if predicted == y_true else "✗"
    print(f"{status} Input: ({x1}, {x2}) → Output: {output:.4f} → Predicted: {predicted} (True: {y_true})")
    print(f"   Hidden: h1={h[0]:.4f}, h2={h[1]:.4f}")
```

<details>
<summary>Solution for blanks</summary>

```python
# Hidden layer
z1 = W1 @ x + b1
h = sigmoid(z1)

# Output layer
z2 = W2.T @ h + b2
output = sigmoid(z2)
```
</details>

### ✏️ Exercise 5.1: Manual Calculation Practice

Calculate the output for input **(1, 1)** manually (show all steps):

1. Calculate $z^{(1)}_1$ and $z^{(1)}_2$
2. Calculate $a^{(1)}_1$ and $a^{(1)}_2$
3. Calculate $z^{(2)}$
4. Calculate final output $\hat{y}$
5. Is this correct for XOR?

**Work it out on paper first!**

<details>
<summary>Solution</summary>

**Step 1: Hidden layer weighted sums**
- $z^{(1)}_1 = 20(1) + 20(1) - 10 = 30$
- $z^{(1)}_2 = 20(1) + 20(1) - 30 = 10$

**Step 2: Hidden layer activations**
- $a^{(1)}_1 = \sigma(30) \approx 1.0$
- $a^{(1)}_2 = \sigma(10) \approx 1.0$

**Step 3: Output weighted sum**
- $z^{(2)} = 20(1) + (-20)(1) + (-10) = -10$

**Step 4: Output activation**
- $\hat{y} = \sigma(-10) \approx 0.0$

**Step 5: Verification**
- XOR(1,1) should be 0 ✓ **CORRECT!**
</details>

---

## 6. Understanding What Hidden Neurons Learn {#understanding}

### Visualizing Hidden Layer Activations

Let's see what each hidden neuron "detects":

```python
def visualize_hidden_activations(W1, b1):
    """
    Visualize what each hidden neuron activates on
    """
    # Create a grid of points
    x1_range = np.linspace(-0.5, 1.5, 100)
    x2_range = np.linspace(-0.5, 1.5, 100)
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
    
    # Calculate activations for each hidden neuron
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx in range(2):  # For h1 and h2
        ax = axes[idx]
        
        # Calculate activation for this hidden neuron on entire grid
        Z = np.zeros_like(X1_grid)
        for i in range(X1_grid.shape[0]):
            for j in range(X1_grid.shape[1]):
                x = np.array([X1_grid[i, j], X2_grid[i, j]])
                z = W1[idx] @ x + b1[idx]
                Z[i, j] = sigmoid(z)
        
        # Plot
        contour = ax.contourf(X1_grid, X2_grid, Z, levels=20, cmap='RdYlBu_r')
        plt.colorbar(contour, ax=ax)
        
        # Overlay XOR points
        ax.scatter([0, 1], [0, 1], s=200, c='blue', marker='o', 
                   edgecolors='black', linewidth=2, label='XOR = 0')
        ax.scatter([0, 1], [1, 0], s=200, c='red', marker='s', 
                   edgecolors='black', linewidth=2, label='XOR = 1')
        
        ax.set_xlabel('$x_1$', fontsize=12)
        ax.set_ylabel('$x_2$', fontsize=12)
        ax.set_title(f'Hidden Neuron h{idx+1} Activation', fontsize=14)
        ax.legend()
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
    
    plt.tight_layout()
    plt.show()

# Run with our XOR weights
visualize_hidden_activations(W1, b1)
```

### 🤔 Interpretation Question

Look at the activation maps above. 

**Q1:** Which regions does h₁ activate on (show high values)?

<details>
<summary>Answer</summary>
h₁ activates on the **top-right region** (where $x_1 + x_2$ is large)
</details>

**Q2:** Which regions does h₂ activate on?

<details>
<summary>Answer</summary>
h₂ activates on the **very top-right region** (where $x_1 + x_2$ is even larger, due to bias=-30)
</details>

**Q3:** How does the output layer combine these to solve XOR?

<details>
<summary>Answer</summary>
Output = 20·h₁ - 20·h₂ - 10

- When only h₁ is active (e.g., at (0,1) or (1,0)): Output is positive → 1
- When both are active (at (1,1)): They cancel out → 0
- When neither is active (at (0,0)): Output is negative → 0
</details>

### The Power of Hidden Layers

**Key Insight:** Each hidden neuron creates its own linear boundary. The output layer **combines** these boundaries to create complex, non-linear decision regions!

### 💻 Code It: Decision Boundary Visualization

Complete this code to visualize the final decision boundary:

```python
def visualize_decision_boundary(W1, b1, W2, b2):
    """Visualize the final decision boundary"""
    # Create grid
    x1_range = np.linspace(-0.5, 1.5, 200)
    x2_range = np.linspace(-0.5, 1.5, 200)
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
    
    # Calculate output for each point
    Z = np.zeros_like(X1_grid)
    for i in range(X1_grid.shape[0]):
        for j in range(X1_grid.shape[1]):
            output, _ = forward_propagation_xor(
                X1_grid[i, j], X2_grid[i, j], W1, b1, W2, b2
            )
            Z[i, j] = output
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # TODO: Fill in - create a contour plot with the decision boundary at 0.5
    plt.contourf(X1_grid, X2_grid, Z, levels=[0, 0.5, 1], 
                colors=['___', '___'], alpha=0.3)  # Fill in colors!
    
    # Draw decision boundary
    plt.contour(X1_grid, X2_grid, Z, levels=[___],  # What level?
               colors='green', linewidths=3)
    
    # Plot XOR points
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    plt.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1],
               s=300, c='blue', marker='o', edgecolors='black', linewidth=3)
    plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1],
               s=300, c='red', marker='s', edgecolors='black', linewidth=3)
    
    plt.xlabel('$x_1$', fontsize=14)
    plt.ylabel('$x_2$', fontsize=14)
    plt.title('XOR Solution: Non-Linear Decision Boundary', fontsize=16)
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.show()

# visualize_decision_boundary(W1, b1, W2, b2)
```

<details>
<summary>Solution for blanks</summary>

```python
plt.contourf(X1_grid, X2_grid, Z, levels=[0, 0.5, 1], 
            colors=['blue', 'red'], alpha=0.3)

plt.contour(X1_grid, X2_grid, Z, levels=[0.5],
           colors='green', linewidths=3)
```
</details>

---

## 7. Manual Weight Tuning Challenge {#manual}

### The Problem

We just saw that with the **right** weights, a 2-2-1 network can solve XOR perfectly.

But how do we **find** those weights?

### Challenge: Manual Weight Search

Let's try to find good weights ourselves!

```python
import ipywidgets as widgets
from IPython.display import display

def interactive_xor_network():
    """
    Interactive widget to manually tune weights
    """
    # Create sliders for weights
    w11_slider = widgets.FloatSlider(value=1, min=-30, max=30, step=1, 
                                     description='W1[0,0]:')
    w12_slider = widgets.FloatSlider(value=1, min=-30, max=30, step=1,
                                     description='W1[0,1]:')
    w21_slider = widgets.FloatSlider(value=1, min=-30, max=30, step=1,
                                     description='W1[1,0]:')
    w22_slider = widgets.FloatSlider(value=1, min=-30, max=30, step=1,
                                     description='W1[1,1]:')
    b1_slider = widgets.FloatSlider(value=0, min=-30, max=30, step=1,
                                    description='b1[0]:')
    b2_slider = widgets.FloatSlider(value=0, min=-30, max=30, step=1,
                                    description='b1[1]:')
    
    w_out1_slider = widgets.FloatSlider(value=1, min=-30, max=30, step=1,
                                       description='W2[0]:')
    w_out2_slider = widgets.FloatSlider(value=1, min=-30, max=30, step=1,
                                       description='W2[1]:')
    b_out_slider = widgets.FloatSlider(value=0, min=-30, max=30, step=1,
                                      description='b2:')
    
    def update_plot(w11, w12, w21, w22, b1, b2, w_out1, w_out2, b_out):
        # Create weight matrices
        W1_manual = np.array([[w11, w12], [w21, w22]])
        b1_manual = np.array([b1, b2])
        W2_manual = np.array([[w_out1], [w_out2]])
        b2_manual = np.array([b_out])
        
        # Test on XOR
        print("Testing on XOR:")
        print("="*50)
        correct = 0
        for x1, x2, y_true in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
            output, h = forward_propagation_xor(x1, x2, W1_manual, b1_manual, 
                                               W2_manual, b2_manual)
            predicted = 1 if output > 0.5 else 0
            status = "✓" if predicted == y_true else "✗"
            if predicted == y_true:
                correct += 1
            print(f"{status} ({x1},{x2}) → {output:.3f} → {predicted} (true: {y_true})")
        
        print(f"\nAccuracy: {correct}/4 = {correct/4*100:.0f}%")
        
        # Visualize
        visualize_decision_boundary(W1_manual, b1_manual, W2_manual, b2_manual)
    
    # Create interactive widget
    widgets.interactive(update_plot, 
                       w11=w11_slider, w12=w12_slider,
                       w21=w21_slider, w22=w22_slider,
                       b1=b1_slider, b2=b2_slider,
                       w_out1=w_out1_slider, w_out2=w_out2_slider,
                       b_out=b_out_slider)

# Uncomment to try (works in Jupyter):
# interactive_xor_network()
```

### ✏️ Exercise 7.1: Manual Tuning Reflection

Try to manually find weights that solve XOR. After 5 minutes, answer:

**Q1:** Were you able to find a perfect solution? _______

**Q2:** How long did it take? _______

**Q3:** How many parameters (weights + biases) do you need to tune? _______

**Q4:** For this simple 2-2-1 network, there are 9 parameters. How would you feel about tuning a 784-100-10 network (79,510 parameters)? 

<details>
<summary>Answer for Q3</summary>
**9 parameters total:**
- Layer 1: 4 weights + 2 biases = 6
- Layer 2: 2 weights + 1 bias = 3
</details>

### The Problem with Manual Tuning

**Issues:**
1. ❌ **Time-consuming**: Even for tiny networks
2. ❌ **Not systematic**: Trial and error
3. ❌ **Doesn't scale**: Impossible for large networks
4. ❌ **Not optimal**: Hard to find the best solution

**We need an AUTOMATIC way to find good weights!**

---

## 8. Why We Need Something Better {#motivation}

### What We Can Do Now

✅ **Design** network architectures  
✅ **Compute** outputs (forward propagation)  
✅ **Understand** what hidden layers do  

### What We Can't Do Yet

❌ **Find** good weights automatically  
❌ **Train** on large datasets efficiently  
❌ **Optimize** systematically  

### The Missing Piece: Gradient Descent

To train networks automatically, we need:

1. **A way to measure "how wrong" we are** → **Loss functions** (Session 5)
2. **A direction to adjust weights** → **Gradients** (Session 5)
3. **A way to calculate gradients efficiently** → **Backpropagation** (Session 6)

### 🤔 Final Reflection

**Before moving to the next module, think about:**

1. We can solve XOR with the right weights
2. Finding those weights manually is impractical
3. We need an algorithm that:
   - Measures how wrong the current weights are
   - Adjusts weights to make them better
   - Repeats until weights are good

**This is exactly what gradient descent + backpropagation does!**

### The Journey Ahead

```
Where we are now:         Where we're going:
                         
 Random weights    →     Measure error      →    Calculate gradients
      │                        │                         │
      ▼                        ▼                         ▼
 Forward prop       →     Loss function     →    Backpropagation
      │                        │                         │
      ▼                        ▼                         ▼
 Manual tuning     →     Gradient descent  →    Automatic training
   (slow! 😫)                                      (fast! 😊)
```

---

## 9. Final Exercises {#exercises}

### 📝 Exercise 9.1: Architecture Design (Easy)

Design an MLP architecture for each task:

**a) Binary classification with 10 input features**
- Input layer: ___
- Hidden layer(s): ___
- Output layer: ___

**b) Multi-class classification (5 classes) with 20 input features**
- Input layer: ___
- Hidden layer(s): ___
- Output layer: ___

**c) For task (a), how many parameters (weights + biases) if you use one hidden layer with 8 neurons?**

<details>
<summary>Solution</summary>

**a)** 10-8-1 (or any reasonable hidden layer size)
- Input: 10 features
- Hidden: 8 neurons (example)
- Output: 1 neuron (binary → single output with sigmoid)

**b)** 20-16-5 (or any reasonable hidden layer size)
- Input: 20 features
- Hidden: 16 neurons (example)
- Output: 5 neurons (one per class, with softmax)

**c)** For 10-8-1:
- Layer 1: $(10 \times 8) + 8 = 88$ parameters
- Layer 2: $(8 \times 1) + 1 = 9$ parameters
- **Total: 97 parameters**
</details>

---

### 📝 Exercise 9.2: Forward Propagation Practice (Medium)

Given this network:
- Input: $[2, 3]$
- Hidden layer (2 neurons, ReLU): 
  - $W^{(1)} = \begin{bmatrix} 1 & -1 \\ 0 & 2 \end{bmatrix}, b^{(1)} = \begin{bmatrix} 0 \\ -3 \end{bmatrix}$
- Output layer (1 neuron, sigmoid):
  - $W^{(2)} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}, b^{(2)} = -1$

**Calculate:**
1. Hidden layer outputs (after ReLU)
2. Final output (after sigmoid)

**Show all steps!**

<details>
<summary>Solution</summary>

**Step 1: Hidden layer weighted sums**
$$
z^{(1)} = W^{(1)} x + b^{(1)} = \begin{bmatrix} 1 & -1 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} 2 \\ 3 \end{bmatrix} + \begin{bmatrix} 0 \\ -3 \end{bmatrix}
$$
$$
z^{(1)} = \begin{bmatrix} 2-3 \\ 0+6 \end{bmatrix} + \begin{bmatrix} 0 \\ -3 \end{bmatrix} = \begin{bmatrix} -1 \\ 3 \end{bmatrix}
$$

**Step 2: Apply ReLU**
$$
h = \text{ReLU}(z^{(1)}) = \begin{bmatrix} \max(0, -1) \\ \max(0, 3) \end{bmatrix} = \begin{bmatrix} 0 \\ 3 \end{bmatrix}
$$

**Step 3: Output weighted sum**
$$
z^{(2)} = W^{(2)T} h + b^{(2)} = \begin{bmatrix} 2 & 1 \end{bmatrix} \begin{bmatrix} 0 \\ 3 \end{bmatrix} + (-1)
$$
$$
z^{(2)} = 0 + 3 - 1 = 2
$$

**Step 4: Apply sigmoid**
$$
\hat{y} = \sigma(2) = \frac{1}{1+e^{-2}} \approx 0.88
$$

**Final answer:** $\hat{y} \approx 0.88$
</details>

---

### 📝 Exercise 9.3: Implementing a 3-Layer Network (Hard)

Implement a 2-3-1 network (2 inputs → 3 hidden neurons → 1 output) with:
- ReLU activation in hidden layer
- Sigmoid activation in output

```python
def three_layer_network(x1, x2, W1, b1, W2, b2):
    """
    Implement forward propagation for 2-3-1 network
    
    Parameters:
    -----------
    x1, x2 : float
        Input values
    W1 : array, shape (3, 2)
        Weights input -> hidden
    b1 : array, shape (3,)
        Biases for hidden layer
    W2 : array, shape (3, 1)
        Weights hidden -> output
    b2 : float
        Bias for output
    
    Returns:
    --------
    output : float
        Final prediction
    hidden : array, shape (3,)
        Hidden layer activations
    """
    # TODO: Implement this!
    # 1. Create input vector
    # 2. Calculate hidden layer (with ReLU)
    # 3. Calculate output (with sigmoid)
    
    pass

# Test your implementation
W1_test = np.random.randn(3, 2)
b1_test = np.random.randn(3)
W2_test = np.random.randn(3, 1)
b2_test = np.random.randn(1)

output, hidden = three_layer_network(0.5, 0.5, W1_test, b1_test, W2_test, b2_test)
print(f"Output: {output}")
print(f"Hidden: {hidden}")
```

<details>
<summary>Solution</summary>

```python
def three_layer_network(x1, x2, W1, b1, W2, b2):
    # Input
    x = np.array([x1, x2])
    
    # Hidden layer
    z1 = W1 @ x + b1
    hidden = np.maximum(0, z1)  # ReLU
    
    # Output layer
    z2 = W2.T @ hidden + b2
    output = sigmoid(z2[0])
    
    return output, hidden
```
</details>

---

### 📝 Exercise 9.4: Activation Function Comparison (Medium)

Compare different activation functions for the hidden layer:

```python
def test_activations_on_xor():
    """
    Test XOR with different activation functions
    """
    # Same architecture, different activations
    activations = {
        'sigmoid': sigmoid,
        'relu': lambda z: np.maximum(0, z),
        'tanh': lambda z: np.tanh(z),
        'step': lambda z: np.where(z >= 0, 1, 0)
    }
    
    # Test each activation
    # TODO: Implement and compare!
    
    pass
```

**Questions:**
1. Which activation works best for XOR?
2. Why might step function fail?
3. What's the advantage of ReLU over sigmoid?

<details>
<summary>Discussion</summary>

1. **Sigmoid and tanh** work well for XOR (smooth, differentiable)
2. **Step function** is problematic because:
   - Not differentiable
   - Can't use gradient-based learning (coming in Session 5!)
3. **ReLU advantages:**
   - Computationally faster (just max(0,z))
   - Avoids vanishing gradient problem
   - Works well in practice for deep networks
</details>

---

### 📝 Exercise 9.5: The Universal Approximation Intuition (Hard)

**Theoretical question:**

The Universal Approximation Theorem states: "A neural network with one hidden layer can approximate any continuous function to arbitrary accuracy (given enough hidden neurons)."

**Thought experiment:**

1. Draw a complex wavy function $f(x)$ on paper
2. How could you approximate it with:
   - Many small step functions?
   - Combining many ReLU neurons?

3. For a 2D input function $f(x_1, x_2)$, how does adding more hidden neurons help?

**Hint:** Think about how each hidden neuron creates a "basis function" that the output layer combines.

---

## Summary

### What We Learned

✅ **XOR Problem**: Single perceptrons can't solve it (not linearly separable)  
✅ **Hidden Layers**: Multiple perceptrons working together can solve non-linear problems  
✅ **MLP Architecture**: Input → Hidden(s) → Output  
✅ **Forward Propagation**: Computing outputs layer by layer  
✅ **Hidden Layer Role**: Each neuron detects different features  
✅ **Manual Tuning**: Impractical and doesn't scale  

### Key Insights

1. **Non-linearity comes from:**
   - Multiple linear boundaries (hidden neurons)
   - Non-linear activations (sigmoid, ReLU)
   - Combination at output layer

2. **Forward propagation is simple:**
   - Just weighted sums + activations
   - Repeat for each layer
   - Easy to implement!

3. **The hard part:**
   - Finding good weights
   - Need automatic training
   - → Motivation for gradient descent!

### What's Next?

**Module 3: The Mathematics of Learning**

In the next session, we'll learn:
- **Loss functions**: How to measure "how wrong" we are
- **Gradient descent**: How to adjust weights systematically
- **Derivatives**: The math we need (don't worry, we'll build it up!)

**The goal:** Automatically train our networks instead of guessing weights!

### Before Next Session

**Think about:**
1. How would you measure if your weights are "good" or "bad"?
2. If you could see which direction to move each weight, how would you use that information?
3. What does "learning" mean mathematically?

**Optional reading:**
- 3Blue1Brown: "But what is a neural network?" (YouTube)
- Visualization: https://playground.tensorflow.org/ (play with the interactive network!)

---

**End of Session 4** 🎓

**You now understand:**
- ✅ Multi-layer networks solve non-linear problems
- ✅ Forward propagation computes predictions
- ✅ We need automatic training methods

**Next up:** Making networks learn by themselves! 🚀
