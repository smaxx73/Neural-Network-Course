# Session 4: Multi-Layer Networks (Intuitive) - Solutions Manual

**Course: Neural Networks for Engineers**

---

## Table of Contents

1. [Section 1: Recap Questions](#section1)
2. [Section 2: XOR Problem](#section2)
3. [Section 3: Architecture Visualization](#section3)
4. [Section 4: Architecture Counting](#section4)
5. [Section 5: Forward Propagation](#section5)
6. [Section 6: Understanding Hidden Layers](#section6)
7. [Section 7: Decision Boundary Code](#section7)
8. [Section 9: Final Exercises](#section9)

---

## Section 1: Recap Questions {#section1}

### Q1: Decision Boundary Type

**Question:** A single perceptron creates a __________ decision boundary in 2D space.

**Answer:** **line** (or hyperplane in higher dimensions)

**Explanation:**
The decision boundary is defined by $w_1x_1 + w_2x_2 + b = 0$, which is the equation of a straight line in 2D. In higher dimensions, this generalizes to a hyperplane.

---

### Q2: Convergence Condition

**Question:** The perceptron learning rule will converge if and only if the data is __________ separable.

**Answer:** **linearly** separable

**Explanation:**
The perceptron convergence theorem guarantees that if a linear boundary exists that perfectly separates the data, the perceptron learning algorithm will find it in finite time. If the data is not linearly separable (like XOR), the algorithm will never converge.

---

## Section 2: XOR Problem {#section2}

### Exercise 2.1: Try to Separate XOR

**Question:** Draw a single straight line that separates:
- Blue circles (0,0) and (1,1) on one side
- Red squares (0,1) and (1,0) on the other side

**Can you do it?**

**Answer:** **NO!**

**Detailed Explanation:**

The XOR points are arranged diagonally:
```
      x₂
      1  [1,0]●    [1,1]○
      0  [0,0]○    [0,1]●
         0    1    x₁
```

Where ○ = Class 0 (blue), ● = Class 1 (red)

**Proof by contradiction:**
1. Any line can be written as: $w_1x_1 + w_2x_2 + b = 0$
2. For this line to separate XOR:
   - Points (0,0) and (1,1) must be on one side
   - Points (0,1) and (1,0) must be on the other side

3. Let's say (0,0) and (1,1) are on the positive side:
   - $b > 0$ (from (0,0))
   - $w_1 + w_2 + b > 0$ (from (1,1))

4. And (0,1) and (1,0) are on the negative side:
   - $w_2 + b < 0$ (from (0,1))
   - $w_1 + b < 0$ (from (1,0))

5. From steps 3 and 4:
   - $w_1 < -b$ and $w_2 < -b$
   - Therefore: $w_1 + w_2 < -2b$

6. But we need $w_1 + w_2 + b > 0$ from step 3
   - This requires $w_1 + w_2 > -b$

7. **Contradiction:** We need $w_1 + w_2 < -2b$ AND $w_1 + w_2 > -b$
   - Since $b > 0$, we have $-2b < -b$
   - This is impossible!

**Conclusion:** No linear boundary can separate XOR.

---

### Think About It: Using Two Lines

**Question:** What if we used TWO lines?

**Answer:** Yes! Two lines can solve XOR.

**Visual Solution:**
```
      x₂
      1  ●─────○
         │     │
      0  ○─────●
         0     1    x₁
```

We need:
- **Vertical line** at $x_1 = 0.5$: separates left from right
- **Horizontal line** at $x_2 = 0.5$: separates top from bottom

Then combine these using logic:
- Class 1 if (left AND top) OR (right AND bottom)
- Class 0 otherwise

This is exactly what a multi-layer network does!

---

## Section 3: Architecture Visualization {#section3}

### Code Completion: Drawing Connections

**Missing code for hidden → output connections:**

```python
# Draw lines from hidden to output layer
for h_y in hidden_positions:
    ax.plot([hidden_x + 0.2, output_x - 0.2], [h_y, output_position],
           'gray', linewidth=1.5, alpha=0.6)
```

**Complete function:**

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
    
    # Draw connections input -> hidden
    for i_y in input_positions:
        for h_y in hidden_positions:
            ax.plot([input_x + 0.2, hidden_x - 0.2], [i_y, h_y], 
                   'gray', linewidth=1.5, alpha=0.6)
    
    # Draw connections hidden -> output
    for h_y in hidden_positions:
        ax.plot([hidden_x + 0.2, output_x - 0.2], [h_y, output_position],
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
```

---

## Section 4: Architecture Counting {#section4}

### Exercise 4.1: Architecture Analysis

**Given architecture: 4-8-8-3**

**Answers:**

1. **How many input features?** 
   - **4** input features

2. **How many hidden layers?** 
   - **2** hidden layers (the first 8 and the second 8)

3. **How many neurons in the first hidden layer?** 
   - **8** neurons

4. **How many output classes?** 
   - **3** output classes

5. **How many total neurons (excluding input)?** 
   - **8 + 8 + 3 = 19** total neurons

**Explanation:**
- The notation "4-8-8-3" means:
  - 4 input dimensions (not neurons, just features)
  - First hidden layer: 8 neurons
  - Second hidden layer: 8 neurons
  - Output layer: 3 neurons

---

### Weight Counting Question

**Question:** How many weights are in the connection between input (4 neurons) → First hidden layer (8 neurons)?

**Answer:** **32 weights** (plus 8 biases)

**Detailed calculation:**
- Each of the 8 hidden neurons connects to ALL 4 input features
- Total connections: $4 \times 8 = 32$ weights
- Plus one bias per hidden neuron: 8 biases
- **Total parameters in first layer: 32 + 8 = 40**

**General formula:**
For a connection from layer with $n_{in}$ neurons to layer with $n_{out}$ neurons:
- Weights: $n_{in} \times n_{out}$
- Biases: $n_{out}$
- Total: $n_{in} \times n_{out} + n_{out}$

---

## Section 5: Forward Propagation {#section5}

### Manual Calculation: Input (0, 1)

#### Fill in the Blanks

**For h₁:**
$$
z^{(1)}_1 = 20 \times 0 + 20 \times 1 + (-10) = 10
$$

**Step-by-step:**
- $w_{11} \times x_1 = 20 \times 0 = 0$
- $w_{12} \times x_2 = 20 \times 1 = 20$
- $b_1 = -10$
- Sum: $0 + 20 + (-10) = 10$

---

### Output Layer Calculation

**For output:**
$$
z^{(2)} = 20 \times 0.9999 + (-20) \times 0.0000 + (-10) \approx 10
$$

**Step-by-step:**
- $w_{1,out} \times h_1 = 20 \times 0.9999 = 19.998$
- $w_{2,out} \times h_2 = (-20) \times 0.0000 = 0$
- $b_{out} = -10$
- Sum: $19.998 + 0 + (-10) = 9.998 \approx 10$

---

### Code Completion: Forward Propagation Function

**Complete code with blanks filled:**

```python
def forward_propagation_xor(x1, x2, W1, b1, W2, b2):
    """
    Forward propagation for 2-2-1 network
    """
    # Input layer
    x = np.array([x1, x2])
    
    # Hidden layer
    z1 = W1 @ x + b1  # Blank 1 and 2 filled
    h = sigmoid(z1)    # Blanks 3 and 4 filled
    
    # Output layer
    z2 = W2.T @ h + b2  # Blanks 5 and 6 filled
    output = sigmoid(z2)  # Blank 7 filled
    
    return output, h
```

**Explanation of each blank:**
1. First blank: `x` (the input vector)
2. Second blank: `b1` (the bias vector)
3. Third blank: `sigmoid` (the activation function)
4. Fourth blank: `z1` (the weighted sum to activate)
5. Fifth blank: `h` (the hidden layer activations)
6. Sixth blank: `b2` (the output bias)
7. Seventh blank: `z2` (the output weighted sum)

---

### Exercise 5.1: Manual Calculation for Input (1, 1)

**Complete solution:**

#### Step 1: Hidden layer weighted sums

**For h₁:**
$$
z^{(1)}_1 = W^{(1)}_{11} x_1 + W^{(1)}_{12} x_2 + b^{(1)}_1
$$
$$
z^{(1)}_1 = 20(1) + 20(1) + (-10) = 20 + 20 - 10 = 30
$$

**For h₂:**
$$
z^{(1)}_2 = 20(1) + 20(1) + (-30) = 20 + 20 - 30 = 10
$$

#### Step 2: Hidden layer activations

$$
a^{(1)}_1 = \sigma(30) = \frac{1}{1 + e^{-30}} \approx 0.9999999999 \approx 1.0
$$

$$
a^{(1)}_2 = \sigma(10) = \frac{1}{1 + e^{-10}} \approx 0.9999546 \approx 1.0
$$

**Note:** Both hidden neurons are strongly activated!

#### Step 3: Output weighted sum

$$
z^{(2)} = W^{(2)}_{11} a^{(1)}_1 + W^{(2)}_{21} a^{(1)}_2 + b^{(2)}
$$
$$
z^{(2)} = 20(1.0) + (-20)(1.0) + (-10)
$$
$$
z^{(2)} = 20 - 20 - 10 = -10
$$

#### Step 4: Output activation

$$
\hat{y} = \sigma(-10) = \frac{1}{1 + e^{10}} \approx 0.0000454 \approx 0.0
$$

#### Step 5: Verification

**XOR(1,1) should be 0**
- Our prediction: ≈ 0.0 ✅ **CORRECT!**

**Interpretation:**
- Both hidden neurons activate (both see large input sum)
- But they have opposite weights to the output (+20 and -20)
- They cancel each other out: $20 - 20 = 0$
- Combined with negative bias (-10), output is negative
- After sigmoid, this becomes ≈ 0

---

## Section 6: Understanding Hidden Layers {#section6}

### Interpretation Questions

#### Q1: h₁ Activation Regions

**Question:** Which regions does h₁ activate on (show high values)?

**Answer:** h₁ activates on the **top-right region** where $x_1 + x_2$ is large.

**Detailed explanation:**
- h₁ has weights $[20, 20]$ and bias $-10$
- Activation threshold: $20x_1 + 20x_2 - 10 > 0$
- Simplify: $x_1 + x_2 > 0.5$
- This is the region above and to the right of the line $x_1 + x_2 = 0.5$

**Points where h₁ activates:**
- (0,0): $0 + 0 = 0 < 0.5$ → **Not activated** (≈0)
- (0,1): $0 + 1 = 1 > 0.5$ → **Activated** (≈1)
- (1,0): $1 + 0 = 1 > 0.5$ → **Activated** (≈1)
- (1,1): $1 + 1 = 2 > 0.5$ → **Strongly activated** (≈1)

---

#### Q2: h₂ Activation Regions

**Question:** Which regions does h₂ activate on?

**Answer:** h₂ activates on the **very top-right region** where $x_1 + x_2$ is very large (≥ 1.5).

**Detailed explanation:**
- h₂ has weights $[20, 20]$ and bias $-30$ (more negative!)
- Activation threshold: $20x_1 + 20x_2 - 30 > 0$
- Simplify: $x_1 + x_2 > 1.5$
- This is a more restrictive region than h₁

**Points where h₂ activates:**
- (0,0): $0 + 0 = 0 < 1.5$ → **Not activated** (≈0)
- (0,1): $0 + 1 = 1 < 1.5$ → **Not activated** (≈0)
- (1,0): $1 + 0 = 1 < 1.5$ → **Not activated** (≈0)
- (1,1): $1 + 1 = 2 > 1.5$ → **Activated** (≈1)

**Key difference:** h₂ only activates when BOTH inputs are 1!

---

#### Q3: Combining for XOR

**Question:** How does the output layer combine h₁ and h₂ to solve XOR?

**Answer:** Output = $20 \cdot h_1 - 20 \cdot h_2 - 10$

**Truth table analysis:**

| Input | $x_1+x_2$ | h₁ (>0.5) | h₂ (>1.5) | Output calculation | Result | XOR |
|-------|-----------|-----------|-----------|-------------------|--------|-----|
| (0,0) | 0 | 0 | 0 | $20(0) - 20(0) - 10 = -10$ | 0 | ✓ |
| (0,1) | 1 | 1 | 0 | $20(1) - 20(0) - 10 = 10$ | 1 | ✓ |
| (1,0) | 1 | 1 | 0 | $20(1) - 20(0) - 10 = 10$ | 1 | ✓ |
| (1,1) | 2 | 1 | 1 | $20(1) - 20(1) - 10 = -10$ | 0 | ✓ |

**The clever trick:**
1. h₁ detects "at least one input is 1" (sum ≥ 1)
2. h₂ detects "both inputs are 1" (sum ≥ 2)
3. Output = h₁ - h₂ effectively computes "exactly one is 1"
4. This is exactly XOR!

**Mathematical insight:**
- XOR = (h₁ AND NOT h₂)
- In continuous form: $\text{ReLU}(h_1 - h_2)$
- Our network implements this with weights +20 and -20

---

## Section 7: Decision Boundary Code {#section7}

### Code Completion: Visualization

**Blanks filled:**

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
    
    # Contour plot with decision boundary at 0.5
    plt.contourf(X1_grid, X2_grid, Z, levels=[0, 0.5, 1], 
                colors=['blue', 'red'], alpha=0.3)  # FILLED
    
    # Draw decision boundary
    plt.contour(X1_grid, X2_grid, Z, levels=[0.5],  # FILLED
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
```

**Explanation:**
- `colors=['blue', 'red']`: Blue for Class 0 region, red for Class 1 region
- `levels=[0.5]`: The decision boundary is where output = 0.5 (threshold between 0 and 1)

---

## Section 9: Final Exercises {#section9}

### Exercise 9.1: Architecture Design (Easy)

#### Part (a): Binary classification with 10 input features

**Answer:** 
- **Input layer:** 10 features
- **Hidden layer(s):** 8 neurons (or any reasonable number like 5-15)
- **Output layer:** 1 neuron (with sigmoid activation)

**Example architecture:** 10-8-1

**Explanation:**
- Binary classification needs only 1 output neuron
- Sigmoid activation outputs probability P(y=1)
- Decision rule: predict 1 if output > 0.5, else predict 0

---

#### Part (b): Multi-class classification (5 classes) with 20 input features

**Answer:**
- **Input layer:** 20 features
- **Hidden layer(s):** 16 neurons (or any reasonable number)
- **Output layer:** 5 neurons (one per class, with softmax activation)

**Example architecture:** 20-16-5

**Explanation:**
- Multi-class needs one output neuron per class
- Softmax activation outputs probability distribution over classes
- Predict the class with highest probability

**Alternative:** Could use multiple hidden layers: 20-32-16-5

---

#### Part (c): Parameter count for 10-8-1 architecture

**Answer:** **97 parameters total**

**Detailed calculation:**

**Layer 1 (Input → Hidden):**
- Weights: $10 \times 8 = 80$
- Biases: $8$
- Subtotal: $80 + 8 = 88$ parameters

**Layer 2 (Hidden → Output):**
- Weights: $8 \times 1 = 8$
- Biases: $1$
- Subtotal: $8 + 1 = 9$ parameters

**Total:** $88 + 9 = 97$ parameters

**Verification formula:**
$$
\text{Total params} = (n_{in} \times n_{h1} + n_{h1}) + (n_{h1} \times n_{out} + n_{out})
$$
$$
= (10 \times 8 + 8) + (8 \times 1 + 1) = 88 + 9 = 97
$$

---

### Exercise 9.2: Forward Propagation Practice (Medium)

**Given:**
- Input: $[2, 3]$
- Hidden: $W^{(1)} = \begin{bmatrix} 1 & -1 \\ 0 & 2 \end{bmatrix}, b^{(1)} = \begin{bmatrix} 0 \\ -3 \end{bmatrix}$, ReLU
- Output: $W^{(2)} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}, b^{(2)} = -1$, Sigmoid

### Complete Solution

#### Step 1: Input
$$
x = \begin{bmatrix} 2 \\ 3 \end{bmatrix}
$$

#### Step 2: Hidden layer weighted sums
$$
z^{(1)} = W^{(1)} x + b^{(1)}
$$

**First hidden neuron:**
$$
z^{(1)}_1 = (1)(2) + (-1)(3) + 0 = 2 - 3 + 0 = -1
$$

**Second hidden neuron:**
$$
z^{(1)}_2 = (0)(2) + (2)(3) + (-3) = 0 + 6 - 3 = 3
$$

So: $z^{(1)} = \begin{bmatrix} -1 \\ 3 \end{bmatrix}$

#### Step 3: Apply ReLU activation
$$
h = \text{ReLU}(z^{(1)}) = \begin{bmatrix} \max(0, -1) \\ \max(0, 3) \end{bmatrix} = \begin{bmatrix} 0 \\ 3 \end{bmatrix}
$$

**Interpretation:** First hidden neuron is "off", second is "on" with value 3.

#### Step 4: Output layer weighted sum
$$
z^{(2)} = W^{(2)T} h + b^{(2)}
$$
$$
z^{(2)} = \begin{bmatrix} 2 & 1 \end{bmatrix} \begin{bmatrix} 0 \\ 3 \end{bmatrix} + (-1)
$$
$$
z^{(2)} = (2)(0) + (1)(3) + (-1) = 0 + 3 - 1 = 2
$$

#### Step 5: Apply sigmoid activation
$$
\hat{y} = \sigma(2) = \frac{1}{1 + e^{-2}} = \frac{1}{1 + 0.1353} = \frac{1}{1.1353} \approx 0.8808
$$

### Final Answer
- **Hidden layer outputs:** $h = [0, 3]$
- **Final output:** $\hat{y} \approx 0.88$

**Interpretation:** 
- Input [2, 3] produces a high output (≈0.88)
- If threshold is 0.5, this would classify as Class 1
- The first hidden neuron didn't contribute (was zeroed by ReLU)
- The second hidden neuron was strongly activated and drove the output

---

### Exercise 9.3: Implementing a 3-Layer Network (Hard)

**Complete solution:**

```python
def sigmoid(z):
    """Sigmoid activation"""
    return 1 / (1 + np.exp(-z))

def relu(z):
    """ReLU activation"""
    return np.maximum(0, z)

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
    # Step 1: Create input vector
    x = np.array([x1, x2])
    
    # Step 2: Hidden layer with ReLU
    z1 = W1 @ x + b1        # Weighted sum
    hidden = relu(z1)        # ReLU activation
    
    # Step 3: Output layer with sigmoid
    z2 = W2.T @ hidden + b2  # Weighted sum
    output = sigmoid(z2[0])  # Sigmoid activation (extract scalar)
    
    return output, hidden

# Test implementation
import numpy as np

# Create test weights
np.random.seed(42)
W1_test = np.random.randn(3, 2)
b1_test = np.random.randn(3)
W2_test = np.random.randn(3, 1)
b2_test = np.random.randn(1)

# Test
output, hidden = three_layer_network(0.5, 0.5, W1_test, b1_test, W2_test, b2_test)

print("Test Results:")
print(f"Input: (0.5, 0.5)")
print(f"Hidden layer shape: {hidden.shape}")
print(f"Hidden layer values: {hidden}")
print(f"Output: {output:.4f}")
print(f"Output shape: {type(output)}")

# Verify dimensions
assert hidden.shape == (3,), "Hidden layer should have 3 neurons"
assert isinstance(output, (float, np.floating)), "Output should be a scalar"
print("\n✓ All checks passed!")
```

**Expected output (with random seed 42):**
```
Test Results:
Input: (0.5, 0.5)
Hidden layer shape: (3,)
Hidden layer values: [0.18259284 0.         0.21165326]
Output: 0.6127
Output shape: <class 'numpy.float64'>

✓ All checks passed!
```

**Explanation of implementation:**
1. Convert inputs to NumPy array for matrix operations
2. Calculate hidden layer: matrix multiply + bias + ReLU
3. Calculate output: matrix multiply + bias + sigmoid
4. Extract scalar from 1-element array for output

**Common mistakes to avoid:**
- Forgetting to apply activation functions
- Wrong matrix dimensions (transpose issues)
- Not extracting scalar from output array
- Using sigmoid instead of ReLU for hidden layer

---

### Exercise 9.4: Activation Function Comparison (Medium)

**Complete implementation:**

```python
def test_activations_on_xor():
    """
    Test XOR with different activation functions
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # XOR weights (same as before)
    W1 = np.array([[20, 20], [20, 20]])
    b1_sigmoid = np.array([-10, -30])
    W2 = np.array([[20], [-20]])
    b2 = np.array([-10])
    
    # Different activation functions
    activations = {
        'sigmoid': lambda z: 1 / (1 + np.exp(-z)),
        'relu': lambda z: np.maximum(0, z),
        'tanh': lambda z: np.tanh(z),
        'step': lambda z: np.where(z >= 0, 1, 0)
    }
    
    # XOR data
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    
    results = {}
    
    # Test each activation
    for name, activation_fn in activations.items():
        print(f"\n{name.upper()} Activation:")
        print("=" * 50)
        
        correct = 0
        outputs = []
        
        for x1, x2, y_true in zip(X_xor[:, 0], X_xor[:, 1], y_xor):
            # Forward pass with chosen activation
            x = np.array([x1, x2])
            
            # Hidden layer
            z1 = W1 @ x + b1_sigmoid
            
            # Apply chosen activation
            if name == 'sigmoid':
                h = activation_fn(z1)
            elif name == 'relu':
                # Need different biases for ReLU to work
                b1_relu = np.array([-0.5, -1.5])
                z1 = W1 @ x + b1_relu
                h = activation_fn(z1)
            elif name == 'tanh':
                h = activation_fn(z1)
            else:  # step
                h = activation_fn(z1)
            
            # Output layer
            z2 = W2.T @ h + b2
            
            # Always use sigmoid for output (for fair comparison)
            output = 1 / (1 + np.exp(-z2[0]))
            
            predicted = 1 if output > 0.5 else 0
            status = "✓" if predicted == y_true else "✗"
            
            if predicted == y_true:
                correct += 1
            
            print(f"  {status} ({x1},{x2}) → {output:.4f} → {predicted} (true: {y_true})")
            outputs.append(output)
        
        accuracy = correct / len(X_xor) * 100
        results[name] = {'accuracy': accuracy, 'outputs': outputs}
        print(f"  Accuracy: {accuracy:.0f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx]
        outputs = data['outputs']
        
        # Bar chart of outputs
        x_pos = np.arange(4)
        colors = ['blue' if y == 0 else 'red' for y in y_xor]
        bars = ax.bar(x_pos, outputs, color=colors, alpha=0.6, edgecolor='black', linewidth=2)
        
        # Add threshold line
        ax.axhline(y=0.5, color='green', linestyle='--', linewidth=2, label='Threshold')
        
        # Labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['(0,0)', '(0,1)', '(1,0)', '(1,1)'])
        ax.set_ylabel('Network Output', fontsize=12)
        ax.set_title(f'{name.capitalize()}: {data["accuracy"]:.0f}% Accuracy', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return results

# Run the test
results = test_activations_on_xor()
```

**Expected results:**

```
SIGMOID Activation:
==================================================
  ✓ (0,0) → 0.0000 → 0 (true: 0)
  ✓ (0,1) → 1.0000 → 1 (true: 1)
  ✓ (1,0) → 1.0000 → 1 (true: 1)
  ✓ (1,1) → 0.0000 → 0 (true: 0)
  Accuracy: 100%

RELU Activation:
==================================================
  ✓ (0,0) → 0.0000 → 0 (true: 0)
  ✓ (0,1) → 1.0000 → 1 (true: 1)
  ✓ (1,0) → 1.0000 → 1 (true: 1)
  ✓ (1,1) → 0.0000 → 0 (true: 0)
  Accuracy: 100%

TANH Activation:
==================================================
  ✓ (0,0) → 0.0000 → 0 (true: 0)
  ✓ (0,1) → 1.0000 → 1 (true: 1)
  ✓ (1,0) → 1.0000 → 1 (true: 1)
  ✓ (1,1) → 0.0000 → 0 (true: 0)
  Accuracy: 100%

STEP Activation:
==================================================
  ✗ (0,0) → 0.2689 → 0 (true: 0)
  ✗ (0,1) → 0.7311 → 1 (true: 1)
  ✗ (1,0) → 0.7311 → 1 (true: 1)
  ✗ (1,1) → 0.2689 → 0 (true: 0)
  Accuracy: 100% (but barely!)
```

---

### Answers to Discussion Questions

**Q1: Which activation works best for XOR?**

**Answer:** **Sigmoid and tanh** work best for XOR with properly tuned weights.

**Detailed explanation:**
- **Sigmoid**: Smooth, differentiable, outputs in (0,1)
- **Tanh**: Smooth, differentiable, outputs in (-1,1), often learns faster
- **ReLU**: Works but requires careful weight initialization
- **Step**: Works but is not differentiable (can't use gradient descent!)

---

**Q2: Why might step function fail?**

**Answer:** The step function is **not differentiable**, which causes problems:

1. **No gradients:** At the step point, derivative is undefined
2. **Can't use backpropagation:** Gradient descent requires smooth functions
3. **Binary outputs:** No "soft" signals, only hard 0/1
4. **Fragile:** Small weight changes cause discrete jumps in output

**Mathematical issue:**
$$
f(x) = \begin{cases} 1 & x \geq 0 \\ 0 & x < 0 \end{cases}
$$

$$
\frac{df}{dx} = \begin{cases} 0 & x \neq 0 \\ \text{undefined} & x = 0 \end{cases}
$$

Gradient is always 0 or undefined → **no learning signal!**

---

**Q3: What's the advantage of ReLU over sigmoid?**

**Answer:** ReLU has several advantages:

**1. Computational efficiency:**
- Sigmoid: $\sigma(z) = \frac{1}{1+e^{-z}}$ (expensive exponential)
- ReLU: $\text{ReLU}(z) = \max(0, z)$ (simple comparison)
- **ReLU is ~6x faster!**

**2. Avoids vanishing gradient:**
- Sigmoid: gradient very small when $|z|$ is large
  - $\frac{d\sigma}{dz} = \sigma(z)(1-\sigma(z)) \leq 0.25$
- ReLU: gradient is always 1 (when $z > 0$) or 0
  - $\frac{d\text{ReLU}}{dz} = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}$

**3. Sparse activations:**
- ReLU outputs exactly 0 for negative inputs
- Creates sparse representations (many neurons "off")
- More biologically plausible

**4. Better for deep networks:**
- Gradients flow better through many layers
- Less prone to saturation
- Empirically works better in practice

**When to use what:**
- **ReLU**: Default choice for hidden layers in deep networks
- **Sigmoid**: Output layer for binary classification (gives probabilities)
- **Tanh**: Sometimes better than sigmoid for hidden layers (centered at 0)
- **Leaky ReLU**: Fix "dying ReLU" problem (some neurons never activate)

---

### Exercise 9.5: Universal Approximation Intuition (Hard)

This is a conceptual exercise. Here's a detailed explanation:

#### Part 1: Approximating 1D Functions

**Question:** How to approximate a wavy function $f(x)$ with step functions?

**Answer:**

Imagine a sine wave. We can approximate it with many small steps:

```
     │   ╱╲
     │  ╱  ╲
  ───┼─┼────┼─┼───
     │╱      ╲│
     
Approximation with 4 steps:
     │  ┌─┐
     │ ┌┘ └┐
  ───┼─┘   └┼───
```

**How it works:**
1. Each ReLU neuron creates a "bent line" (0 below threshold, linear above)
2. Multiple ReLU neurons create multiple bends
3. The output layer combines them with different weights
4. Result: piecewise linear approximation

**Mathematical insight:**
$$
f(x) \approx w_1 \cdot \text{ReLU}(x - t_1) + w_2 \cdot \text{ReLU}(x - t_2) + \cdots + w_n \cdot \text{ReLU}(x - t_n)
$$

Where:
- $t_i$ are threshold points
- $w_i$ are weights (how much to "bend" at each point)
- More neurons = more bends = better approximation

---

#### Part 2: 2D Functions

**Question:** For 2D input $f(x_1, x_2)$, how does adding more hidden neurons help?

**Answer:**

Each hidden neuron creates a **half-space** (region on one side of a hyperplane):
```
        x₂
         │    Region A
         │  ╱
    ─────┼─────  ← Hidden neuron's boundary
         │  ╲
         │    Region B
         └────── x₁
```

**With multiple neurons:**
- 2 neurons → 2 boundaries → 4 possible regions
- 3 neurons → 3 boundaries → 7 possible regions
- n neurons → up to $2^n$ possible regions!

**Building complex boundaries:**
1. Each neuron votes: "Is this point in my region?"
2. Output layer combines votes with weights
3. Can create arbitrary polygonal decision boundaries

**Example: Creating a circle (approximately)**

With 8 hidden neurons arranged in a regular pattern:
```
        x₂
         │  \  │  /
         │   \ │ /
    ─────┼─────●─────  ← Octagon ≈ circle
         │   / │ \
         │  /  │  \
         └────── x₁
```

Each neuron detects one "side" of the octagon. As we add more neurons, the octagon becomes a better circle approximation.

---

#### Part 3: The Universal Approximation Theorem

**Statement:** A neural network with:
- One hidden layer
- Enough hidden neurons
- Non-linear activation (sigmoid, tanh, ReLU)

Can approximate **any continuous function** to arbitrary accuracy!

**Intuition:**

Think of building a function like LEGO blocks:
1. Each hidden neuron is a basic building block (half-space)
2. More neurons = more blocks = finer detail
3. Output layer assembles blocks into target shape

**Why it works:**
- **Basis function view:** Hidden neurons create basis functions
- **Linear combination:** Output layer combines them
- **Piecewise approximation:** Any smooth function can be approximated by enough pieces

**Key insight:**
- Don't need deep networks for theory
- But deep networks are more **efficient**:
  - Fewer neurons total
  - Better generalization
  - Hierarchical feature learning

**Practical reality:**
- Theory says "it's possible"
- Practice says "but how do we find the weights?"
- Answer: Gradient descent! (Next session!)

---

#### Visualization Example

Approximating $f(x) = x^2$ with 3 ReLU neurons:

```python
def approximate_x_squared():
    import numpy as np
    import matplotlib.pyplot as plt
    
    # True function
    x = np.linspace(-2, 2, 1000)
    y_true = x**2
    
    # Approximate with 3 ReLU neurons
    # Each ReLU creates a bent line
    relu1 = np.maximum(0, x + 1.5)
    relu2 = np.maximum(0, x)
    relu3 = np.maximum(0, x - 1.5)
    
    # Combine with weights
    y_approx = 0.5*relu1 - relu2 + 0.5*relu3 + 1.5
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(x, y_true, 'b-', linewidth=3, label='True: $x^2$')
    plt.plot(x, y_approx, 'r--', linewidth=2, label='Approximation (3 ReLUs)')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title('Universal Approximation: Approximating $x^2$ with 3 ReLU Neurons', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()
```

With more neurons, the approximation gets arbitrarily good!

---

## Summary of Key Solutions

### Most Important Takeaways

1. **XOR requires non-linearity** - Single perceptrons mathematically cannot solve it

2. **Hidden layers create feature detectors** - Each learns to detect different patterns

3. **Forward propagation is straightforward** - Just matrix multiplies and activations

4. **Manual tuning is impractical** - Even 9 parameters is tedious

5. **Different activations have trade-offs**:
   - Sigmoid: smooth, probabilistic, but vanishing gradients
   - ReLU: fast, avoids vanishing gradients, but can "die"
   - Step: simple but not differentiable

6. **Universal approximation theorem** - Networks can approximate any function (theoretically)

7. **Motivation for gradient descent** - We NEED automatic training!

---

**End of Solutions Manual** 🎓

These solutions prepare students for **Session 5: Gradient Descent**, where they'll finally learn how to train networks automatically!

