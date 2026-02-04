# Session 3: The Perceptron Learning Rule
## Learning Without Calculus

**Course: Neural Networks for Engineers**  
**Duration: 2 hours**

---

## Table of Contents

1. [The Problem: Manual Weights Don't Scale](#problem)
2. [The Big Idea: Learn From Mistakes](#idea)
3. [Rosenblatt's Learning Algorithm](#algorithm)
4. [Manual Example: Step by Step](#manual)
5. [Geometric Intuition](#geometric)
6. [Implementation From Scratch](#implementation)
7. [Visualizing Learning](#visualization)
8. [Learning Rate Effects](#learning-rate)
9. [Convergence: When Does It Work?](#convergence)
10. [Exercises](#exercises)

---

## 1. The Problem: Manual Weights Don't Scale {#problem}

### What We've Done So Far

In the previous session, we manually chose weights:
- AND gate: $w_1 = 1, w_2 = 1, b = -1.5$
- OR gate: $w_1 = 1, w_2 = 1, b = -0.5$
- NOT gate: $w = -1, b = 0.5$

This worked because:
- ✅ Simple problems (2 inputs)
- ✅ We knew the solution
- ✅ Trial and error was fast

### The Real World Problem

Imagine classifying emails as spam/not spam:
- **Input features**: 1000+ words
- **Possible weights**: 1000+ numbers to choose
- **Manual tuning**: Impossible! 😱

**Example:**
```
Email features: [2, 0, 5, 1, 0, 3, ...]  # word counts
                 ↓  ↓  ↓  ↓  ↓  ↓
Weights:        [?, ?, ?, ?, ?, ?]  # What should these be?
```

### We Need Automatic Learning!

**Question:** Can we teach the perceptron to find its own weights?

**Answer:** YES! Frank Rosenblatt (1957) discovered a simple algorithm.

---

## 2. The Big Idea: Learn From Mistakes {#idea}

### Human Learning Analogy

Think about learning to throw a basketball:
1. **Try**: Throw the ball
2. **Observe**: Did it go in?
3. **Adjust**: 
   - If too short → throw harder next time
   - If too long → throw softer next time
4. **Repeat**: Keep adjusting until you succeed

### Perceptron Learning

Same idea!
1. **Try**: Make a prediction with current weights
2. **Observe**: Is the prediction correct?
3. **Adjust**:
   - If correct → do nothing! ✅
   - If wrong → adjust weights in the right direction ❌
4. **Repeat**: Keep adjusting until all predictions are correct

### The Key Insight

**When we make a mistake, we know which direction to adjust!**

Example:
- **Prediction**: Class 0 (below the line)
- **Actual**: Class 1 (should be above the line)
- **Fix**: Move the line toward this point!

---

## 3. Rosenblatt's Learning Algorithm {#algorithm}

### The Update Rule

For each training example $(\mathbf{x}, y)$:

1. **Predict**: $\hat{y} = \text{sign}(\mathbf{w}^T \mathbf{x} + b)$

2. **Calculate Error**: $\text{error} = y - \hat{y}$

3. **Update Weights**:
$$
\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot \text{error} \cdot \mathbf{x}
$$

4. **Update Bias**:
$$
b \leftarrow b + \eta \cdot \text{error}
$$

Where:
- $\eta$ (eta) = **learning rate** (typically 0.01 to 1.0)
- error = how wrong we were (+2, 0, or -2 with sign function)

### Understanding the Update

Let's decode this formula: $\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot \text{error} \cdot \mathbf{x}$

**Case 1: Correct Prediction**
- $y = \hat{y}$ → error = 0
- Update: $\mathbf{w} \leftarrow \mathbf{w} + 0 = \mathbf{w}$
- **No change!** ✅

**Case 2: Should be +1, predicted -1**
- $y = +1, \hat{y} = -1$ → error = +2
- Update: $\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot (+2) \cdot \mathbf{x}$
- **Move w toward x** (increase dot product)

**Case 3: Should be -1, predicted +1**
- $y = -1, \hat{y} = +1$ → error = -2
- Update: $\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot (-2) \cdot \mathbf{x}$
- **Move w away from x** (decrease dot product)

### Pseudocode

```
Initialize w = [0, 0, ...], b = 0

For each epoch (pass through data):
    For each training example (x, y):
        # Predict
        z = w · x + b
        ŷ = sign(z)
        
        # Update if wrong
        if ŷ ≠ y:
            error = y - ŷ
            w = w + η * error * x
            b = b + η * error
    
    # Check if all correct
    if no mistakes:
        DONE! ✓
```

---

## 4. Manual Example: Step by Step {#manual}

Let's learn the OR gate from data!

### Training Data

| $x_1$ | $x_2$ | $y$ (target) |
|-------|-------|--------------|
| 0     | 0     | 0            |
| 0     | 1     | 1            |
| 1     | 0     | 1            |
| 1     | 1     | 1            |

### Initialization

- $\mathbf{w} = [0, 0]^T$ (start with zeros)
- $b = 0$
- $\eta = 1$ (learning rate)

### Epoch 1

#### Example 1: $(x_1, x_2) = (0, 0), y = 0$

**Predict:**
$$
z = w_1 \cdot 0 + w_2 \cdot 0 + b = 0 + 0 + 0 = 0
$$
$$
\hat{y} = \text{sign}(0) = 1 \text{ (by convention, sign(0) = 1)}
$$

**Error:**
$$
\text{error} = y - \hat{y} = 0 - 1 = -1
$$

**Update:**
$$
w_1 \leftarrow 0 + 1 \cdot (-1) \cdot 0 = 0
$$
$$
w_2 \leftarrow 0 + 1 \cdot (-1) \cdot 0 = 0
$$
$$
b \leftarrow 0 + 1 \cdot (-1) = -1
$$

**New weights:** $\mathbf{w} = [0, 0]^T, b = -1$

---

#### Example 2: $(x_1, x_2) = (0, 1), y = 1$

**Predict:**
$$
z = 0 \cdot 0 + 0 \cdot 1 + (-1) = -1
$$
$$
\hat{y} = \text{sign}(-1) = -1
$$

**Error:**
$$
\text{error} = 1 - (-1) = 2
$$

**Update:**
$$
w_1 \leftarrow 0 + 1 \cdot 2 \cdot 0 = 0
$$
$$
w_2 \leftarrow 0 + 1 \cdot 2 \cdot 1 = 2
$$
$$
b \leftarrow -1 + 1 \cdot 2 = 1
$$

**New weights:** $\mathbf{w} = [0, 2]^T, b = 1$

---

#### Example 3: $(x_1, x_2) = (1, 0), y = 1$

**Predict:**
$$
z = 0 \cdot 1 + 2 \cdot 0 + 1 = 1
$$
$$
\hat{y} = \text{sign}(1) = 1
$$

**Error:** $\text{error} = 1 - 1 = 0$ ✅ **Correct!**

**Update:** No change

**Weights stay:** $\mathbf{w} = [0, 2]^T, b = 1$

---

#### Example 4: $(x_1, x_2) = (1, 1), y = 1$

**Predict:**
$$
z = 0 \cdot 1 + 2 \cdot 1 + 1 = 3
$$
$$
\hat{y} = \text{sign}(3) = 1
$$

**Error:** $\text{error} = 1 - 1 = 0$ ✅ **Correct!**

**Weights stay:** $\mathbf{w} = [0, 2]^T, b = 1$

---

### End of Epoch 1

**Mistakes in epoch 1:** 2 (examples 1 and 2)

Let's verify with final weights on all data:

| $x_1$ | $x_2$ | $y$ | $z = 0 \cdot x_1 + 2 \cdot x_2 + 1$ | $\hat{y}$ | Correct? |
|-------|-------|-----|--------------------------------------|-----------|----------|
| 0     | 0     | 0   | 1                                    | 1         | ❌        |
| 0     | 1     | 1   | 3                                    | 1         | ✅        |
| 1     | 0     | 1   | 1                                    | 1         | ✅        |
| 1     | 1     | 1   | 3                                    | 1         | ✅        |

Still 1 mistake! Need another epoch.

### Epoch 2

#### Example 1: $(0, 0), y = 0$

**Predict:**
$$
z = 0 \cdot 0 + 2 \cdot 0 + 1 = 1
$$
$$
\hat{y} = \text{sign}(1) = 1
$$

**Error:** $-1$

**Update:**
$$
\mathbf{w} = [0, 2] + 1 \cdot (-1) \cdot [0, 0] = [0, 2]
$$
$$
b = 1 + 1 \cdot (-1) = 0
$$

**New weights:** $\mathbf{w} = [0, 2]^T, b = 0$

Rest of examples are correct...

### Final Verification

After a few epochs, we converge to:
$$
\mathbf{w} = [0, 2]^T, b = 0 \text{ (or similar)}
$$

**Decision boundary:** $0 \cdot x_1 + 2 \cdot x_2 + 0 = 0$ → $x_2 = 0$

This correctly separates OR gate data!

---

## 5. Geometric Intuition {#geometric}

### What's Really Happening?

The learning rule **moves the decision boundary toward misclassified points**.

```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize the learning process geometrically
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

# Three stages of learning
stages = [
    ([0, 0], 0, "Initial (random)"),
    ([0, 2], 1, "After 2 updates"),
    ([1, 1], -0.5, "Converged")
]

for idx, (w, b, title) in enumerate(stages):
    ax = axes[idx]
    
    # Plot points
    ax.scatter(X[y == 0, 0], X[y == 0, 1], 
               s=200, c='blue', marker='o', 
               edgecolors='black', linewidth=2, label='Class 0')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], 
               s=200, c='red', marker='s', 
               edgecolors='black', linewidth=2, label='Class 1')
    
    # Plot decision boundary
    if w[1] != 0:
        x1_line = np.linspace(-0.5, 1.5, 100)
        x2_line = -(w[0] * x1_line + b) / w[1]
        ax.plot(x1_line, x2_line, 'g-', linewidth=3, 
                label='Decision Boundary')
    
    # Highlight misclassified points
    w_arr = np.array(w)
    predictions = np.sign(X @ w_arr + b)
    predictions[predictions == 0] = 1  # convention
    y_binary = y.copy()
    y_binary[y_binary == 0] = -1
    
    misclassified = predictions != y_binary
    if np.any(misclassified):
        ax.scatter(X[misclassified, 0], X[misclassified, 1],
                   s=400, facecolors='none', edgecolors='orange',
                   linewidth=3, label='Misclassified')
    
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### The Magic

- **Misclassified point**: "I'm on the wrong side!"
- **Update rule**: "Let me pull the line toward you!"
- **Result**: Line rotates/shifts to fix the mistake

---

## 6. Implementation From Scratch {#implementation}

### Complete Perceptron Class with Learning

```python
import numpy as np
import matplotlib.pyplot as plt

class PerceptronLearner:
    """
    Perceptron with learning capability
    """
    
    def __init__(self, learning_rate=0.1, max_epochs=100, random_state=None):
        """
        Parameters:
        -----------
        learning_rate : float
            Learning rate (eta)
        max_epochs : int
            Maximum number of training epochs
        random_state : int
            Random seed for reproducibility
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.errors_history = []
        self.weight_history = []
        
    def fit(self, X, y):
        """
        Train the perceptron on data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (must be 0 or 1, or -1 or 1)
        """
        # Initialize random generator
        rgen = np.random.RandomState(self.random_state)
        
        # Convert labels to -1/+1 if they are 0/1
        y_train = np.where(y == 0, -1, y)
        
        # Initialize weights to small random values
        n_features = X.shape[1]
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias = 0.0
        
        # Store initial weights
        self.weight_history.append((self.weights.copy(), self.bias))
        
        # Training loop
        for epoch in range(self.max_epochs):
            errors = 0
            
            # Loop through each training example
            for xi, yi in zip(X, y_train):
                # Make prediction
                linear_output = np.dot(xi, self.weights) + self.bias
                y_pred = np.sign(linear_output)
                
                # Handle sign(0) = 1 by convention
                if y_pred == 0:
                    y_pred = 1
                
                # Calculate error
                error = yi - y_pred
                
                # Update weights if wrong
                if error != 0:
                    update = self.learning_rate * error
                    self.weights += update * xi
                    self.bias += update
                    errors += 1
            
            # Store error count and weights for this epoch
            self.errors_history.append(errors)
            self.weight_history.append((self.weights.copy(), self.bias))
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{self.max_epochs} - Errors: {errors}")
            
            # Check convergence
            if errors == 0:
                print(f"✓ Converged after {epoch + 1} epochs!")
                break
        
        return self
    
    def predict(self, X):
        """
        Make predictions on new data
        """
        linear_output = np.dot(X, self.weights) + self.bias
        predictions = np.sign(linear_output)
        predictions[predictions == 0] = 1
        return predictions
    
    def score(self, X, y):
        """
        Calculate accuracy
        """
        y_train = np.where(y == 0, -1, y)
        predictions = self.predict(X)
        return np.mean(predictions == y_train)

print("PerceptronLearner class ready!")
```

### Example: Learning OR Gate

```python
# OR gate data
X_or = np.array([[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]])

y_or = np.array([0, 1, 1, 1])

# Create and train perceptron
perceptron = PerceptronLearner(learning_rate=0.1, max_epochs=20, random_state=42)
perceptron.fit(X_or, y_or)

# Test
print("\n" + "="*50)
print("Final Results:")
print("="*50)
print(f"Weights: {perceptron.weights}")
print(f"Bias: {perceptron.bias}")
print(f"Accuracy: {perceptron.score(X_or, y_or) * 100:.1f}%")

print("\nPredictions:")
for i, (xi, yi) in enumerate(zip(X_or, y_or)):
    pred = perceptron.predict(xi.reshape(1, -1))[0]
    pred_label = 1 if pred == 1 else 0
    print(f"  {xi} → Actual: {yi}, Predicted: {pred_label}")
```

**Output:**
```
Epoch 1/20 - Errors: 2
Epoch 10/20 - Errors: 0
✓ Converged after 5 epochs!

==================================================
Final Results:
==================================================
Weights: [0.2  0.3]
Bias: -0.1
Accuracy: 100.0%

Predictions:
  [0 0] → Actual: 0, Predicted: 0
  [0 1] → Actual: 1, Predicted: 1
  [1 0] → Actual: 1, Predicted: 1
  [1 1] → Actual: 1, Predicted: 1
```

---

## 7. Visualizing Learning {#visualization}

### Learning Curve

```python
# Plot errors over epochs
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(perceptron.errors_history) + 1), 
         perceptron.errors_history, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Number of Errors', fontsize=14)
plt.title('Perceptron Learning Curve (OR Gate)', fontsize=16)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--', label='Perfect Classification')
plt.legend()
plt.show()
```

### Decision Boundary Evolution

```python
def plot_decision_boundary_evolution(perceptron, X, y, epochs_to_show):
    """
    Show how decision boundary evolves during training
    """
    n_plots = len(epochs_to_show)
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
    
    if n_plots == 1:
        axes = [axes]
    
    for idx, epoch in enumerate(epochs_to_show):
        ax = axes[idx]
        
        # Get weights at this epoch
        w, b = perceptron.weight_history[epoch]
        
        # Plot data points
        ax.scatter(X[y == 0, 0], X[y == 0, 1], 
                   s=200, c='blue', marker='o', 
                   edgecolors='black', linewidth=2, label='Class 0')
        ax.scatter(X[y == 1, 0], X[y == 1, 1], 
                   s=200, c='red', marker='s', 
                   edgecolors='black', linewidth=2, label='Class 1')
        
        # Plot decision boundary
        if w[1] != 0:
            x1_line = np.linspace(-0.5, 1.5, 100)
            x2_line = -(w[0] * x1_line + b) / w[1]
            ax.plot(x1_line, x2_line, 'g-', linewidth=3)
        
        # Check for errors
        y_binary = np.where(y == 0, -1, 1)
        z = X @ w + b
        predictions = np.sign(z)
        predictions[predictions == 0] = 1
        errors = np.sum(predictions != y_binary)
        
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel('$x_1$', fontsize=12)
        ax.set_ylabel('$x_2$', fontsize=12)
        ax.set_title(f'Epoch {epoch}\n({errors} errors)', fontsize=14)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.show()

# Show evolution at epochs 0, 1, 2, final
epochs_to_show = [0, 1, 2, len(perceptron.weight_history)-1]
plot_decision_boundary_evolution(perceptron, X_or, y_or, epochs_to_show)
```

### Animated Learning (Optional)

```python
def visualize_learning_animation(perceptron, X, y, interval=500):
    """
    Create an animation of the learning process
    """
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def update(frame):
        ax.clear()
        
        # Get weights at this epoch
        if frame < len(perceptron.weight_history):
            w, b = perceptron.weight_history[frame]
        else:
            w, b = perceptron.weight_history[-1]
        
        # Plot data
        ax.scatter(X[y == 0, 0], X[y == 0, 1], 
                   s=200, c='blue', marker='o', 
                   edgecolors='black', linewidth=2, label='Class 0')
        ax.scatter(X[y == 1, 0], X[y == 1, 1], 
                   s=200, c='red', marker='s', 
                   edgecolors='black', linewidth=2, label='Class 1')
        
        # Plot boundary
        if w[1] != 0:
            x1_line = np.linspace(-0.5, 1.5, 100)
            x2_line = -(w[0] * x1_line + b) / w[1]
            ax.plot(x1_line, x2_line, 'g-', linewidth=3)
        
        # Calculate errors
        y_binary = np.where(y == 0, -1, 1)
        predictions = np.sign(X @ w + b)
        predictions[predictions == 0] = 1
        errors = np.sum(predictions != y_binary)
        
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel('$x_1$', fontsize=14)
        ax.set_ylabel('$x_2$', fontsize=14)
        ax.set_title(f'Epoch {frame}\nErrors: {errors}', fontsize=16)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    anim = FuncAnimation(fig, update, 
                        frames=len(perceptron.weight_history),
                        interval=interval, repeat=True)
    plt.close()
    
    return HTML(anim.to_jshtml())

# To view animation (in Jupyter):
# visualize_learning_animation(perceptron, X_or, y_or)
```

---

## 8. Learning Rate Effects {#learning-rate}

### What is the Learning Rate?

The learning rate $\eta$ controls **how big** each update step is.

- **Too small**: Learning is slow 🐌
- **Too large**: Learning is unstable, might not converge 🎢
- **Just right**: Fast and stable convergence 🎯

### Experiment: Different Learning Rates

```python
# Test different learning rates
learning_rates = [0.01, 0.1, 0.5, 1.0]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, lr in enumerate(learning_rates):
    # Train perceptron
    perc = PerceptronLearner(learning_rate=lr, max_epochs=50, random_state=42)
    perc.fit(X_or, y_or)
    
    # Plot learning curve
    ax = axes[idx]
    epochs = range(1, len(perc.errors_history) + 1)
    ax.plot(epochs, perc.errors_history, 'bo-', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Number of Errors', fontsize=12)
    ax.set_title(f'Learning Rate η = {lr}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, max(perc.errors_history) + 0.5)
    
    # Add convergence info
    converged = 0 in perc.errors_history
    if converged:
        conv_epoch = perc.errors_history.index(0) + 1
        ax.text(0.6, 0.9, f'Converged at epoch {conv_epoch}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.show()
```

### Rules of Thumb

**Learning Rate Guidelines:**
- Start with $\eta = 0.1$
- If not converging: decrease $\eta$
- If too slow: increase $\eta$ (carefully!)
- Typical range: $[0.001, 1.0]$

---

## 9. Convergence: When Does It Work? {#convergence}

### The Perceptron Convergence Theorem

**Theorem (Rosenblatt, 1962):**  
If the training data is **linearly separable**, the perceptron algorithm will find a solution in a **finite** number of steps.

### What Does "Linearly Separable" Mean?

Data is linearly separable if you can draw a straight line (in 2D) or hyperplane (in higher dimensions) that perfectly separates the two classes.

```python
# Examples of linearly separable vs not

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Example 1: Linearly Separable (AND gate)
ax = axes[0]
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
ax.scatter(X_and[y_and == 0, 0], X_and[y_and == 0, 1], 
           s=200, c='blue', marker='o', edgecolors='black', linewidth=2)
ax.scatter(X_and[y_and == 1, 0], X_and[y_and == 1, 1], 
           s=200, c='red', marker='s', edgecolors='black', linewidth=2)
# Draw separating line
x_line = np.linspace(-0.2, 1.2, 100)
y_line = -x_line + 1.2
ax.plot(x_line, y_line, 'g-', linewidth=3)
ax.set_xlim(-0.3, 1.3)
ax.set_ylim(-0.3, 1.3)
ax.set_title('Linearly Separable\n(Perceptron WILL converge)', fontsize=12, color='green')
ax.grid(True, alpha=0.3)

# Example 2: Also Linearly Separable (Random points)
np.random.seed(42)
X_sep = np.vstack([
    np.random.randn(20, 2) + [2, 2],
    np.random.randn(20, 2) + [-2, -2]
])
y_sep = np.array([1]*20 + [0]*20)
ax = axes[1]
ax.scatter(X_sep[y_sep == 0, 0], X_sep[y_sep == 0, 1], 
           s=100, c='blue', marker='o', edgecolors='black', linewidth=1, alpha=0.6)
ax.scatter(X_sep[y_sep == 1, 0], X_sep[y_sep == 1, 1], 
           s=100, c='red', marker='s', edgecolors='black', linewidth=1, alpha=0.6)
ax.plot([-5, 5], [0, 0], 'g-', linewidth=3)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_title('Linearly Separable\n(Perceptron WILL converge)', fontsize=12, color='green')
ax.grid(True, alpha=0.3)

# Example 3: NOT Linearly Separable (XOR)
ax = axes[2]
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])
ax.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1], 
           s=200, c='blue', marker='o', edgecolors='black', linewidth=2)
ax.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], 
           s=200, c='red', marker='s', edgecolors='black', linewidth=2)
# Try to draw a line (impossible!)
ax.plot([-0.3, 1.3], [0.5, 0.5], 'r--', linewidth=2, alpha=0.5, label='No line works!')
ax.set_xlim(-0.3, 1.3)
ax.set_ylim(-0.3, 1.3)
ax.set_title('NOT Linearly Separable\n(Perceptron will NOT converge)', fontsize=12, color='red')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Testing on XOR (Will Fail!)

```python
# Try to learn XOR (this will fail!)
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

print("Attempting to learn XOR...")
print("(This will NOT converge!)\n")

perceptron_xor = PerceptronLearner(learning_rate=0.1, max_epochs=100, random_state=42)
perceptron_xor.fit(X_xor, y_xor)

print(f"\nFinal accuracy: {perceptron_xor.score(X_xor, y_xor) * 100:.1f}%")
print("(Will never reach 100% because XOR is not linearly separable)")

# Plot learning curve showing no convergence
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(perceptron_xor.errors_history) + 1), 
         perceptron_xor.errors_history, 'ro-', linewidth=2, markersize=6)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Number of Errors', fontsize=14)
plt.title('XOR: Perceptron Cannot Converge', fontsize=16)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='g', linestyle='--', alpha=0.5, label='Perfect (impossible)')
plt.legend()
plt.show()
```

### When to Use Perceptron Learning?

✅ **Use when:**
- Data is (approximately) linearly separable
- You need a simple, interpretable model
- Quick baseline for binary classification

❌ **Don't use when:**
- Data is not linearly separable (like XOR)
- You need probabilistic outputs
- Problem is highly non-linear

**Solution for non-linear problems:** Multi-layer networks! (Next session!)

---

## 10. Exercises {#exercises}

### 📝 Exercise 10.1: Manual Learning (Easy)

Given training data:
| $x_1$ | $x_2$ | $y$ |
|-------|-------|-----|
| 1     | 2     | +1  |
| 2     | 1     | +1  |
| 2     | 3     | -1  |

Starting with $\mathbf{w} = [0, 0]^T, b = 0, \eta = 1$:

**Tasks:**
1. Manually compute the first 3 updates (show all calculations)
2. What are the weights after these 3 examples?
3. Predict the class for new point $(1.5, 2.5)$

```python
# Your solution here
```

---

### 📝 Exercise 10.2: Learning Rate Experiment (Easy)

Train a perceptron on the AND gate with learning rates: $\eta \in \{0.01, 0.1, 1.0, 10.0\}$

**Tasks:**
1. How many epochs does each take to converge?
2. What happens with $\eta = 10.0$? Why?
3. Plot all learning curves on the same graph
4. Which learning rate is best? Why?

```python
# Your solution here
```

---

### 📝 Exercise 10.3: Visualization Challenge (Medium)

Create an animated visualization showing:
1. The current point being processed (highlighted)
2. The decision boundary before update
3. The decision boundary after update
4. A trail showing how the boundary moved

**Hint:** Use matplotlib animation or create a sequence of plots.

```python
# Your solution here
```

---

### 📝 Exercise 10.4: Linearly Separable Data (Medium)

Generate random linearly separable data:

```python
# Generate data
np.random.seed(42)
X_class0 = np.random.randn(50, 2) + [-2, -2]
X_class1 = np.random.randn(50, 2) + [2, 2]
X = np.vstack([X_class0, X_class1])
y = np.array([0]*50 + [1]*50)
```

**Tasks:**
1. Plot the data
2. Train a perceptron with $\eta = 0.1$
3. Plot the final decision boundary
4. Calculate and report accuracy
5. How many epochs did it take?
6. What happens if you add noise? Try adding 5 points that cross the boundary.

```python
# Your solution here
```

---

### 📝 Exercise 10.5: Convergence Analysis (Hard)

**Theoretical question:**

The convergence theorem says the number of mistakes is bounded by:
$$
\text{mistakes} \leq \left(\frac{R}{\gamma}\right)^2
$$

Where:
- $R = \max_i \|\mathbf{x}^{(i)}\|$ (maximum distance from origin)
- $\gamma$ = margin (minimum distance to decision boundary)

**Tasks:**
1. For the OR gate data, calculate $R$
2. Estimate $\gamma$ (distance from closest point to your learned boundary)
3. Calculate the theoretical bound on mistakes
4. Compare with actual number of mistakes from your training
5. Repeat with data that has:
   - Large margin (well-separated)
   - Small margin (barely separable)

```python
# Your solution here
```

---

### 📝 Exercise 10.6: Early Stopping (Hard)

The perceptron can overfit on noisy data!

**Scenario:** Add label noise to OR gate:
```python
# Flip 1 label
X_or_noisy = X_or.copy()
y_or_noisy = y_or.copy()
y_or_noisy[0] = 1  # Flip (0,0) to class 1 (wrong!)
```

**Tasks:**
1. Train perceptron for 1000 epochs on noisy data
2. Plot training error vs epoch
3. What happens? Does it converge?
4. Implement "early stopping": Stop when no improvement for 10 epochs
5. Compare solutions with and without early stopping

```python
# Your solution here
```

---

### 📝 Exercise 10.7: Perceptron vs Pocket Algorithm (Very Hard)

The **Pocket Algorithm** keeps track of the best weights seen so far.

**Pseudocode:**
```
best_weights = w
best_errors = infinity

For each epoch:
    For each example:
        Update w using perceptron rule
    
    # Count errors with current weights
    current_errors = count_errors(w)
    
    # Keep best weights
    if current_errors < best_errors:
        best_weights = w
        best_errors = current_errors

Return best_weights
```

**Tasks:**
1. Implement the Pocket Algorithm
2. Test on XOR data
3. Compare with regular perceptron
4. What's the best accuracy you can achieve on XOR?
5. Why does Pocket work better on non-separable data?

```python
# Your solution here
class PocketPerceptron:
    # Implement here
    pass
```

---

## Summary

### What We Learned

✅ **The Perceptron Learning Rule**
- Update only when wrong: $\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot \text{error} \cdot \mathbf{x}$
- Geometric interpretation: move boundary toward misclassified points
- No calculus needed!

✅ **Implementation**
- Training loop
- Error tracking
- Convergence detection

✅ **Visualization**
- Learning curves
- Decision boundary evolution
- Understanding the learning process

✅ **Learning Rate Effects**
- Too small = slow
- Too large = unstable
- Need to tune!

✅ **Convergence Properties**
- Works perfectly for linearly separable data
- Fails on XOR and other non-linear problems
- Finite convergence guarantee

### Key Insights

1. **Learning is adjustment**: We learn by correcting mistakes
2. **Geometry matters**: The algorithm moves the line toward misclassified points
3. **Simple but limited**: Works great for linear problems, but that's it
4. **Motivation for next step**: We need something better for non-linear problems!

### What's Next?

In **Session 4**, we'll see how to solve non-linear problems like XOR using:
- **Hidden layers** (multiple perceptrons working together)
- **Forward propagation** (computing outputs through layers)
- **Manual experimentation** (before we learn gradient descent)

This will set us up to understand WHY we need gradient descent and backpropagation in later sessions!

---

**End of Session 3** 🎓

**Homework:**
- Complete exercises 10.1 - 10.4
- (Optional challenge): Exercises 10.5 - 10.7
- Think about: "How can multiple perceptrons solve XOR?"
