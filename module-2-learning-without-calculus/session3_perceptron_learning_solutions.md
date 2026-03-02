
# Solutions: Session 3 - The Perceptron Learning Rule

## 📝 Exercise 10.1: Manual Learning (Easy)

**Initial data:**
*   $\mathbf{w} = [0, 0]^T, b = 0, \eta = 1$
*   By convention (seen in the lesson): $\text{sign}(0) = 1$

**1. Manual calculation of the first 3 updates:**

*   **Example 1:** $\mathbf{x}^{(1)} = [1, 2]^T, y^{(1)} = 1$
    *   Prediction: $z = (0 \cdot 1) + (0 \cdot 2) + 0 = 0$
    *   $\hat{y} = \text{sign}(0) = 1$
    *   Error = $y - \hat{y} = 1 - 1 = 0$
    *   **Update:** None (Correct prediction ✅)
    *   *Current weights:* $\mathbf{w} = [0, 0]^T, b = 0$

*   **Example 2:** $\mathbf{x}^{(2)} = [2, 1]^T, y^{(2)} = 1$
    *   Prediction: $z = (0 \cdot 2) + (0 \cdot 1) + 0 = 0$
    *   $\hat{y} = \text{sign}(0) = 1$
    *   Error = $y - \hat{y} = 1 - 1 = 0$
    *   **Update:** None (Correct prediction ✅)
    *   *Current weights:* $\mathbf{w} = [0, 0]^T, b = 0$

*   **Example 3:** $\mathbf{x}^{(3)} = [2, 3]^T, y^{(3)} = -1$
    *   Prediction: $z = (0 \cdot 2) + (0 \cdot 3) + 0 = 0$
    *   $\hat{y} = \text{sign}(0) = 1$
    *   Error = $y - \hat{y} = -1 - 1 = -2$
    *   **Update:** 
        *   $\mathbf{w} \leftarrow [0, 0] + 1 \cdot (-2) \cdot [2, 3] = [-4, -6]^T$
        *   $b \leftarrow 0 + 1 \cdot (-2) = -2$

**2. Weights after these 3 examples:**
$\mathbf{w} = [-4, -6]^T$ and $b = -2$

**3. Prediction for the new point $(1.5, 2.5)$:**
*   $z = (-4 \cdot 1.5) + (-6 \cdot 2.5) + (-2) = -6 - 15 - 2 = -23$
*   $\hat{y} = \text{sign}(-23) = -1$
*   **Prediction: Class -1**

---

## 📝 Exercise 10.2: Learning Rate Experiment (Easy)

```python
import numpy as np
import matplotlib.pyplot as plt

# AND gate data
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

learning_rates = [0.01, 0.1, 1.0, 10.0]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for lr, ax in zip(learning_rates, axes.flatten()):
    # Note: Assuming the PerceptronLearner class is defined as in the lesson
    perc = PerceptronLearner(learning_rate=lr, max_epochs=20, random_state=42)
    perc.fit(X_and, y_and)
    
    epochs = range(1, len(perc.errors_history) + 1)
    ax.plot(epochs, perc.errors_history, 'mo-', linewidth=2)
    ax.set_title(f'Learning Rate = {lr}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Errors')
    ax.grid(True)
    
    print(f"LR {lr}: Converged in {len(perc.errors_history)} epochs. Final weights: {perc.weights}, Bias: {perc.bias}")

plt.tight_layout()
plt.show()
```

**Answers to the questions:**
1. **How many epochs?** The number of epochs to converge is often **identical** (or very similar) regardless of the learning rate for a simple perceptron starting at zero.
2. **What happens with $\eta = 10.0$?** The perceptron still converges perfectly if the data is linearly separable. However, the final weights will simply be much larger (scaled up). *Math note:* If initial weights are 0, the perceptron algorithm makes exactly the same sequence of mistakes regardless of $\eta$; it just multiplies the weight vector by this scalar.
4. **Which learning rate is best?** $\eta = 0.1$ or $1.0$ is generally a good compromise. A rate that is too large (like 10) creates huge weights, which can cause numerical instability (overflow) in deeper networks, even if it doesn't bother a simple single-layer perceptron.

---

## 📝 Exercise 10.3: Visualization Challenge (Medium)

*Here is a straightforward approach without complex animation, generating a clear side-by-side plot showing the decision boundary "Before" and "After" an update.*

```python
def visualize_single_update(X, y):
    w = np.array([0.0, 0.0])
    b = 0.0
    lr = 1.0
    
    # Force an error for visualization purposes
    x_mis = np.array([0, 1]) 
    y_mis = 1  # Should be 1, but w=0 predicts 0 (or -1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- BEFORE UPDATE ---
    ax1.scatter(X[y==0, 0], X[y==0, 1], c='blue', s=100, label='Class 0')
    ax1.scatter(X[y==1, 0], X[y==1, 1], c='red', s=100, label='Class 1')
    ax1.scatter(x_mis[0], x_mis[1], s=300, facecolors='none', edgecolors='orange', linewidth=3, label='Error')
    ax1.set_title("Before update (w=[0,0])")
    ax1.set_xlim(-0.5, 1.5); ax1.set_ylim(-0.5, 1.5)
    
    # --- MANUAL UPDATE ---
    error = y_mis - (-1) # y - y_pred
    w_new = w + lr * error * x_mis
    b_new = b + lr * error
    
    # --- AFTER UPDATE ---
    ax2.scatter(X[y==0, 0], X[y==0, 1], c='blue', s=100)
    ax2.scatter(X[y==1, 0], X[y==1, 1], c='red', s=100)
    
    # Draw new boundary
    x1_line = np.linspace(-0.5, 1.5, 100)
    x2_line = -(w_new[0] * x1_line + b_new) / w_new[1]
    ax2.plot(x1_line, x2_line, 'g-', linewidth=2, label='New boundary')
    
    ax2.set_title(f"After update (w={w_new})")
    ax2.set_xlim(-0.5, 1.5); ax2.set_ylim(-0.5, 1.5)
    ax2.legend()
    
    plt.show()

visualize_single_update(X_and, y_and)
```

---

## 📝 Exercise 10.4: Linearly Separable Data (Medium)

```python
# 1. Generate data
np.random.seed(42)
X_class0 = np.random.randn(50, 2) + [-2, -2]
X_class1 = np.random.randn(50, 2) + [2, 2]
X_sep = np.vstack([X_class0, X_class1])
y_sep = np.array([0]*50 + [1]*50)

# 2. Train perceptron
perc_sep = PerceptronLearner(learning_rate=0.1, max_epochs=50)
perc_sep.fit(X_sep, y_sep)

# 3 & 4. Plot and calculate accuracy
plt.figure(figsize=(8, 6))
plt.scatter(X_class0[:, 0], X_class0[:, 1], c='blue', label='Class 0')
plt.scatter(X_class1[:, 0], X_class1[:, 1], c='red', label='Class 1')

x_bounds = np.array([-5, 5])
y_bounds = -(perc_sep.weights[0] * x_bounds + perc_sep.bias) / perc_sep.weights[1]
plt.plot(x_bounds, y_bounds, 'g-', linewidth=3, label='Decision Boundary')

plt.title(f"Accuracy: {perc_sep.score(X_sep, y_sep)*100}% | Converged in {len(perc_sep.errors_history)} epochs")
plt.legend()
plt.show()
```

**6. What happens if you add noise?** By adding points from class 0 into the middle of class 1 (and vice versa), the data is no longer linearly separable. **The perceptron will loop indefinitely (up to `max_epochs`) without ever reaching 0 errors.** The decision boundary (hyperplane) will endlessly oscillate back and forth.

---

## 📝 Exercise 10.5: Convergence Analysis (Hard)

**Theory:** The maximum bound of mistakes/updates is $(R/\gamma)^2$.

1.  **Calculate R (Max distance from origin) for the OR gate:**
    The points are $(0,0), (0,1), (1,0), (1,1)$.
    The farthest point is $(1,1)$.
    $R = \sqrt{1^2 + 1^2} = \sqrt{2} \approx 1.414$
2.  **Estimate $\gamma$ (Margin):**
    Imagine a perfectly symmetric decision boundary: $x_1 + x_2 - 0.5 = 0$.
    The distance from the closest point, $(0,0)$, to this line is $\gamma = \frac{|0 + 0 - 0.5|}{\sqrt{1^2+1^2}} = \frac{0.5}{\sqrt{2}} \approx 0.353$.
3.  **Calculate theoretical bound:**
    $\text{Mistakes} \leq \left(\frac{\sqrt{2}}{0.353}\right)^2 = \left(\frac{1.414}{0.353}\right)^2 \approx 16$
4.  **Comparison:**
    In the manual example from the lesson, the algorithm made around 3 to 5 errors before converging. $5 \leq 16$. The theoretical bound is indeed respected!

---

## 📝 Exercise 10.6: Early Stopping (Hard)

```python
class PerceptronEarlyStopping(PerceptronLearner):
    def fit_early_stopping(self, X, y, patience=10):
        # Initialization similar to parent class
        y_train = np.where(y == 0, -1, y)
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        best_error_count = float('inf')
        patience_counter = 0
        best_weights = None
        best_bias = None
        
        for epoch in range(self.max_epochs):
            errors = 0
            for xi, yi in zip(X, y_train):
                y_pred = np.sign(np.dot(xi, self.weights) + self.bias)
                if y_pred == 0: y_pred = 1
                
                if yi != y_pred:
                    update = self.learning_rate * (yi - y_pred)
                    self.weights += update * xi
                    self.bias += update
                    errors += 1
            
            self.errors_history.append(errors)
            
            # Early Stopping Mechanism
            if errors < best_error_count:
                best_error_count = errors
                best_weights = self.weights.copy()
                best_bias = self.bias
                patience_counter = 0 # Reset patience
            else:
                patience_counter += 1
                
            if errors == 0:
                print(f"Perfect convergence at epoch {epoch+1}")
                break
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best error count: {best_error_count}")
                self.weights = best_weights
                self.bias = best_bias
                break

# Test on noisy data
X_or_noisy = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or_noisy = np.array([1, 1, 1, 1]) # Flipped class of (0,0) to 1 (extreme noise)

es_perc = PerceptronEarlyStopping(max_epochs=1000)
es_perc.fit_early_stopping(X_or_noisy, y_or_noisy, patience=10)
```

**Why is this useful?** It prevents the model from destroying a "good enough" decision boundary while desperately trying to correctly classify a single noisy outlier.

---

## 📝 Exercise 10.7: Perceptron vs Pocket Algorithm (Very Hard)

```python
class PocketPerceptron:
    def __init__(self, learning_rate=0.1, max_epochs=100):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        
    def count_errors(self, X, y_train, w, b):
        predictions = np.sign(np.dot(X, w) + b)
        predictions[predictions == 0] = 1
        return np.sum(predictions != y_train)

    def fit(self, X, y):
        y_train = np.where(y == 0, -1, y)
        w = np.zeros(X.shape[1])
        b = 0.0
        
        best_w = w.copy()
        best_b = b
        best_errors = float('inf')
        
        self.errors_history = []
        
        for epoch in range(self.max_epochs):
            # Standard Perceptron update
            for xi, yi in zip(X, y_train):
                y_pred = np.sign(np.dot(xi, w) + b)
                if y_pred == 0: y_pred = 1
                
                if yi != y_pred:
                    w += self.lr * (yi - y_pred) * xi
                    b += self.lr * (yi - y_pred)
            
            # Evaluate complete model (Pocket mechanism)
            current_errors = self.count_errors(X, y_train, w, b)
            self.errors_history.append(current_errors)
            
            # Is this the best configuration seen so far?
            if current_errors < best_errors:
                best_errors = current_errors
                best_w = w.copy()
                best_b = b
                
        # Keep the best weights found
        self.weights = best_w
        self.bias = best_b
        return self

# --- TEST ON XOR ---
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

pocket = PocketPerceptron(max_epochs=50)
pocket.fit(X_xor, y_xor)

# Calculate Accuracy
y_xor_train = np.where(y_xor == 0, -1, y_xor)
final_errors = pocket.count_errors(X_xor, y_xor_train, pocket.weights, pocket.bias)
accuracy = 100 * (1 - final_errors / len(y_xor))

print(f"Best weights found by Pocket: {pocket.weights}, Bias: {pocket.bias}")
print(f"Maximum accuracy on XOR: {accuracy}%")
```

**Answers:**
3. **Comparison:** A regular perceptron finishes with the weights of the *very last* epoch (which can be terrible on non-separable data due to constant oscillation). The Pocket algorithm returns the *best* boundary found during the entire training process.
4. **Max accuracy on XOR:** $75\%$. It is geometrically impossible to draw a single straight line that correctly separates the 4 points of the XOR problem. The best a linear classifier can do is get 3 out of 4 correct.
5. **Why does Pocket work better on non-separable data?** It acts like a "survival memory". If it luckily stumbles upon a decision boundary that correctly classifies 90% of a noisy dataset, it stores those weights in its "pocket". Subsequent updates will immediately ruin that good boundary as the algorithm tries to fix the remaining 10%, but the optimal version is safely stored away.