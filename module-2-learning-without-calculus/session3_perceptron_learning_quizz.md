

# 🧠 The Perceptron Learning Rule

Test your knowledge on how the perceptron learns! Choose the best answer for each question. *(The answer key is at the bottom of the page).*

### 1. Why do we need Rosenblatt's Perceptron Learning Algorithm?
- [ ] A) Because manual weight tuning is mathematically incorrect.
- [ ] B) Because manual weight tuning does not scale to real-world problems with hundreds or thousands of features.
- [ ] C) Because computers cannot do manual trial-and-error.
- [ ] D) Because it is the only way to solve the XOR problem.

### 2. When does the Perceptron algorithm update its weights?
- [ ] A) After every single data point, regardless of the prediction.
- [ ] B) Only when the prediction matches the target label.
- [ ] C) Only when it makes an incorrect prediction.
- [ ] D) Only at the very end of an epoch.

### 3. What is the correct formula for the weight update rule?
*(Where $\eta$ is the learning rate, $y$ is the actual label, and $\hat{y}$ is the predicted label)*
- [ ] A) $\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot (y + \hat{y}) \cdot \mathbf{x}$
- [ ] B) $\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot (y - \hat{y}) \cdot \mathbf{x}$
- [ ] C) $\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot (y - \hat{y}) \cdot \mathbf{x}$
- [ ] D) $\mathbf{w} \leftarrow \mathbf{w} \cdot \eta + (y - \hat{y})$

### 4. Geometrically, what happens when the perceptron makes a mistake?
- [ ] A) The decision boundary is pushed as far away from the point as possible.
- [ ] B) The data point changes its class label to match the decision boundary.
- [ ] C) The decision boundary is pulled/rotated toward the misclassified point.
- [ ] D) The algorithm adds a second decision boundary.

### 5. What happens if you set the learning rate ($\eta$) too high?
- [ ] A) The algorithm will learn perfectly but take a very long time.
- [ ] B) The updates will be too large, causing the decision boundary to oscillate unstably and potentially never converge.
- [ ] C) The perceptron will automatically transform into a multi-layer neural network.
- [ ] D) The weights will slowly shrink to zero.

### 6. According to the Perceptron Convergence Theorem, when is the algorithm *guaranteed* to find a perfect solution in a finite number of steps?
- [ ] A) When the dataset is small (less than 100 points).
- [ ] B) When the data is linearly separable.
- [ ] C) When the learning rate is exactly 0.1.
- [ ] D) It is always guaranteed to converge perfectly on any dataset.

### 7. Why does the perceptron fail to achieve 100% accuracy on the XOR gate?
- [ ] A) Because XOR is a non-linear problem and cannot be separated by a single straight line.
- [ ] B) Because we didn't train it for enough epochs.
- [ ] C) Because the learning rate is usually set too low for XOR.
- [ ] D) Because XOR has 3 input features instead of 2.

### 8. How does the "Pocket Algorithm" improve upon the standard perceptron when dealing with noisy, non-separable data?
- [ ] A) It mathematically removes the noise from the dataset before training.
- [ ] B) It stops the training process the moment a single error is made.
- [ ] C) It keeps track of and saves the "best" weights seen during training, rather than just returning whatever weights it had at the final epoch.
- [ ] D) It uses a curved decision boundary instead of a straight line.

***
<br><br><br><br><br><br>

## 🔑 Answer Key & Explanations

1. **B** - Manual tuning works for an AND gate (2 inputs), but is impossible for tasks like email spam detection which might rely on 10,000+ word counts (features).
2. **C** - The core philosophy of the perceptron is "learn from mistakes". If $y = \hat{y}$, the error is 0, and the update equation does nothing.
3. **B** - The error is calculated as $(y - \hat{y})$. This error determines both the magnitude and the direction of the update applied to the weights.
4. **C** - If a point is misclassified, the mathematical update effectively pulls the line closer to that point so it might be on the correct side in the next epoch.
5. **B** - A very high learning rate means the line takes massive "jumps" across the graph, overshooting the optimal solution and causing instability.
6. **B** - If you can draw a single straight line to perfectly separate the two classes (linearly separable), Rosenblatt's theorem proves the perceptron will eventually find it. 
7. **A** - If you plot XOR on a 2D graph, you'll see it requires a curved boundary or at least two straight lines to separate the classes. A single perceptron can only draw one straight line.
8. **C** - Because non-separable data causes the standard perceptron to oscillate forever (often ruining good boundaries), the Pocket algorithm acts as a "save state" for the highest accuracy achieved.