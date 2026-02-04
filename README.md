# Machine Learning Fundamentals: From Perceptron to CNNs

## Course Overview

**Target Audience:** Engineering students  
**Total Duration:** 24 hours (12 sessions × 2 hours)  
**Prerequisites:** Basic Python, linear algebra fundamentals

This course teaches the fundamentals of machine learning, progressing from the perceptron to convolutional neural networks. The structure emphasizes intuitive understanding before mathematical rigor, ensuring students grasp core concepts effectively before moving to implementation.

---

## Course Structure

| Module | Sessions | Hours | Focus |
|--------|----------|-------|-------|
| 1. Foundations | 1-2 | 4h | Python, Perceptron basics |
| 2. Learning Without Calculus | 3-4 | 4h | Learning rules, MLPs (intuitive) |
| 3. Mathematics of Learning | 5-7 | 6h | Loss, gradients, backpropagation |
| 4. Generalization & Practice | 8-9 | 4h | Regularization, PyTorch |
| 5. Convolutional Networks | 10-11 | 4h | Convolution, CNNs |
| 6. Integration | 12 | 2h | Final project |

---

## Module 1: Foundations (4 hours)

### Session 1 (2h): Python Basics for ML ✅

**Learning Objectives:**
- Write Python functions and use control structures
- Manipulate NumPy arrays efficiently
- Perform vector and matrix operations

**Content:**
1. Python refresher: variables, functions, loops
2. NumPy fundamentals
   - Array creation and indexing
   - Broadcasting rules
3. Vectors and matrices
   - Dot product, matrix multiplication
   - Practical exercises with data manipulation

**Practical Work:**
- Implement basic vector operations from scratch
- Compare loop-based vs vectorized implementations

---

### Session 2 (2h): Perceptron & Linear Classification ✅

**Learning Objectives:**
- Explain how a perceptron computes its output
- Implement activation functions
- Visualize and interpret decision boundaries

**Content:**
1. The perceptron model
   - Biological inspiration (brief)
   - Weights, bias, weighted sum
2. Activation functions
   - Step function, sigmoid, ReLU
   - Manual calculations
3. Geometric interpretation
   - Hyperplanes and decision boundaries
   - Linear separability

**Practical Work:**
- Manual perceptron calculations
- Visualize decision boundaries for 2D data
- Experiment with different weight values

---

## Module 2: Learning Without Calculus (4 hours)

### Session 3 (2h): Perceptron Learning Rule

**Learning Objectives:**
- Explain why automatic learning is necessary
- Implement Rosenblatt's perceptron learning rule
- Analyze convergence behavior on linearly separable data

**Content:**
1. Motivation: why not set weights manually?
   - Scalability problem
   - The learning paradigm
2. Rosenblatt's update rule
   - Intuitive derivation (no calculus required)
   - Update rule: `w ← w + η(y - ŷ)x`
3. Convergence theorem (intuitive)
   - When does it work?
   - Limitations: linear separability requirement

**Practical Work:**
- Implement perceptron learning from scratch
- Visualize weight updates and decision boundary evolution
- Test on separable vs non-separable datasets

---

### Session 4 (2h): Multi-Layer Networks (Intuitive)

**Learning Objectives:**
- Explain why single perceptrons cannot solve XOR
- Compute forward propagation through a multi-layer network
- Motivate the need for systematic training methods

**Content:**
1. The XOR problem
   - Demonstration of linear inseparability
   - Solution: hidden layers
2. Multi-layer perceptron architecture
   - Hidden layers and non-linearity
   - Universal approximation (intuitive)
3. Forward propagation
   - Computing outputs layer by layer
   - Manual calculations
4. The training problem
   - Manual weight adjustment experiments
   - Why we need gradients → motivation for Module 3

**Practical Work:**
- Solve XOR with a 2-layer network (given weights)
- Experiment with manual weight tuning
- Visualize hidden layer representations

---

## Module 3: The Mathematics of Learning (6 hours)

### Session 5 (2h): Loss Functions & Gradient Descent

**Learning Objectives:**
- Define and compute common loss functions
- Explain gradient descent geometrically and mathematically
- Implement basic SGD for linear regression

**Content:**
1. Loss functions
   - Mean Squared Error (MSE)
   - Loss as "how wrong we are"
   - Loss landscapes visualization
2. Derivatives and gradients
   - Review: derivative as slope
   - Partial derivatives and gradients
3. Gradient descent
   - The optimization idea
   - Learning rate and its effects
   - Stochastic Gradient Descent (SGD)
4. Linear regression example
   - Closed-form vs gradient descent
   - Implementation and comparison

**Practical Work:**
- Implement gradient descent for linear regression
- Visualize loss landscape and optimization trajectory
- Experiment with learning rates

---

### Session 6 (2h): Backpropagation

**Learning Objectives:**
- Apply the chain rule to compute gradients in a network
- Implement backpropagation for a 2-layer MLP
- Train an MLP on a simple dataset

**Content:**
1. The chain rule
   - Mathematical formulation
   - Computational graphs
2. Backpropagation algorithm
   - Forward pass: compute and store activations
   - Backward pass: propagate gradients
   - Weight updates
3. Implementation considerations
   - Numerical gradient checking
   - Common pitfalls

**Practical Work:**
- Implement backpropagation from scratch
- Train MLP on XOR (finally!)
- Gradient checking implementation

---

### Session 7 (2h): Logistic Regression & Softmax

**Learning Objectives:**
- Implement binary classification with logistic regression
- Extend to multi-class classification with softmax
- Use cross-entropy loss appropriately

**Content:**
1. Binary classification
   - Sigmoid output interpretation as probability
   - Binary cross-entropy loss
   - Decision threshold
2. Multi-class classification
   - Softmax function
   - Categorical cross-entropy
3. Complete classification pipeline
   - From raw data to predictions
   - Evaluation metrics: accuracy, confusion matrix

**Practical Work:**
- Implement logistic regression from scratch
- Multi-class classifier on a toy dataset
- Visualize decision regions

---

## Module 4: Generalization & Practice (4 hours)

### Session 8 (2h): Generalization & Regularization

**Learning Objectives:**
- Design proper train/validation/test splits
- Diagnose overfitting and underfitting
- Apply regularization techniques

**Content:**
1. The generalization problem
   - Training vs test performance
   - Bias-variance tradeoff (intuitive)
2. Data splits
   - Train/validation/test methodology
   - Cross-validation
3. Overfitting detection
   - Learning curves
   - Early stopping
4. Regularization techniques
   - L1 and L2 regularization
   - Dropout (intuitive)
5. Modern optimizers (brief overview)
   - Momentum: "SGD with inertia"
   - Adam: adaptive learning rates
   - When to use what (practical guidelines)

**Practical Work:**
- Diagnose overfitting on a real dataset
- Implement and compare regularization methods
- Learning curves visualization

---

### Session 9 (2h): PyTorch Introduction

**Learning Objectives:**
- Use PyTorch tensors and automatic differentiation
- Build neural networks with nn.Module
- Train models using PyTorch's training loop

**Content:**
1. PyTorch fundamentals
   - Tensors: creation, operations, GPU
   - Autograd: automatic differentiation
2. Building networks
   - nn.Module and nn.Linear
   - Activation functions in PyTorch
   - Sequential vs custom modules
3. Training loop
   - Optimizers (SGD, Adam)
   - Loss functions
   - The training loop pattern
4. Rebuilding previous models
   - Perceptron in PyTorch
   - MLP in PyTorch

**Practical Work:**
- Reimplement MLP using PyTorch
- Compare with from-scratch implementation
- Train on a larger dataset (e.g., MNIST subset)

---

## Module 5: Convolutional Networks (4 hours)

### Session 10 (2h): The Convolution Operation

**Learning Objectives:**
- Explain why MLPs are inefficient for image data
- Compute convolutions by hand
- Describe how kernels act as feature detectors

**Content:**
1. Motivation: images and MLPs
   - The curse of dimensionality
   - Locality and translation invariance
   - Why fully-connected layers fail
2. The convolution operation
   - Kernels/filters as sliding windows
   - Mathematical definition
   - Hand calculations on small examples
3. Convolution as a neural network layer
   - Shared weights interpretation
   - Link to what students know: "a special linear layer"
   - Parameter efficiency
4. Convolution parameters
   - Padding: preserving spatial dimensions
   - Stride: controlling output size
   - Output size formula
5. Feature detection intuition
   - Edge detectors (Sobel, etc.)
   - Learning filters from data

**Practical Work:**
- Hand-calculate convolutions on 5×5 images
- Implement convolution from scratch (NumPy)
- Visualize effect of different kernels (edges, blur, sharpen)
- Verify with PyTorch nn.Conv2d

---

### Session 11 (2h): Building CNNs

**Learning Objectives:**
- Design a CNN architecture for image classification
- Implement and train a CNN in PyTorch
- Visualize and interpret learned features

**Content:**
1. Pooling layers
   - Max pooling and average pooling
   - Spatial dimension reduction
   - Translation invariance
2. CNN architecture patterns
   - Conv → ReLU → Pool pattern
   - Feature maps and channels
   - Flattening for classification
3. Classic architectures
   - LeNet-5: the original CNN
   - Intuition about depth
4. Training CNNs
   - Data augmentation for images
   - Practical considerations
5. Understanding what CNNs learn
   - Feature map visualization
   - Filter visualization

**Practical Work:**
- Implement LeNet-5 in PyTorch
- Train on MNIST (full dataset)
- Visualize feature maps at different layers
- Experiment with architecture variations

---

## Module 6: Integration (2 hours)

### Session 12 (2h): Final Project & Best Practices

**Learning Objectives:**
- Design and implement an end-to-end ML pipeline
- Apply professional ML practices
- Present and defend technical work

**Content:**
1. End-to-end pipeline review
   - Data loading and preprocessing
   - Model selection and architecture
   - Training, validation, testing
   - Results analysis
2. Professional practices
   - Code organization
   - Experiment tracking
   - Reproducibility
   - Documentation
3. Project presentations
   - Student presentations
   - Peer feedback
   - Discussion

**Final Project Options:**
- Image classification on a new dataset
- Comparison study: MLP vs CNN
- Custom architecture design challenge


---

## Progression Map

```
Module 1          Module 2              Module 3                Module 4           Module 5
────────          ────────              ────────                ────────           ────────
                                        
Perceptron   →   Learning Rule    →    Gradient Descent   →   PyTorch      →    Convolution
    │                 │                      │                    │                   │
    ▼                 ▼                      ▼                    ▼                   ▼
"computes"       "updates"             "optimizes"           "automates"        "specializes"
                                        
Linear       →   Multi-layer      →    Backpropagation    →   Practice     →    CNNs
Classification        MLP                    │                    │                   │
                      │                      ▼                    ▼                   ▼
                      ▼              Classification          Genertic          Feature
                   "deep"              pipeline              training          detection
```

---

## Resources

### Required
- Course Jupyter notebooks (provided)
- PyTorch documentation: https://pytorch.org/docs/

### Recommended
- "Deep Learning" by Goodfellow, Bengio, Courville (Chapters 6, 9)
- 3Blue1Brown: Neural Networks series (YouTube)
- CS231n course notes (Stanford)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2026.1 | 2026 | Added convolution module, streamlined optimization |
| 2025.0 | 2025 | Initial version |