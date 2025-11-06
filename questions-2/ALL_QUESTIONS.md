# Deep Learning Questions - Set 2

## Table of Contents
1. [Question 1 — Perceptron weighted sum and activation](#question-1)
2. [Question 2 — List layers in order (Sequential model)](#question-2)
3. [Question 3 — Fill the code and name the activation](#question-3)
4. [Question 4 — Role of gradients during backpropagation](#question-4)
5. [Question 5 — Match ANN terms to definitions](#question-5)
6. [Question 6 — Where do optimizer and loss get specified in Keras?](#question-6)
7. [Question 7 — Fill in backpropagation steps](#question-7)
8. [Question 8 — LSTM: which component decides what to forget?](#question-8)
9. [Question 9 — Convolution with a 3×3 vertical edge kernel](#question-9)
10. [Question 10 — Property of a Feedforward Neural Network (FNN)](#question-10)
11. [Question 11 — 3×3 pooling over the highlighted region](#question-11)
12. [Question 12 — Parameters updated during training](#question-12)
13. [Question 13 — Keras model development lifecycle (fill step 3 and 4)](#question-13)
14. [Question 14 — Regression head activation and loss](#question-14)
15. [Question 15 — Activation that mitigates vanishing gradients](#question-15)

---

<div id="question-1"></div>

![Question 1](https://github.com/ulster-msc/deepl-notes/blob/main/questions-2/Screenshot%202025-11-06%20at%2017.07.09.png?raw=true)

Question 1 — Perceptron weighted sum and activation

- Final answers
  - z term: **+ b** (bias)
  - Activation: **step** (threshold) function

Detailed explanation

**Understanding the Perceptron:**

The perceptron is the fundamental building block of neural networks, invented by Frank Rosenblatt in 1958. It computes a weighted sum of inputs and applies a threshold function.

**Step 1: Fill in the missing term in z**

The formula shows:
```
z = (w₁ × x₁) + (w₂ × x₂) + ... + (wₙ × xₙ) + ___
```

The missing term is the **bias (b)**.

**Complete formula:**
```
z = Σ(wᵢ × xᵢ) + b
  = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

**Why is bias important?**
- Allows the activation function to be shifted left or right
- Without bias, the decision boundary must pass through the origin
- With bias, the model has more flexibility to fit the data
- Similar to the intercept term in linear regression

**Step 2: Identify the activation function**

The question states z is "passed through a(n) ___ function to produce the final output."

For a **simple (classical) Perceptron**, the activation is a **step function** (also called threshold function or Heaviside function):

```
output = step(z) = {
    1  if z ≥ threshold (often 0)
    0  if z < threshold
}
```

**Characteristics of the step function:**
- Binary output: 0 or 1
- Non-differentiable (which limited early perceptrons)
- Creates a hard decision boundary
- Used in the original perceptron for binary classification

**Modern alternatives:**
- Sigmoid: smooth, differentiable, outputs (0, 1)
- ReLU: outputs max(0, x), most common in deep learning
- Tanh: outputs (-1, 1), zero-centered

**Complete perceptron operation:**
1. Compute weighted sum: z = Σ(wᵢxᵢ) + b
2. Apply step function: output = step(z)
3. Result: Binary classification (0 or 1)

References (lectures/practicals used)
- lectures/Lecture 2 - 2025.pdf — p.2 (neuron: linear combination + activation)
- practicals/Pratice -W2 solution.pdf — p.1 (ANN building blocks / activation), p.21 (forward computation with bias)

---

<div id="question-2"></div>

![Question 2](https://github.com/ulster-msc/deepl-notes/blob/main/questions-2/Screenshot%202025-11-06%20at%2017.07.17.png?raw=true)

Question 2 — List layers in order (Sequential model)

- Final answer (from first to last):
  1) Dense, 2) Dropout, 3) Dense, 4) Dense

Explanation
- The code calls `model.add` four times in sequence: a 64‑unit Dense (ReLU), a `Dropout(0.5)`, a 32‑unit Dense (ReLU), and finally a 1‑unit Dense (Sigmoid).

References (lectures/practicals used)
- lectures/Lecture 3-2025.pdf — p.6 (Keras Sequential examples with stacked layers)
- practicals/Practice - W5-r Answers.pdf — p.2–3 (Sequential API usage and layer stacking)

---

<div id="question-3"></div>

![Question 3](https://github.com/ulster-msc/deepl-notes/blob/main/questions-2/Screenshot%202025-11-06%20at%2017.07.25.png?raw=true)

Question 3 — Fill the code and name the activation

- Final answers
  - y = activation(x)
  - Activation function name: Sigmoid

Explanation
- The function body `1 / (1 + np.exp(-x))` is the logistic sigmoid. To create the plotted `y` values we call it on `x` and plot.

References (lectures/practicals used)
- practicals/Pratice -W2 solution.pdf — p.3 (Sigmoid function implementation/plot)
- lectures/Lecture 2 - 2025.pdf — p.2 (activation functions overview including sigmoid)

---

<div id="question-4"></div>

![Question 4](https://github.com/ulster-msc/deepl-notes/blob/main/questions-2/Screenshot%202025-11-06%20at%2017.07.32.png?raw=true)

Question 4 — Role of gradients during backpropagation

- Final answer: B — To determine the direction and magnitude for updating the weights to minimize the loss.

Explanation
- Backpropagation computes the gradient of the loss with respect to each weight. The gradient indicates both the direction and step size (paired with a learning rate/optimizer) for weight updates that reduce the loss.

References (lectures/practicals used)
- lectures/Lecture 4 - 2025.pdf — p.6 (chain rule/gradients used for updating parameters)
- lectures/Lecture 2 - 2025.pdf — p.3 (error/gradient concept in learning)

---

<div id="question-5"></div>

![Question 5](https://github.com/ulster-msc/deepl-notes/blob/main/questions-2/Screenshot%202025-11-06%20at%2017.07.42.png?raw=true)

Question 5 — Match ANN terms to definitions

Detailed explanation

This question tests understanding of fundamental ANN terminology. Let's match each term with its correct definition:

**1. Epoch**
→ **"One full pass over the entire training dataset"**

- During training, data is processed in batches
- After processing all batches once, one epoch is complete
- Example: If you have 1000 samples and batch size=100, one epoch = 10 batches
- Models typically train for multiple epochs (e.g., 100 epochs)
- More epochs ≠ better (can lead to overfitting)

**2. Bias**
→ **"A trainable offset added to the weighted sum before activation"**

- Mathematical form: z = Σ(wᵢxᵢ) + **b**
- The bias 'b' is a learnable parameter (like weights)
- Allows shifting the activation function
- Essential for model flexibility
- Without bias, decision boundaries must pass through origin

**3. Loss Function**
→ **"A measure of prediction error to be minimized (objective)"**

- Quantifies how wrong the model's predictions are
- Examples:
  - **Mean Squared Error (MSE):** for regression
  - **Categorical Crossentropy:** for multi-class classification
  - **Binary Crossentropy:** for binary classification
- Training seeks to minimize this function
- Also called "cost function" or "objective function"

**4. Optimizer**
→ **"The algorithm that updates weights/biases using gradients"**

- Takes gradients from backpropagation and adjusts parameters
- Common optimizers:
  - **SGD (Stochastic Gradient Descent):** basic, uses learning rate
  - **Adam:** adaptive learning rates, momentum, very popular
  - **RMSprop:** adaptive learning rates
- Update rule example (SGD): w_new = w_old - learning_rate × gradient
- Determines how the model learns

**5. Hidden Layer**
→ **"Any layer between the input and output layers"**

- Not directly visible to the input or output
- Learns intermediate representations/features
- Deep networks have multiple hidden layers
- Example: Input(784) → Hidden(128) → Hidden(64) → Output(10)
- More hidden layers = deeper network = can learn more complex patterns

**Summary of matches:**
1. Epoch → One full pass over the entire training dataset
2. Bias → A trainable offset added to the weighted sum before activation
3. Loss Function → A measure of prediction error to be minimized
4. Optimizer → The algorithm that updates weights/biases using gradients
5. Hidden Layer → Any layer between the input and output layers

References (lectures/practicals used)
- lectures/Lecture 3-2025.pdf — p.5–6 (Keras training loop: compile/fit/evaluate concepts)
- practicals/Practice - W5-r Answers.pdf — p.3 (compile and train settings: optimizer/loss)
- practicals/Pratice -W2 solution.pdf — p.1 (ANN components including bias/hidden layer)

---

<div id="question-6"></div>

![Question 6](https://github.com/ulster-msc/deepl-notes/blob/main/questions-2/Screenshot%202025-11-06%20at%2017.07.48.png?raw=true)

Question 6 — Where do optimizer and loss get specified in Keras?

- Final answer: D — model.compile()

Explanation
- `model.compile(optimizer=..., loss=..., metrics=...)` is where the optimizer and loss function are configured before training with `model.fit(...)`.

References (lectures/practicals used)
- lectures/Lecture 3-2025.pdf — p.5–6 (examples showing `compile` with optimizer/loss)
- practicals/Practice - W5-r Answers.pdf — p.3 (complete program with `compile` and `fit`)

---

<div id="question-7"></div>

![Question 7](https://github.com/ulster-msc/deepl-notes/blob/main/questions-2/Screenshot%202025-11-06%20at%2017.07.53.png?raw=true)

Question 7 — Fill in backpropagation steps

- Final answers
  - A: Compute/measure the error (loss) between the predicted and actual output.
  - B: Compute/calculate the gradients of the loss with respect to each weight.
  - C: Update the network weights using those gradients (via the optimizer).

References (lectures/practicals used)
- lectures/Lecture 2 - 2025.pdf — p.3 (learning with error/gradient)
- lectures/Lecture 4 - 2025.pdf — p.6 (backprop/chain rule computing gradients to update weights)

---

<div id="question-8"></div>

![Question 8](https://github.com/ulster-msc/deepl-notes/blob/main/questions-2/Screenshot%202025-11-06%20at%2017.08.00.png?raw=true)

Question 8 — LSTM: which component decides what to forget?

- Final answer: B — The Forget Gate

Explanation
- In an LSTM cell, the forget gate controls how much of the cell state is erased (what information to throw away) at each timestep.

References (lectures/practicals used)
- lectures/Lecture 6 - 2025.pdf — p.2–4 (RNN/LSTM overview)
- practicals/Practice - W6 Answers.pdf — p.4 (RNN building blocks and discussion around gating/activations)

---

<div id="question-9"></div>

![Question 9](https://github.com/ulster-msc/deepl-notes/blob/main/questions-2/Screenshot%202025-11-06%20at%2017.08.09.png?raw=true)

Question 9 — Convolution with a 3×3 vertical edge kernel

- Final answer: **8** (absolute value of convolution result)

Detailed explanation

**Understanding Convolution and Edge Detection:**

Convolution is the fundamental operation in CNNs. The kernel (also called filter) slides over the input, computing element-wise products and summing them. Certain kernels detect specific features like edges.

**Step 1: Extract the 3×3 region**

From the screenshot, the highlighted (yellow) region in rows 1-3, columns 1-3 contains:

```
5  0  8
3  1  6
0  2  2
```

**Step 2: Identify the kernel**

The given kernel is a **vertical edge detector**:

```
 1  0  -1
 1  0  -1
 1  0  -1
```

**Why this detects vertical edges:**
- Left column: +1 (responds to bright pixels on the left)
- Middle column: 0 (ignores center)
- Right column: -1 (responds to dark pixels on the right)
- Result: strong response where intensity changes left → right (vertical edges)

**Step 3: Perform convolution (element-wise multiply and sum)**

Align the kernel with the region and multiply corresponding elements:

```
Region:          Kernel:        Element-wise product:
5  0  8      ×   1  0  -1   =   5×1   0×0   8×(-1)
3  1  6          1  0  -1       3×1   1×0   6×(-1)
0  2  2          1  0  -1       0×1   2×0   2×(-1)
```

**Calculate each product:**
- Row 1: 5×1 = 5,   0×0 = 0,   8×(-1) = -8
- Row 2: 3×1 = 3,   1×0 = 0,   6×(-1) = -6
- Row 3: 0×1 = 0,   2×0 = 0,   2×(-1) = -2

**Sum all products:**
```
Convolution result = 5 + 0 + (-8) + 3 + 0 + (-6) + 0 + 0 + (-2)
                   = 5 - 8 + 3 - 6 - 2
                   = (5 + 3) - (8 + 6 + 2)
                   = 8 - 16
                   = -8
```

**Step 4: Take absolute value**

The question asks for the **absolute value**:
```
|−8| = 8
```

**Final answer: 8**

**Interpretation:**
- The negative result (-8) indicates the region has brighter pixels on the right than the left
- The magnitude (8) indicates a moderate edge strength
- Vertical edge detectors are commonly used in CNNs for feature extraction

References (lectures/practicals used)
- lectures/Lecture 7 - 2025.pdf — p.1–3 (convolution and kernels in CNNs)
- practicals/Practice - W7 Answers-r.pdf — p.2–3 (1D/2D convolution exercises and vertical detector example)

---

<div id="question-10"></div>

![Question 10](https://github.com/ulster-msc/deepl-notes/blob/main/questions-2/Screenshot%202025-11-06%20at%2017.08.15.png?raw=true)

Question 10 — Property of a Feedforward Neural Network (FNN)

- Final answer: C — Information flows in one direction, from input to output, with no cycles.

Explanation
- An FNN does not contain feedback connections; signals pass forward layer‑by‑layer until the output layer.

References (lectures/practicals used)
- lectures/Lecture 3-2025.pdf — p.5 (MLP/Feed‑forward model description)

---

<div id="question-11"></div>

![Question 11](https://github.com/ulster-msc/deepl-notes/blob/main/questions-2/Screenshot%202025-11-06%20at%2017.08.22.png?raw=true)

Question 11 — 3×3 pooling over the highlighted region

Detailed explanation

**Understanding the problem:**

The question provides a 6×6 input feature map and asks us to apply both **Max Pooling** and **Average Pooling** over a 3×3 highlighted region.

**Key note:** The question mentions "stride = 0" which is unusual (stride is typically ≥1). Given that only one highlighted window is shown and single scalar answers are requested, we interpret this as computing pooling for the one shown 3×3 region.

**Step 1: Extract the highlighted 3×3 region**

From the screenshot, the highlighted (yellow/tan) region contains:

```
2  2  9
2  1  3
1  1  3
```

**Step 2: Apply Max Pooling**

**Definition:** Max pooling selects the **maximum value** from the pooling window.

**Process:**
- List all 9 values: {2, 2, 9, 2, 1, 3, 1, 1, 3}
- Identify the maximum: max{2, 2, 9, 2, 1, 3, 1, 1, 3} = **9**

**Max Pooling Result: 9**

**Why max pooling?**
- Preserves the strongest activation/feature
- Provides translation invariance
- Most commonly used in CNNs
- Helps detect whether a feature is present anywhere in the region

**Step 3: Apply Average Pooling**

**Definition:** Average pooling computes the **mean** of all values in the pooling window.

**Process:**
1. Sum all values:
   - Row 1: 2 + 2 + 9 = 13
   - Row 2: 2 + 1 + 3 = 6
   - Row 3: 1 + 1 + 3 = 5
   - **Total sum: 13 + 6 + 5 = 24**

2. Count elements: 3 × 3 = 9

3. Calculate average: 24 / 9 = 2.666...

4. Round to 2 decimal places: **2.67**

**Average Pooling Result: 2.67**

**Why average pooling?**
- Smoother downsampling
- Preserves overall information from the region
- Less commonly used than max pooling in modern CNNs
- Can be useful when all features matter, not just the strongest

**Final Answers:**
- **Max Pooling: 9**
- **Average Pooling: 2.67**

References (lectures/practicals used)
- lectures/Lecture 7 - 2025.pdf — p.3–6 (pooling types and examples)

---

<div id="question-12"></div>

![Question 12](https://github.com/ulster-msc/deepl-notes/blob/main/questions-2/Screenshot%202025-11-06%20at%2017.08.27.png?raw=true)

Question 12 — Parameters updated during training

- Final answers: Weights and Biases

Explanation
- Training adjusts both weights and biases of each layer to minimize the loss; both are learnable parameters.

References (lectures/practicals used)
- lectures/Lecture 2 - 2025.pdf — p.5 (parameters in neurons)
- practicals/Pratice -W2 solution.pdf — p.21 (forward pass includes bias terms)

---

<div id="question-13"></div>

![Question 13](https://github.com/ulster-msc/deepl-notes/blob/main/questions-2/Screenshot%202025-11-06%20at%2017.08.32.png?raw=true)

Question 13 — Keras model development lifecycle (fill step 3 and 4)

- Final answers (in order):
  3) Compile the model
  4) Fit/train the model

Explanation
- After defining the architecture, Keras requires a `compile(...)` step to set the optimizer/loss, then `fit(...)` to train on data. After training, the model can be used to predict on new data.

References (lectures/practicals used)
- lectures/Lecture 3-2025.pdf — p.5–6 (compile and fit steps in Keras)
- practicals/Practice - W5-r Answers.pdf — p.2–3 (complete scripts with `compile` and `fit`)

---

<div id="question-14"></div>

![Question 14](https://github.com/ulster-msc/deepl-notes/blob/main/questions-2/Screenshot%202025-11-06%20at%2017.08.37.png?raw=true)

Question 14 — Regression head activation and loss

- Final answer: True

Explanation
- For standard regression to real‑valued targets, a linear activation is typically used in the output layer so the network can predict any real value, and Mean Squared Error (MSE) (or MAE) is a common loss choice.

References (lectures/practicals used)
- lectures/Lecture 3-2025.pdf — p.5 (examples with regression heads)

---

<div id="question-15"></div>

![Question 15](https://github.com/ulster-msc/deepl-notes/blob/main/questions-2/Screenshot%202025-11-06%20at%2017.08.45.png?raw=true)

Question 15 — Activation that mitigates vanishing gradients

- Final answer: **B) ReLU (Rectified Linear Unit)**

Detailed explanation

**Understanding the Vanishing Gradient Problem:**

The vanishing gradient problem is a critical challenge in training deep neural networks:
- During backpropagation, gradients are multiplied layer by layer
- If gradients are < 1, they exponentially decrease in deeper layers
- Deeper layers receive extremely small gradients → very slow learning or no learning
- This prevented training of deep networks before modern solutions

**Why Sigmoid and Tanh cause vanishing gradients:**

**Sigmoid: σ(x) = 1/(1 + e^(-x))**
- Output range: (0, 1)
- **Problem:** Saturates at both ends
- For large |x|, gradient ≈ 0
- Maximum gradient: 0.25 (at x=0)
- Gradients always < 1, get multiplied through many layers → vanish

**Tanh: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))**
- Output range: (-1, 1)
- **Problem:** Similar saturation issues
- Maximum gradient: 1.0 (at x=0)
- Still suffers from vanishing gradients in deep networks

**Why ReLU solves this:**

**ReLU: f(x) = max(0, x)**

```
f(x) = {
    x   if x > 0
    0   if x ≤ 0
}
```

**Gradient:**
```
f'(x) = {
    1   if x > 0
    0   if x ≤ 0
}
```

**Advantages:**
1. **No saturation for positive inputs:** Gradient = 1 for x > 0
2. **Gradients don't diminish:** Multiplying by 1 preserves gradient magnitude
3. **Faster convergence:** Stronger gradient signals
4. **Computational efficiency:** Simple max operation, no expensive exponentials
5. **Sparse activation:** Only some neurons activate (biological plausibility)

**Comparison table:**

| Activation | Range | Max Gradient | Vanishing Gradient? |
|------------|-------|--------------|---------------------|
| Sigmoid | (0, 1) | 0.25 | Yes ✗ |
| Tanh | (-1, 1) | 1.0 | Yes (in deep networks) ✗ |
| **ReLU** | **[0, ∞)** | **1.0** | **No ✓** |
| Linear | (-∞, ∞) | 1.0 | No, but can't learn nonlinear patterns ✗ |
| Step | {0, 1} | 0 (undefined) | Worst - no gradient ✗ |

**Why other options are wrong:**

- **A) Linear:** Has gradient = 1, but can't learn nonlinear functions (useless for hidden layers)
- **C) Step Function:** Has zero gradient everywhere (except at discontinuity where it's undefined)
- **D) Sigmoid:** Actually causes vanishing gradients!

**Answer: B) ReLU**

**Modern variants of ReLU:**
- **Leaky ReLU:** f(x) = max(0.01x, x) - allows small negative gradients
- **PReLU:** Learnable negative slope
- **ELU:** Smooth for negative values
All aim to further improve upon ReLU's benefits.

References (lectures/practicals used)
- lectures/Lecture 2 - 2025.pdf — p.2 (activation functions including ReLU)
- lectures/Lecture 3-2025.pdf — p.5 (deep MLPs commonly using ReLU)
