# Deep Learning Questions - Set 1

## Table of Contents
1. [Question 1 — One-hot vector size](#question-1)
2. [Question 2 — Activation function and plotting blanks](#question-2)
3. [Question 3 — Keras Functional API model()](#question-3)
4. [Question 4 — Convolution between two same-size matrices (elementwise product then sum)](#question-4)
5. [Question 5 — Compile and fit parameters (Keras Sequential)](#question-5)
6. [Question 6 — OneHotEncoder usage and output](#question-6)
7. [Question 7 — Role of Backpropagation (matches)](#question-7)
8. [Question 8 — Iris MLP with 3 hidden layers, save/load, predict](#question-8)
9. [Question 9 — Two-layer Keras Sequential model (binary task)](#question-9)
10. [Question 10 — 1D Conv layer: define kernel and set weights](#question-10)
11. [Question 11 — Pooling over the highlighted region](#question-11)
12. [Question 12 — MLP output size](#question-12)
13. [Question 13 — Identify layer names](#question-13)
14. [Question 14 — Complete CNN (Keras Functional API)](#question-14)
15. [Question 15 — Compute z and ŷ(z)](#question-15)

---

<div id="question-1"></div>

![Question 1](https://github.com/ulster-msc/deepl-notes/blob/main/questions/Screenshot%202025-11-06%20at%2015.31.18.png?raw=true)

Question 1 — One‑hot vector size

- Final answer: 4

Detailed explanation

**Understanding One-Hot Encoding:**
One-hot encoding is a fundamental technique in Natural Language Processing (NLP) and machine learning for representing categorical data (like words) as numerical vectors. In this representation:
- Each unique category (word) gets its own dimension in the vector space
- Only one dimension has the value 1 (hence "one-hot"), all others are 0
- The position of the 1 identifies which category/word it represents

**Step-by-step solution:**

1. **Count the vocabulary size:**
   The question provides a vocabulary containing these words:
   - "great"
   - "terrible"
   - "fantastic"
   - "boring"

   Total distinct words = 4

2. **Apply the one-hot encoding rule:**
   In one-hot encoding, the dimensionality (length) of each word vector MUST equal the total vocabulary size.
   - Vocabulary size = 4 words
   - Therefore, vector size = 4 dimensions

3. **Why this size?**
   Each word needs its own unique position in the vector. With 4 words, we need exactly 4 positions to uniquely identify each one.

**Example encodings (one possible assignment):**
  - great     → [1, 0, 0, 0]  (position 0 represents "great")
  - terrible  → [0, 1, 0, 0]  (position 1 represents "terrible")
  - fantastic → [0, 0, 1, 0]  (position 2 represents "fantastic")
  - boring    → [0, 0, 0, 1]  (position 3 represents "boring")

**Key properties:**
- Each vector has exactly one 1 and three 0s
- Each word has a unique pattern
- All vectors are orthogonal to each other (no similarity between words is captured)

References (lectures/practicals used)
- lectures/Lecture 2 - 2025.pdf — p.2 (feature representations overview; basis for one‑hot idea)
- practicals/Practice - W3 - Answer.pdf — p.3 (Label encoding for categorical targets prior to model training)

---

<div id="question-2"></div>

![Question 2](https://github.com/ulster-msc/deepl-notes/blob/main/questions/Screenshot%202025-11-06%20at%2015.31.27.png?raw=true)

Question 2 — Activation function and plotting blanks

Filled blanks
- Activation function name: **tanh**
- y = **tanh(x)**
- plt.plot(**x**, **y**)
- plt.ylabel(**'tanh(x)'**)

Detailed explanation

**Understanding the requirements:**
The question asks for an activation function that:
1. Produces an "S-shaped curve" (sigmoid curve)
2. Outputs values between -1 and 1

**Why tanh (hyperbolic tangent)?**

The hyperbolic tangent function perfectly matches both requirements:

1. **Mathematical definition:**
   tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

   Equivalently: tanh(x) = 2·sigmoid(2x) - 1

2. **Output range:**
   - As x → -∞, tanh(x) → -1
   - As x → +∞, tanh(x) → +1
   - tanh(0) = 0
   - Range: (-1, 1), exactly as required

3. **Shape characteristics:**
   - S-shaped (sigmoidal) curve
   - Symmetric around the origin (0, 0)
   - Smooth and differentiable everywhere
   - Steepest gradient at x = 0

**Comparison with other activations:**
- **Sigmoid:** Also S-shaped, but outputs [0, 1] not [-1, 1] ✗
- **ReLU:** Linear for x > 0, not S-shaped ✗
- **Tanh:** S-shaped AND outputs [-1, 1] ✓

**Filling in the code blanks:**

1. **y = tanh(x):** Apply the tanh function to generate y-values
2. **plt.plot(x, y):** Plot x-values against y-values
3. **plt.ylabel('tanh(x)'):** Label the y-axis to indicate we're plotting tanh(x)

Complete runnable snippet
```python
import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

x = np.arange(-5, 5, 0.1)
y = tanh(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.title('Tanh Activation Function')
plt.grid(True)
plt.show()
```

References (lectures/practicals used)
- lectures/Lecture 2 - 2025.pdf — p.2 (activation functions overview)
- lectures/Lecture 6 - 2025.pdf — p.2–4 (activation functions incl. tanh)
- practicals/Practice - W6 Answers.pdf — p.4 (adding tanh activation in RNN exercise)

---

<div id="question-3"></div>

![Question 3](https://github.com/ulster-msc/deepl-notes/blob/main/questions/Screenshot%202025-11-06%20at%2015.31.34.png?raw=true)

Question 3 — Keras Functional API model()

- Final fill: **Model(inputs=x_in, outputs=x_out)**

Detailed explanation

**Understanding the Keras Functional API:**

Unlike the Sequential API (which stacks layers linearly), the Functional API builds models by explicitly defining the computational graph using tensor operations. This requires specifying both the starting point (inputs) and ending point (outputs).

**Step-by-step analysis of the code:**

1. **Input layer creation:**
   ```python
   x_in = Input(shape=(8,))
   ```
   - Creates an input placeholder that expects 8-dimensional feature vectors
   - `x_in` is a tensor object representing the input

2. **First hidden layer:**
   ```python
   x = Dense(10)(x_in)
   ```
   - Creates a Dense layer with 10 neurons
   - The layer is called on `x_in`, connecting it to the input
   - `x` is now a tensor representing the output of this layer

3. **Output layer:**
   ```python
   x_out = Dense(3)(x)
   ```
   - Creates a Dense layer with 3 neurons (likely 3-class classification)
   - The layer is called on `x`, connecting it to the previous layer
   - `x_out` is a tensor representing the final output

4. **Model creation:**
   ```python
   model = Model(inputs=x_in, outputs=x_out)
   ```
   - The `Model` constructor requires two arguments:
     - **inputs**: The starting tensor(s) of the computational graph
     - **outputs**: The ending tensor(s) of the computational graph
   - Keras traces the graph backwards from `x_out` to `x_in` to determine all layers and connections

**Why both inputs and outputs are needed:**
- **inputs=x_in**: Tells Keras where data enters the model
- **outputs=x_out**: Tells Keras what the model should output
- Keras automatically infers all intermediate layers and connections by tracing the tensor graph

**Complete architecture:**
Input(8) → Dense(10) → Dense(3) → Output

References (lectures/practicals used)
- lectures/Lecture 3-2025.pdf — p.3–4 (Keras Functional API usage examples)
- practicals/Practice - W4 - Answers.pdf — p.3 (exercise explicitly using Functional model)

---

<div id="question-4"></div>

![Question 4](https://github.com/ulster-msc/deepl-notes/blob/main/questions/Screenshot%202025-11-06%20at%2015.31.39.png?raw=true)

Question 4 — Convolution between two same‑size matrices (elementwise product then sum)

- Final answer: **29**

Detailed explanation

**Understanding the operation:**

The formula shown in the question describes a convolution between two 2×2 matrices as:
```
[a  b] * [w  x] = aw + bx + ey + fz
[e  f]   [y  z]
```

This is an element-wise multiplication followed by summation (also called the Frobenius inner product or correlation operation).

**Step-by-step calculation:**

Given:
- Matrix A = [[2, 3], [3, 4]]
- Matrix B = [[1, 2], [3, 3]]

**Step 1: Map the values to the formula notation**
```
Matrix A:        Matrix B:
[2  3]          [1  2]
[3  4]          [3  3]

a = 2           w = 1
b = 3           x = 2
e = 3           y = 3
f = 4           z = 3
```

**Step 2: Apply the formula (element-wise multiplication)**
- a × w = 2 × 1 = 2
- b × x = 3 × 2 = 6
- e × y = 3 × 3 = 9
- f × z = 4 × 3 = 12

**Step 3: Sum all products**
Total = 2 + 6 + 9 + 12 = 29

**Verification with visual mapping:**
```
[2  3]  ⊙  [1  2]  =  [2×1  3×2]  =  [2   6]
[3  4]     [3  3]     [3×3  4×3]     [9  12]

Sum of all elements = 2 + 6 + 9 + 12 = 29
```

Where ⊙ denotes element-wise (Hadamard) product.

Python check
```python
import numpy as np
A = np.array([[2, 3],[3, 4]])
B = np.array([[1, 2],[3, 3]])
print(np.sum(A*B))  # 29
```

Final answer
- 29

References (lectures/practicals used)
- lectures/Lecture 7 - 2025.pdf — p.1–3 (convolution/kernels and pooling intro)
- practicals/Practice - W1 Answers-r (1).pdf — p.1 (convolution between two same‑size matrices example and solution)

---

<div id="question-5"></div>

![Question 5](https://github.com/ulster-msc/deepl-notes/blob/main/questions/Screenshot%202025-11-06%20at%2015.31.45.png?raw=true)

Question 5 — Compile and fit parameters (Keras Sequential)

Completed lines
```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy')

model.fit(Train_x, Train_y,
          epochs=100,
          batch_size=32)
```

Detailed explanation

**Understanding the model architecture:**

The code shows a Sequential model with:
- Input layer: Dense(10) with ReLU activation and He normal initialization, accepting n_features inputs
- Hidden layer: Dense(8) with ReLU activation and He normal initialization
- Output layer: Dense(10) with no activation specified

The final Dense(10) layer suggests a **10-class classification problem**.

**Step-by-step reasoning for compile parameters:**

**1. Choosing the optimizer:**
- **Answer: 'adam'**
- Adam (Adaptive Moment Estimation) is the most widely used optimizer
- Advantages over SGD: adaptive learning rates per parameter, momentum, bias correction
- Works well for most problems without extensive hyperparameter tuning
- Other valid options: 'sgd', 'rmsprop', but 'adam' is the modern default

**2. Choosing the loss function:**
- **Answer: 'categorical_crossentropy'**
- The output layer has 10 units (suggesting 10 classes)
- For multi-class classification (>2 classes), we use categorical crossentropy
- **Important distinction:**
  - `categorical_crossentropy`: Use when labels are one-hot encoded (e.g., [0,0,1,0,...])
  - `sparse_categorical_crossentropy`: Use when labels are integers (e.g., 2)
- The question doesn't specify label format, but 'categorical_crossentropy' is the standard answer
- **Note:** The output layer would typically need `activation='softmax'` added to work properly with this loss

**Step-by-step reasoning for fit parameters:**

**3. Choosing epochs:**
- **Answer: 100** (reasonable default)
- One epoch = one complete pass through the entire training dataset
- 100 is a common starting point for small-medium datasets
- Too few: underfitting; too many: overfitting (mitigated by early stopping)

**4. Choosing batch_size:**
- **Answer: 32** (standard default)
- Batch size determines how many samples are processed before updating weights
- 32 is a common default that balances:
  - Memory efficiency (smaller batches use less memory)
  - Training stability (larger batches give more stable gradients)
  - Convergence speed (moderate batch size often works best)
- Other common choices: 16, 64, 128, 256

References (lectures/practicals used)
- lectures/Lecture 3-2025.pdf — p.3, p.5–6 (compile+loss for multi‑class models with softmax/categorical crossentropy)
- practicals/Practice - W5-r Answers.pdf — p.3 (Keras compile for Iris multi‑class example)

---

<div id="question-6"></div>

![Question 6](https://github.com/ulster-msc/deepl-notes/blob/main/questions/Screenshot%202025-11-06%20at%2015.31.50.png?raw=true)

Question 6 — OneHotEncoder usage and output

Detailed explanation

**Understanding OneHotEncoder from sklearn:**

The `OneHotEncoder` class from scikit-learn converts categorical features into a one-hot numeric array. Key properties:
- Input must be 2D: shape (n_samples, n_features)
- By default, categories are sorted **lexicographically (alphabetically)**
- Each unique category becomes its own binary column
- The method `fit_transform()` learns categories and transforms in one step

**Step-by-step solution:**

**Step 1: Prepare the input data**
```python
encoder.fit_transform(['red', 'blue', 'green', 'blue'])
```

The input list has 4 values, but OneHotEncoder expects a 2D array:
```python
# Reshape to (4 samples, 1 feature)
X = np.array(['red', 'blue', 'green', 'blue']).reshape(-1, 1)
```

**Step 2: Understand the category ordering**

The unique values are: {'red', 'blue', 'green'}

OneHotEncoder sorts these **alphabetically**:
1. 'blue'   → column 0
2. 'green'  → column 1
3. 'red'    → column 2

**Step 3: Apply the encoding**

Each sample gets encoded based on which column represents its category:

| Input   | Column 0 (blue) | Column 1 (green) | Column 2 (red) | One-hot vector |
|---------|-----------------|------------------|----------------|----------------|
| 'red'   | 0               | 0                | 1              | [0, 0, 1]      |
| 'blue'  | 1               | 0                | 0              | [1, 0, 0]      |
| 'green' | 0               | 1                | 0              | [0, 1, 0]      |
| 'blue'  | 1               | 0                | 0              | [1, 0, 0]      |

**Final result:**
```python
array([[0., 0., 1.],   # red
       [1., 0., 0.],   # blue
       [0., 1., 0.],   # green
       [1., 0., 0.]])  # blue
```

Code
```python
import numpy as np
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X = np.array(['red', 'blue', 'green', 'blue']).reshape(-1, 1)
result = encoder.fit_transform(X)

print(encoder.categories_)  # [array(['blue', 'green', 'red'], dtype='<U5')]
print(result)
# array([
#   [0., 0., 1.],  # red
#   [1., 0., 0.],  # blue
#   [0., 1., 0.],  # green
#   [1., 0., 0.],  # blue
# ])
```

References (lectures/practicals used)
- practicals/Practice - W3 - Answer.pdf — p.3 (categorical label encoding via LabelEncoder prior to model training)
- lectures/Lecture 3-2025.pdf — p.6 (softmax multi‑class pipeline context; complements one‑hot representation rationale)

---

<div id="question-7"></div>

![Question 7](https://github.com/ulster-msc/deepl-notes/blob/main/questions/Screenshot%202025-11-06%20at%2015.31.56.png?raw=true)

Question 7 — Role of Backpropagation (matches)

Detailed explanation

**Understanding Backpropagation:**

Backpropagation is the cornerstone algorithm for training neural networks. It efficiently computes gradients of the loss function with respect to all network parameters using the chain rule of calculus.

**Matching each component:**

**1. The network's internal parameters (weights and biases):**
→ **"Updated by gradient-based optimization"**

- Weights and biases are the learnable parameters
- After backpropagation computes gradients, an optimizer (like SGD or Adam) uses them to update parameters
- Update rule example: w_new = w_old - learning_rate × gradient
- The goal: adjust parameters to minimize the loss function

**2. The gradients of the loss function:**
→ **"Computed by the backpropagation algorithm"**

- Backpropagation computes ∂Loss/∂w for every weight w
- Uses the chain rule to propagate derivatives backwards through the network
- These gradients tell us how much each parameter contributes to the error
- Direction and magnitude information for parameter updates

**3. The learning signal through the network layers:**
→ **"The error signal propagated backward"**

- The error (loss) starts at the output layer
- Backpropagation passes this error backward, layer by layer
- Each layer receives error signals from layers ahead of it
- This allows all layers (including deep/early ones) to learn from the final prediction error
- Solves the "credit assignment problem" - determining which parameters are responsible for errors

**Why Backpropagation is Essential:**

Without backpropagation, we couldn't efficiently train deep networks because:
- Computing gradients by finite differences would be computationally prohibitive
- Deeper layers wouldn't receive meaningful learning signals
- The chain rule allows us to decompose complex derivatives into manageable pieces

References (lectures/practicals used)
- lectures/Lecture 2 - 2025.pdf — p.3, p.5 (backpropagation and gradient concepts)
- lectures/Lecture 3-2025.pdf — p.2–3 (training loop and gradient‑based updates)
- lectures/Lecture 4 - 2025.pdf — p.6 (role of gradients/chain rule)

---

<div id="question-8"></div>

![Question 8](https://github.com/ulster-msc/deepl-notes/blob/main/questions/Screenshot%202025-11-06%20at%2015.32.16.png?raw=true)

Question 8 — Iris MLP with 3 hidden layers, save/load, predict

Design choices explained
- The task requires three hidden layers: 20 relu → 10 tanh → 8 sigmoid. The final layer is with 'softmax' activation function (think about why?)". Softmax gives a probability distribution over the three classes.
- We encode labels to integers (0,1,2) and use `sparse_categorical_crossentropy`, which matches integer labels and softmax output.
- After training, the model is saved and reloaded to demonstrate persistence before making a prediction on `[5.1, 3.5, 1.4, 0.2]`.

Complete program
```python
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import numpy as np

# load the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
df = read_csv(path, header=None)

# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]

# ensure float data
X = X.astype('float32')

# encode class labels to integers 0..2
y = LabelEncoder().fit_transform(y)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

n_features = X_train.shape[1]

# model: 3 hidden layers (20 relu, 10 tanh, 8 sigmoid) + softmax
model = Sequential()
model.add(Dense(20, activation='relu', input_shape=(n_features,)))
model.add(Dense(10, activation='tanh'))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0)

# save and load
model.save('iris_mlp.h5')
restored = load_model('iris_mlp.h5')

# predict given array [5.1,3.5,1.4,0.2]
sample = np.array([[5.1, 3.5, 1.4, 0.2]], dtype='float32')
yhat = restored.predict(sample, verbose=0)
print('probabilities:', yhat[0])
print('predicted class:', int(argmax(yhat)))
```

References (lectures/practicals used)
- lectures/Lecture 3-2025.pdf — p.5–6 (MLP classification with softmax and compile line)
- practicals/Practice - W4 - Answers.pdf — p.3 (Iris multi‑class classification exercise)

---

<div id="question-9"></div>

![Question 9](https://github.com/ulster-msc/deepl-notes/blob/main/questions/Screenshot%202025-11-06%20at%2015.32.38.png?raw=true)

Question 9 — Two‑layer Keras Sequential model (binary task)

Rationale
- The screenshot asks for a two‑layer network and ends by using `argmax(yhat)` to print a class ID. To keep that output shape consistent with `argmax`, the solution uses a 2‑unit softmax output and trains with `sparse_categorical_crossentropy`. (An equivalent alternative is a 1‑unit sigmoid with a ≥0.5 threshold.)

Complete program
```python
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# load the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv'
df = read_csv(path, header=None)

# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
X = X.astype('float32')
y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
n_features = X_train.shape[1]

# define model: Dense with ReLU, then 2‑unit softmax output
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(n_features,)))
model.add(Dense(2, activation='softmax'))

# compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# fit for 100 epochs, batch size 32
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f'test loss={loss:.4f}, acc={acc:.4f}')

# make a prediction
sample = np.array([[6,148,72,35,0,33.6,0.627,50]], dtype='float32')
yhat = model.predict(sample, verbose=0)
print('Predicted:', yhat, '(class=%d)' % argmax(yhat))
```

References (lectures/practicals used)
- practicals/Practice - W5-r Answers.pdf — p.2–3 (classification compile/predict template used here)
- lectures/Lecture 3-2025.pdf — p.5–6 (classification network patterns; softmax vs sigmoid discussion)

---

<div id="question-10"></div>

![Question 10](https://github.com/ulster-msc/deepl-notes/blob/main/questions/Screenshot%202025-11-06%20at%2015.32.44.png?raw=true)

Question 10 — 1D Conv layer: define kernel and set weights

Key details
- A 1D convolution in Keras expects a 3‑D input with shape `(batch, length, channels)`. The prompt states the shape should be `[1, 10, 1]` (one sequence of length 10, one channel).
- The Conv1D kernel tensor has shape `(kernel_size, in_channels, out_channels)`.
- We set a 3‑tap kernel and a bias of 0.5, then explicitly load them into the layer with `set_weights`.

Code
```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D

# one‑dimensional input values
x = np.array([1,2,3,4,3,2,1,0,1,2], dtype='float32').reshape((1, 10, 1))  # shape [1,10,1]

# 1 filter, kernel size 3
model = Sequential([Conv1D(filters=1, kernel_size=3, input_shape=(10, 1), use_bias=True)])

# kernel shape for Conv1D is (kernel_size, in_channels, out_channels)
kernel = np.asarray([[[0.2]],  # w0
                     [[0.5]],  # w1
                     [[-0.3]]], dtype='float32')  # w2
bias = np.asarray([0.5], dtype='float32')

model.set_weights([kernel, bias])
```

References (lectures/practicals used)
- lectures/Lecture 7 - 2025.pdf — p.5 (1D convolution concept and kernel description)
- practicals/Practice - W7 Answers-r.pdf — p.2–3 (Conv1D example and weight setting)

---

<div id="question-11"></div>

![Question 11](https://github.com/ulster-msc/deepl-notes/blob/main/questions/Screenshot%202025-11-06%20at%2015.32.50.png?raw=true)

Question 11 — Pooling over the highlighted region

Detailed explanation

**Understanding Pooling Operations:**

Pooling is a downsampling technique used in CNNs to:
- Reduce spatial dimensions (width × height)
- Reduce computational cost
- Provide translation invariance
- Extract dominant features from regions

**Step 1: Extract values from the highlighted 3×3 region**

From the screenshot, the yellow/highlighted region contains these nine values:

```
Row 1: 2  2  9
Row 2: 2  1  3
Row 3: 1  1  3
```

**Step 2: Apply Max Pooling**

**Definition:** Max pooling selects the maximum value within the pooling window.

**Calculation:**
- List all values: {2, 2, 9, 2, 1, 3, 1, 1, 3}
- Find the maximum: max{2, 2, 9, 2, 1, 3, 1, 1, 3} = **9**

**Result:** Max Pooling = **9**

**Step 3: Apply Average Pooling**

**Definition:** Average pooling computes the mean of all values in the pooling window.

**Calculation:**
- Sum all values:
  - Row 1: 2 + 2 + 9 = 13
  - Row 2: 2 + 1 + 3 = 6
  - Row 3: 1 + 1 + 3 = 5
  - Total: 13 + 6 + 5 = 24

- Count of elements: 3 × 3 = 9

- Average: 24 ÷ 9 = 2.666...

- Rounded to two decimal places: **2.67**

**Result:** Average Pooling = **2.67**

**Summary of Results:**
- Max Pooling result: **9**
- Average Pooling result: **2.67**

**Key Differences:**
- **Max pooling:** Preserves strongest activations, commonly used in practice
- **Average pooling:** Smoother, preserves overall information from the region

Results
- Max pooling result: 9
- Average pooling result: 2.67

References (lectures/practicals used)
- lectures/Lecture 7 - 2025.pdf — p.1, p.3–6 (pooling layers: max and average, with examples)

---

<div id="question-12"></div>

![Question 12](https://github.com/ulster-msc/deepl-notes/blob/main/questions/Screenshot%202025-11-06%20at%2015.32.58.png?raw=true)

Question 12 — MLP output size

- Final fill: **1**

Detailed explanation

**Understanding the question:**

The code shows:
```python
model.add(Dense(10, input_shape=(8,)))
model.add(Dense(___))  # Fill in the number of neurons
```

The second Dense layer is described as having "a Sigmoid activation function" in the output layer.

**Step-by-step reasoning:**

**Step 1: Identify the activation function**
- The output layer uses **Sigmoid** activation
- Sigmoid formula: σ(x) = 1 / (1 + e^(-x))
- Sigmoid range: (0, 1)
- Sigmoid maps any real number to a value between 0 and 1

**Step 2: Determine the task type**
- Sigmoid activation in the output layer is the standard choice for **binary classification**
- Binary classification: predicting one of two classes (e.g., yes/no, spam/not spam, 0/1)
- The sigmoid output can be interpreted as P(class=1)
- Decision rule: if σ(x) ≥ 0.5 → predict class 1, else → predict class 0

**Step 3: Determine the number of output neurons**
- For binary classification with sigmoid: **1 neuron** is sufficient
- This single neuron outputs a probability between 0 and 1
- Why not 2 neurons? Because P(class=0) = 1 - P(class=1), so the second probability is redundant

**Answer: 1**

**Comparison with other scenarios:**
- **Binary classification (2 classes):** 1 neuron with sigmoid, OR 2 neurons with softmax
- **Multi-class (K > 2 classes):** K neurons with softmax
- **Regression:** 1 or more neurons with linear/no activation

**Complete code:**
```python
model.add(Dense(10, input_shape=(8,)))
model.add(Dense(1, activation='sigmoid'))  # Binary classification
```

References (lectures/practicals used)
- lectures/Lecture 2 - 2025.pdf — p.2 (sigmoid properties)
- lectures/Lecture 4 - 2025.pdf — p.3–4 (binary vs. multi‑class outputs; activations)

---

<div id="question-13"></div>

![Question 13](https://github.com/ulster-msc/deepl-notes/blob/main/questions/Screenshot%202025-11-06%20at%2015.33.03.png?raw=true)

Question 13 — Identify layer names

Detailed explanation

**Understanding CNN Architecture:**

This is a typical Convolutional Neural Network (CNN) for image classification. The code shows the standard pipeline: Convolution → Pooling → Flatten → Dense layers.

**Layer-by-layer identification:**

**1. Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))**
→ **Convolutional Layer (Conv2D)**

- Creates 32 feature maps (filters/kernels)
- Each filter is 3×3 pixels
- Uses ReLU activation
- Input: 28×28 grayscale images (likely MNIST digits)
- Purpose: Extract spatial features like edges, textures

**2. MaxPooling2D((2, 2))**
→ **Max Pooling Layer**

- Pool size: 2×2
- Reduces spatial dimensions by half (28×28 → 14×14 assuming valid padding after conv)
- Keeps the maximum value in each 2×2 region
- Purpose: Downsampling, translation invariance, reduced computation

**3. Flatten()**
→ **Flatten Layer**

- Converts multi-dimensional feature maps into a 1D vector
- Example: 14×14×32 → 6272-dimensional vector
- Purpose: Prepare data for fully connected layers
- No learnable parameters

**4. Dense(64, activation='relu')**
→ **Fully Connected (Dense) Hidden Layer**

- 64 neurons
- ReLU activation
- Fully connected to all neurons from the flattened vector
- Purpose: High-level feature combination and learning

**5. Dense(10, activation='softmax')**
→ **Fully Connected (Dense) Output Layer**

- 10 neurons (for 10 classes, e.g., digits 0-9)
- Softmax activation produces probability distribution
- Sum of outputs = 1.0
- Purpose: Final classification

**Complete architecture summary:**
Input (28×28×1) → Conv2D → MaxPooling → Flatten → Dense(64) → Dense(10) → Output probabilities

References (lectures/practicals used)
- lectures/Lecture 7 - 2025.pdf — p.1–4 (CNN pipeline: Conv2D → MaxPooling2D → Flatten → Dense)

---

<div id="question-14"></div>

![Question 14](https://github.com/ulster-msc/deepl-notes/blob/main/questions/Screenshot%202025-11-06%20at%2015.33.15.png?raw=true)

Question 14 — Complete CNN (Keras Functional API)

Design rationale and constraints satisfied
- Input: RGB images of size 128×128 → tensor shape `(128, 128, 3)` for the visible input.
- Three convolution + pooling stages act as feature extractors. We insert Batch Normalization to stabilize training and Dropout for regularization as instructed.
- After feature extraction, we flatten and use two fully connected layers with Dropout in between.
- Output: 4‑class prediction with softmax activation.

Completed code
```python
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

visible = Input(shape=(128, 128, 3))

conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(visible)
bn1 = BatchNormalization()(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)
drop1 = Dropout(0.25)(pool1)

conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(drop1)
bn2 = BatchNormalization()(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)
drop2 = Dropout(0.25)(pool2)

conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(drop2)
bn3 = BatchNormalization()(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)
drop3 = Dropout(0.5)(pool3)

flat1 = Flatten()(drop3)
hidden1 = Dense(512, activation='relu')(flat1)
drop4 = Dropout(0.5)(hidden1)
hidden2 = Dense(256, activation='relu')(drop4)

output = Dense(4, activation='softmax')(hidden2)

model = Model(inputs=visible, outputs=output)
```

References (lectures/practicals used)
- lectures/Lecture 7 - 2025.pdf — p.1–4 (CNN blocks, BatchNormalization and Dropout usage)

---

<div id="question-15"></div>

![Question 15](https://github.com/ulster-msc/deepl-notes/blob/main/questions/Screenshot%202025-11-06%20at%2015.33.23.png?raw=true)

Question 15 — Compute z and ŷ(z)

Detailed explanation

**Understanding the Neural Network Unit:**

The diagram shows a simple perceptron with:
- Input vector: x = [x₁, x₂, x₃, x₄]
- Weight vector: w = [w₁, w₂, w₃, w₄]
- Bias: b₀ with its own weight w₀
- Pre-activation sum: z
- Activation function: ŷ (appears to be sigmoid based on context)

**Given values:**
- b₀ = 0.5 (bias value)
- w₀ = 1 (bias weight)
- x = [2, 1, 0, 1]ᵀ (input vector)
- w = [0.2, -0.1, 0.3, 0.4]ᵀ (weight vector)

**Step 1: Compute the weighted sum z**

The pre-activation value z is computed as:
```
z = (bias term) + (weighted sum of inputs)
z = b₀ × w₀ + Σ(xᵢ × wᵢ)
```

**Calculate each component:**

**Bias term:**
- b₀ × w₀ = 0.5 × 1 = 0.5

**Weighted inputs (element-wise multiplication):**
- x₁ × w₁ = 2 × 0.2 = 0.4
- x₂ × w₂ = 1 × (-0.1) = -0.1
- x₃ × w₃ = 0 × 0.3 = 0.0
- x₄ × w₄ = 1 × 0.4 = 0.4

**Sum the weighted inputs:**
- Σ(xᵢ × wᵢ) = 0.4 + (-0.1) + 0.0 + 0.4 = 0.4 - 0.1 + 0.4 = 0.7

**Total z:**
- z = 0.5 + 0.7 = **1.2**

**Step 2: Apply activation function ŷ = σ(z)**

The activation function is sigmoid:
```
σ(z) = 1 / (1 + e^(-z))
```

**Calculate σ(1.2):**
- e^(-1.2) = e^(-1.2) ≈ 0.30119
- 1 + e^(-1.2) ≈ 1 + 0.30119 = 1.30119
- σ(1.2) = 1 / 1.30119 ≈ 0.7685

Therefore: ŷ ≈ **0.7685** (or **0.77** rounded to 2 decimal places)

**Final answers:**
- **z = 1.2**
- **ŷ ≈ 0.7685**

References (lectures/practicals used)
- practicals/Pratice -W2 solution.pdf — p.1, p.3, p.9–10 (sigmoid function and perceptron forward pass)
- lectures/Lecture 2 - 2025.pdf — p.2 (sigmoid curve), p.5 (linear combination + activation context)
