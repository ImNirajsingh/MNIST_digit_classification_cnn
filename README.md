# 🧠 MNIST Digit Classification Using CNN

A Deep Learning project for handwritten digit classification using the **MNIST dataset** and a custom **Convolutional Neural Network (CNN)** implemented with **PyTorch**.

The model is trained to recognize handwritten digits from:

```text
0 → 9
```

The trained model achieved:

> **98.65% test accuracy after only 5 epochs**

The project also demonstrates GPU acceleration using CUDA, model training, evaluation, model saving, and visualization of a prediction.

---

# 📌 Project Overview

Handwritten digit recognition is one of the classic introductory Computer Vision problems.

In this project, a CNN is trained on the **MNIST handwritten digit dataset**.

The complete workflow is:

```text
MNIST Dataset
      ↓
Image Transformation
      ↓
DataLoader
      ↓
CNN Architecture
      ↓
Forward Pass
      ↓
Cross-Entropy Loss
      ↓
Backpropagation
      ↓
Adam Optimizer
      ↓
5 Training Epochs
      ↓
Test Evaluation
      ↓
98.65% Accuracy
      ↓
Save Trained Model
      ↓
Visualize Prediction
```

---

# 🎯 Objective

The objective is to build a neural network that takes a handwritten digit image as input and predicts which digit it represents.

For example:

```text
Input Image

┌───────────────────┐
│                   │
│       ███         │
│         ██        │
│          █        │
│          █        │
│         ██        │
│        ██         │
│       ██          │
│                   │
└───────────────────┘

        ↓

       CNN

        ↓

Predicted Digit: 7
```

---

# 📊 Dataset

The project uses the:

> **MNIST Handwritten Digit Dataset**

MNIST contains grayscale images of handwritten digits from `0` to `9`.

Each image has a resolution of:

```text
28 × 28 pixels
```

Each image contains:

```text
28 × 28 = 784 pixels
```

The dataset is divided into:

```text
Training Set → 60,000 images
Test Set     → 10,000 images
```

---

# 🖼️ Input Image Format

Each MNIST image is:

```text
Height = 28
Width  = 28
Channels = 1
```

Therefore, the PyTorch input shape is:

```text
[Batch Size, 1, 28, 28]
```

For example, with a batch size of 64:

```text
[64, 1, 28, 28]
```

---

# 🛠️ Technologies Used

## Programming Language

```text
Python
```

## Deep Learning Framework

```text
PyTorch
```

## Computer Vision

```text
Torchvision
```

## Visualization

```text
Matplotlib
```

## Hardware Acceleration

```text
CUDA / NVIDIA GPU
```

---

# 📦 Libraries Used

The project imports:

```python
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets import MNIST
from torchvision import transforms

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
```

---

# 🔍 Library Explanation

## PyTorch

```python
import torch
```

PyTorch is the main Deep Learning framework used to:

- Create tensors
- Perform mathematical operations
- Run neural networks
- Calculate gradients
- Train the model
- Use GPU acceleration

---

## PyTorch Neural Network Module

```python
import torch.nn as nn
```

`torch.nn` provides building blocks for neural networks.

Examples used in this project:

```text
nn.Module
nn.Conv2d
nn.ReLU
nn.MaxPool2d
nn.Linear
nn.CrossEntropyLoss
```

---

## PyTorch Optimizer

```python
import torch.optim as optim
```

This module provides optimization algorithms.

The project uses:

```text
Adam
```

---

## Torchvision

```python
from torchvision.datasets import MNIST
from torchvision import transforms
```

Torchvision provides:

- Computer Vision datasets
- Image transformations
- Computer Vision utilities

The MNIST dataset is loaded directly through torchvision.

---

## DataLoader

```python
from torch.utils.data import DataLoader
```

`DataLoader` creates batches of data and handles iteration over the dataset.

Instead of training on all 60,000 images simultaneously, the model processes:

```text
64 images per batch
```

---

## Matplotlib

```python
import matplotlib.pyplot as plt
```

Matplotlib is used to display an MNIST image and its predicted label.

---

# ⚙️ Device Selection

The project automatically checks whether CUDA is available:

```python
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

print("Using device : ", device)
```

The output from this project was:

```text
Using device : cuda
```

This means the model was trained using an NVIDIA CUDA-compatible GPU.

---

# 🚀 CPU vs GPU

PyTorch can execute tensor operations on:

```text
CPU
```

or:

```text
GPU
```

For Deep Learning workloads, GPUs can significantly accelerate:

- Matrix multiplication
- Convolution
- Neural network training
- Backpropagation

The code automatically selects the best available device:

```text
CUDA available
      ↓
Use GPU

CUDA unavailable
      ↓
Use CPU
```

This makes the notebook portable between systems.

---

# 🔄 Image Transformation

The project uses:

```python
transform = transforms.ToTensor()
```

`ToTensor()` converts the image into a PyTorch tensor.

It also scales standard image pixel values from approximately:

```text
0 – 255
```

to:

```text
0.0 – 1.0
```

Therefore, instead of:

```text
Pixel = 255
```

the tensor contains approximately:

```text
Pixel = 1.0
```

This normalization makes the input more suitable for neural network training.

---

# 📥 Loading MNIST

Training dataset:

```python
train_dataset = MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)
```

Testing dataset:

```python
test_dataset = MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)
```

The parameters mean:

```text
root='./data'
```

stores the dataset inside the `data` directory.

```text
train=True
```

loads the training dataset.

```text
train=False
```

loads the test dataset.

```text
download=True
```

automatically downloads the dataset if it is not already available.

---

# 📦 Creating DataLoaders

Training DataLoader:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)
```

Testing DataLoader:

```python
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)
```

---

# 🔀 Why Shuffle Training Data?

The training loader uses:

```python
shuffle=True
```

This randomly changes the order of training examples between epochs.

This helps prevent the model from learning unwanted patterns from the ordering of the dataset.

For example:

```text
Epoch 1:
Image A → B → C → D

Epoch 2:
Image C → A → D → B
```

The test loader uses:

```python
shuffle=False
```

because the order of test samples does not need to be randomized for evaluation.

---

# 🧠 CNN Architecture

The model is defined as:

```python
class simpleCNN(nn.Module):
    def __init__(self):
        super(simpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

---

# 🏗️ Architecture Overview

The network can be represented as:

```text
Input Image
    │
    │
    ▼
28 × 28 × 1
    │
    ▼
Conv2D
1 → 8 channels
3 × 3 kernel
    │
    ▼
ReLU
    │
    ▼
MaxPool
2 × 2
    │
    ▼
14 × 14 × 8
    │
    ▼
Conv2D
8 → 16 channels
3 × 3 kernel
    │
    ▼
ReLU
    │
    ▼
MaxPool
2 × 2
    │
    ▼
7 × 7 × 16
    │
    ▼
Flatten
    │
    ▼
784 features
    │
    ▼
Linear
784 → 64
    │
    ▼
ReLU
    │
    ▼
Linear
64 → 10
    │
    ▼
10 Output Logits
    │
    ▼
Predicted Digit
0 – 9
```

---

# 🔬 Layer-by-Layer Explanation

## 1. First Convolution Layer

```python
nn.Conv2d(
    1,
    8,
    kernel_size=3,
    padding=1
)
```

Parameters:

```text
Input Channels  = 1
Output Channels = 8
Kernel Size     = 3 × 3
Padding         = 1
```

The input MNIST image contains one grayscale channel:

```text
1 × 28 × 28
```

The convolution creates 8 feature maps:

```text
8 × 28 × 28
```

Because:

```text
kernel_size = 3
padding = 1
```

the spatial dimensions remain:

```text
28 × 28
```

---

# 🧩 What Does Convolution Do?

A convolution applies learnable filters to the image.

For example, a filter may gradually learn to detect:

```text
Edges
Corners
Curves
Lines
Digit strokes
```

The network learns these filters automatically during training.

---

# ⚡ 2. ReLU Activation

```python
nn.ReLU(inplace=True)
```

ReLU means:

> Rectified Linear Unit

The mathematical function is:

```text
ReLU(x) = max(0, x)
```

Therefore:

```text
Negative value → 0
Positive value → unchanged
```

Example:

```text
Input:
[-3, -1, 0, 2, 5]

ReLU:
[ 0,  0, 0, 2, 5]
```

ReLU introduces non-linearity into the network.

---

# 🧊 3. Max Pooling

```python
nn.MaxPool2d(2)
```

This performs 2 × 2 max pooling.

It reduces the spatial dimensions by a factor of 2.

Before pooling:

```text
8 × 28 × 28
```

After pooling:

```text
8 × 14 × 14
```

Max pooling keeps the maximum value from each 2 × 2 region.

It helps:

- Reduce computation
- Reduce spatial dimensions
- Preserve strong features
- Provide some translation robustness

---

# 🔬 Second Convolution Layer

```python
nn.Conv2d(
    8,
    16,
    kernel_size=3,
    padding=1
)
```

The second convolution receives:

```text
8 feature maps
```

and produces:

```text
16 feature maps
```

Before the second convolution:

```text
8 × 14 × 14
```

After convolution:

```text
16 × 14 × 14
```

---

# ⚡ Second ReLU

```python
nn.ReLU(inplace=True)
```

Again, ReLU introduces non-linearity.

The network can now learn more complex patterns from the features produced by the first convolution.

---

# 🧊 Second Max Pooling

```python
nn.MaxPool2d(2)
```

The spatial dimensions are reduced:

```text
16 × 14 × 14
```

to:

```text
16 × 7 × 7
```

---

# 🔄 Flattening

After the convolutional feature extractor:

```text
16 × 7 × 7
```

The model uses:

```python
x = x.view(x.size(0), -1)
```

This converts the feature maps into a one-dimensional vector.

Calculation:

```text
16 × 7 × 7 = 784
```

Therefore:

```text
16 × 7 × 7
```

becomes:

```text
784
```

features per image.

For a batch of 64 images:

```text
[64, 16, 7, 7]
```

becomes:

```text
[64, 784]
```

---

# 🧠 Fully Connected Classifier

The classifier is:

```python
self.classifier = nn.Sequential(
    nn.Linear(16 * 7 * 7, 64),
    nn.ReLU(inplace=True),
    nn.Linear(64, 10)
)
```

---

# 🔢 First Linear Layer

```python
nn.Linear(16 * 7 * 7, 64)
```

Since:

```text
16 × 7 × 7 = 784
```

the layer is:

```text
784 → 64
```

It takes 784 extracted features and learns how they contribute to digit classification.

---

# ⚡ Classifier ReLU

```python
nn.ReLU(inplace=True)
```

This introduces another non-linear transformation.

---

# 🎯 Output Layer

```python
nn.Linear(64, 10)
```

The final layer contains:

```text
10 output neurons
```

because MNIST has 10 possible classes:

```text
0
1
2
3
4
5
6
7
8
9
```

The output is therefore:

```text
10 logits
```

---

# 📊 Logits

The final layer does not directly output probabilities.

For example, it may produce:

```text
[-2.1, -1.4, 0.5, -3.0, 1.2, -2.5, -1.0, 8.7, -2.2, -1.8]
```

The largest value is:

```text
8.7
```

at index:

```text
7
```

Therefore:

```text
Predicted Digit = 7
```

---

# 🎯 Prediction

The project uses:

```python
predictions = outputs.argmax(dim=1)
```

`argmax()` returns the index of the largest logit.

For example:

```text
Output:

Digit:       0    1    2    3    4    5    6    7    8    9
Logit:      -2.1 -1.4 0.5 -3.0 1.2 -2.5 -1.0 8.7 -2.2 -1.8

                                      ↑
                                   Maximum

Prediction = 7
```

---

# 📉 Loss Function

The project uses:

```python
loss = nn.CrossEntropyLoss()
```

Cross-Entropy Loss is commonly used for multi-class classification.

The model outputs:

```text
10 logits
```

and the target is:

```text
One integer class label
```

For example:

```text
Output → 10 logits
Label  → 7
```

---

# 🧮 Cross-Entropy Concept

Conceptually, Cross-Entropy Loss encourages the model to assign a high probability to the correct class.

For one sample:

```text
Loss = -log(P(correct class))
```

If the model gives the correct class a high probability:

```text
P(correct) → 1
```

then:

```text
Loss → 0
```

If the model gives the correct class a low probability:

```text
P(correct) → 0
```

then:

```text
Loss → large
```

Therefore:

```text
Better Prediction
      ↓
Lower Loss
```

---

# ⚙️ Optimizer

The project uses:

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

The optimizer is:

> Adam — Adaptive Moment Estimation

Learning rate:

```text
0.001
```

---

# 🧠 What Does the Optimizer Do?

During training:

```text
Prediction
    ↓
Loss Calculation
    ↓
Backpropagation
    ↓
Gradients
    ↓
Adam Optimizer
    ↓
Update Model Parameters
```

The optimizer changes the model's weights to reduce the loss.

---

# 🔄 Training Process

The model is trained for:

```python
epochs = 5
```

Each epoch processes the entire training dataset once.

Since the training dataset contains approximately:

```text
60,000 images
```

and batch size is:

```text
64
```

the model performs roughly:

```text
60,000 / 64
≈ 938 batches
```

per epoch.

Across 5 epochs, this is roughly:

```text
938 × 5
≈ 4,690 training batches
```

---

# 🏋️ Training Loop

The training loop is:

```python
for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss_value = loss(outputs, labels)

        loss_value.backward()

        optimizer.step()

        total_loss += loss_value.item()
```

---

# 🔍 Training Loop Explained

## Step 1 — Training Mode

```python
model.train()
```

This tells PyTorch that the model is being trained.

This becomes particularly important for layers such as:

```text
Dropout
BatchNorm
```

This particular CNN does not contain those layers, but using `model.train()` is still the correct training practice.

---

# Step 2 — Move Data to Device

```python
images, labels = images.to(device), labels.to(device)
```

If CUDA is available:

```text
Images → GPU
Labels → GPU
```

This ensures the model and data are operating on the same device.

---

# Step 3 — Clear Gradients

```python
optimizer.zero_grad()
```

PyTorch accumulates gradients by default.

Therefore, gradients from the previous batch need to be cleared before calculating new gradients.

---

# Step 4 — Forward Pass

```python
outputs = model(images)
```

The images pass through:

```text
Conv
 ↓
ReLU
 ↓
MaxPool
 ↓
Conv
 ↓
ReLU
 ↓
MaxPool
 ↓
Flatten
 ↓
Linear
 ↓
ReLU
 ↓
Linear
```

The result is:

```text
10 logits per image
```

---

# Step 5 — Calculate Loss

```python
loss_value = loss(outputs, labels)
```

The predictions are compared with the actual labels.

Example:

```text
Predicted → 7
Actual    → 7

Good prediction
↓
Lower loss
```

---

# Step 6 — Backpropagation

```python
loss_value.backward()
```

Backpropagation calculates gradients for the model's learnable parameters.

Conceptually:

```text
Loss
 ↓
Calculate Gradients
 ↓
Determine how each weight affected the error
```

---

# Step 7 — Update Weights

```python
optimizer.step()
```

Adam uses the calculated gradients to update the model's parameters.

The goal is:

```text
Minimize Loss
```

---

# 📉 Training Loss

The training results were:

| Epoch | Training Loss |
|---:|---:|
| 1 | 281.6750 |
| 2 | 84.0372 |
| 3 | 60.4121 |
| 4 | 47.7313 |
| 5 | 37.6779 |

The loss decreased significantly:

```text
Epoch 1
281.6750

       ↓

Epoch 5
37.6779
```

This indicates that the model was learning from the training data.

---

# 📉 Understanding the Loss Values

The code prints:

```python
total_loss += loss_value.item()
```

and then:

```python
print(
    f"Epoch {epoch + 1}/{epochs}, "
    f"Loss: {total_loss:.4f}"
)
```

Therefore, the reported value is the **sum of the batch losses across the epoch**, not the average loss per batch.

This is important when interpreting the numbers.

The reported loss:

```text
281.6750
```

should not be interpreted as:

```text
Average loss = 281.6750
```

It is the accumulated batch loss according to the current implementation.

A more comparable metric would be the average batch loss:

```python
average_loss = total_loss / len(train_loader)
```

---

# 🧪 Model Evaluation

After training, the model is switched to evaluation mode:

```python
model.eval()
```

Then gradients are disabled:

```python
with torch.no_grad():
```

This is useful because gradients are not required during testing.

It:

- Reduces memory usage
- Avoids unnecessary gradient calculations
- Makes inference more efficient

---

# 🔬 Test Prediction

For every test batch:

```python
outputs = model(images)
```

The predicted class is obtained using:

```python
predictions = outputs.argmax(dim=1)
```

Then predictions are compared with the actual labels:

```python
predictions == labels
```

Correct predictions are counted using:

```python
correct += (
    predictions == labels
).sum().item()
```

---

# 📊 Accuracy Calculation

The final accuracy is calculated using:

```python
100 * correct / total
```

The project achieved:

```text
Accuracy on the test set: 98.65%
```

Therefore:

```text
Test Accuracy = 98.65%
```

---

# 🎯 What Does 98.65% Accuracy Mean?

Accuracy is:

```text
Correct Predictions
──────────────────── × 100
Total Predictions
```

For example, if the model correctly classifies approximately 9,865 out of 10,000 test images:

```text
9,865
────── × 100
10,000
```

would be:

```text
98.65%
```

Therefore, approximately:

```text
98.65% → Correct
1.35%  → Incorrect
```

The exact count should be taken from the runtime values of `correct` and `total` rather than inferred solely from the rounded percentage.

---

# 🏆 Final Model Performance

| Metric | Result |
|---|---:|
| Dataset | MNIST |
| Problem | Image Classification |
| Classes | 10 |
| Model | Custom CNN |
| Epochs | 5 |
| Batch Size | 64 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | CrossEntropyLoss |
| Device | CUDA |
| Test Accuracy | **98.65%** |

---

# 💾 Saving the Model

The trained model is saved using:

```python
torch.save(
    model.state_dict(),
    'simpleCNN_mnist.pth'
)
```

The output file is:

```text
simpleCNN_mnist.pth
```

---

# 🧠 What Is `state_dict()`?

A PyTorch model contains learnable parameters such as:

```text
Weights
Biases
```

`state_dict()` contains these learned parameters.

Instead of saving the entire Python object, this project saves the model's learned weights.

---

# 🔄 Loading the Model Later

To load the trained model:

```python
model = simpleCNN().to(device)

model.load_state_dict(
    torch.load(
        'simpleCNN_mnist.pth',
        map_location=device
    )
)

model.eval()
```

The architecture must match the architecture used during training.

---

# 🖼️ Prediction Visualization

The project displays an image from the test batch:

```python
plt.imshow(
    images[0].cpu().squeeze(),
    cmap='gray'
)

plt.title(
    f"Predicted: {predictions[0].item()}"
)

plt.axis('off')
plt.show()
```

---

# 🔍 Why `.cpu()`?

If the tensor is stored on the GPU:

```text
CUDA Tensor
```

Matplotlib generally requires the data to be moved to CPU memory.

Therefore:

```python
images[0].cpu()
```

moves the image tensor from:

```text
GPU → CPU
```

---

# 🔍 Why `.squeeze()`?

The image originally has a channel dimension:

```text
1 × 28 × 28
```

For visualization, we want:

```text
28 × 28
```

Therefore:

```python
.squeeze()
```

removes dimensions of size 1.

---

# 🖼️ Prediction Example

The project visualizes a handwritten digit and displays:

```text
Predicted: <digit>
```

The visualization helps verify that the model's prediction corresponds to the displayed image.

You can add the generated screenshot to your repository, for example:

```text
images/
└── prediction.png
```

Then display it in this README using:

```markdown
![MNIST Prediction](images/prediction.png)
```

---

# 🧠 CNN vs SNN — Important Note

The project folder may be named:

```text
snn_vision
```

However, based on the provided implementation, this model is a:

> **Conventional Convolutional Neural Network (CNN)**

It is **not currently a Spiking Neural Network (SNN)**.

The code uses:

```python
nn.Conv2d
nn.ReLU
nn.MaxPool2d
nn.Linear
```

A typical SNN implementation would additionally involve concepts such as:

```text
Spiking Neurons
Membrane Potential
Spike Generation
Discrete Timesteps
Surrogate Gradients
Spike Trains
```

None of these are present in the provided implementation.

Therefore, the technically correct description of this project is:

```text
MNIST Image Classification
using a Convolutional Neural Network
implemented in PyTorch
```

If the goal of the project is specifically to learn **SNN-based computer vision**, this CNN can serve as a useful baseline for comparison against a future SNN implementation.

---

# 🔬 CNN Feature Extraction

The network can conceptually be divided into two parts:

```text
                CNN
                 │
       ┌─────────┴─────────┐
       │                   │
       ▼                   ▼
 Feature Extractor     Classifier
       │                   │
       ▼                   ▼
 Conv + ReLU + Pool    Linear + ReLU
 Conv + ReLU + Pool    Linear
       │                   │
       └─────────┬─────────┘
                 ▼
             10 Classes
```

The convolutional layers learn visual features.

The fully connected layers use those features for classification.

---

# 📐 Tensor Shape Flow

One of the most important parts of understanding CNNs is tracking tensor dimensions.

The input:

```text
[Batch, 1, 28, 28]
```

After first convolution:

```text
[Batch, 8, 28, 28]
```

After first max pooling:

```text
[Batch, 8, 14, 14]
```

After second convolution:

```text
[Batch, 16, 14, 14]
```

After second max pooling:

```text
[Batch, 16, 7, 7]
```

After flattening:

```text
[Batch, 784]
```

After first Linear layer:

```text
[Batch, 64]
```

After final Linear layer:

```text
[Batch, 10]
```

Therefore:

```text
1 × 28 × 28
      ↓
8 × 28 × 28
      ↓
8 × 14 × 14
      ↓
16 × 14 × 14
      ↓
16 × 7 × 7
      ↓
784
      ↓
64
      ↓
10
```

---

# 🔢 Number of Trainable Parameters

The model contains learnable parameters in its convolutional and linear layers.

## First Convolution

```text
Input Channels = 1
Output Channels = 8
Kernel = 3 × 3
```

Weights:

```text
8 × 1 × 3 × 3 = 72
```

Bias:

```text
8
```

Total:

```text
80
```

---

## Second Convolution

```text
Input Channels = 8
Output Channels = 16
Kernel = 3 × 3
```

Weights:

```text
16 × 8 × 3 × 3 = 1152
```

Bias:

```text
16
```

Total:

```text
1168
```

---

## First Linear Layer

Input:

```text
784
```

Output:

```text
64
```

Weights:

```text
784 × 64 = 50,176
```

Bias:

```text
64
```

Total:

```text
50,240
```

---

## Final Linear Layer

Input:

```text
64
```

Output:

```text
10
```

Weights:

```text
64 × 10 = 640
```

Bias:

```text
10
```

Total:

```text
650
```

---

# 🧮 Total Trainable Parameters

Therefore:

```text
First Conv       = 80
Second Conv      = 1,168
First Linear     = 50,240
Final Linear     = 650
--------------------------------
Total            = 52,138
```

The network has approximately:

```text
52.1K trainable parameters
```

This is a relatively small CNN, which makes it suitable for learning and experimentation.

---

# 📉 Training Loss Trend

The reported loss decreases across the five epochs:

```text
281.6750
    ↓
84.0372
    ↓
60.4121
    ↓
47.7313
    ↓
37.6779
```

This generally indicates successful optimization.

However, training loss alone does not tell us whether the model generalizes well.

For that reason, test-set evaluation is important.

---

# ⚠️ Loss vs Accuracy

Loss and accuracy measure different things.

## Loss

Loss measures how wrong the model's predictions are according to the chosen objective function.

This project uses:

```text
CrossEntropyLoss
```

## Accuracy

Accuracy measures:

```text
How many predictions are exactly correct
```

A model can sometimes have:

```text
Lower Loss
```

without a large increase in:

```text
Accuracy
```

because the model can become more confident in predictions that were already correct.

Therefore, both training loss and validation/test performance should be monitored.

---

# 🚨 Current Evaluation Limitation

The current implementation evaluates only:

```text
Test Accuracy
```

It does not calculate:

```text
Test Loss
Precision
Recall
F1 Score
Confusion Matrix
Per-Class Accuracy
```

For a more complete Computer Vision evaluation, these metrics can be added.

---

# 📊 Recommended Future Evaluation

A more complete evaluation pipeline could include:

```text
Accuracy
    ↓
Confusion Matrix
    ↓
Precision
    ↓
Recall
    ↓
F1 Score
    ↓
Per-Class Accuracy
```

For MNIST, a confusion matrix can help identify which digits the model confuses most frequently.

For example:

```text
3 ↔ 5
4 ↔ 9
7 ↔ 1
```

The actual confusion patterns should be calculated from the trained model rather than assumed.

---

# 🚀 Possible Improvements

The current CNN achieves:

```text
98.65% test accuracy
```

after only:

```text
5 epochs
```

Possible improvements include:

### 1. Increase Training Epochs

Try:

```text
10
15
20
```

epochs.

However, more epochs do not automatically guarantee better test performance.

---

### 2. Add Batch Normalization

Example:

```python
nn.BatchNorm2d(8)
```

Batch normalization can help stabilize training.

---

### 3. Add Dropout

Example:

```python
nn.Dropout(0.5)
```

Dropout can help reduce overfitting.

---

### 4. Data Augmentation

For image classification, transformations such as:

```text
Random Rotation
Random Affine
Random Crop
```

can be experimented with.

However, augmentation should preserve the identity of the digit.

---

### 5. Learning Rate Experiments

Current:

```text
0.001
```

Possible experiments:

```text
0.0001
0.0005
0.001
0.005
```

The best value should be selected based on validation performance.

---

### 6. Model Architecture Improvements

The network could be expanded:

```text
1 → 16 → 32 → 64
```

instead of:

```text
1 → 8 → 16
```

However, increasing model size also increases computational cost and the number of parameters.

---

# 🧪 CNN Baseline for Future SNN Experiments

If the ultimate objective is to study:

> **Spiking Neural Networks for Vision**

this CNN is a useful baseline.

You can compare:

```text
                  MNIST
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
      Standard CNN          SNN
          │                   │
          ▼                   ▼
     Accuracy              Accuracy
          │                   │
          ▼                   ▼
   Training Time          Training Time
          │                   │
          ▼                   ▼
    Parameters             Parameters
          │                   │
          ▼                   ▼
     Energy Cost            Energy Cost
```

A future SNN implementation could replace conventional ReLU activations with spiking neuron models and process inputs across discrete timesteps.

This would allow a meaningful comparison between conventional Deep Learning and neuromorphic/spiking approaches.

---

# 📁 Recommended Project Structure

A clean GitHub structure could be:

```text
snn_vision/
│
├── data/
│   └── MNIST/
│
├── images/
│   └── prediction.png
│
├── simpleCNN_mnist.pth
│
├── mnist_cnn.ipynb
│
├── README.md
│
├── requirements.txt
│
└── .gitignore
```

---

# 📦 Requirements

Example `requirements.txt`:

```text
torch
torchvision
matplotlib
```

If CUDA-specific PyTorch installation is required, the correct PyTorch build should be selected according to the installed NVIDIA driver and CUDA compatibility.

---

# ▶️ How to Run

## Step 1 — Clone the Repository

```bash
git clone <your-repository-url>
```

## Step 2 — Navigate to the Project

```bash
cd snn_vision
```

## Step 3 — Install Dependencies

```bash
pip install torch torchvision matplotlib
```

## Step 4 — Run the Notebook

Open the notebook in:

```text
Jupyter Notebook
```

or:

```text
VS Code
```

Run the cells sequentially.

The MNIST dataset will automatically download into:

```text
./data
```

---

# 💾 Generated Model

After training, the following model file is created:

```text
simpleCNN_mnist.pth
```

This file contains the learned weights of the CNN.

---

# 📈 Results

The model produced the following training results:

```text
Epoch 1/5, Loss: 281.6750
Epoch 2/5, Loss: 84.0372
Epoch 3/5, Loss: 60.4121
Epoch 4/5, Loss: 47.7313
Epoch 5/5, Loss: 37.6779
```

Final test performance:

```text
Accuracy on the test set: 98.65%
```

Model status:

```text
Model saved successfully. ✅
```

---

# 🏆 Final Results Summary

| Category | Result |
|---|---|
| Dataset | MNIST |
| Task | Handwritten Digit Classification |
| Input Size | 28 × 28 |
| Channels | 1 |
| Number of Classes | 10 |
| Architecture | CNN |
| Conv Layers | 2 |
| Fully Connected Layers | 2 |
| Activation | ReLU |
| Pooling | Max Pooling |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss | CrossEntropyLoss |
| Batch Size | 64 |
| Epochs | 5 |
| Device | CUDA |
| Trainable Parameters | 52,138 |
| Test Accuracy | **98.65%** |
| Saved Model | `simpleCNN_mnist.pth` |

---

# 🧠 Key Deep Learning Concepts Demonstrated

This project demonstrates practical understanding of:

- PyTorch
- Tensor operations
- CUDA acceleration
- MNIST
- Computer Vision
- CNNs
- Convolution
- Feature extraction
- ReLU activation
- Max pooling
- Flattening
- Fully connected layers
- Logits
- Cross-Entropy Loss
- Backpropagation
- Gradient descent
- Adam optimizer
- Learning rate
- Mini-batch training
- Epochs
- Model evaluation
- Classification accuracy
- Model serialization
- GPU inference
- Image visualization

---

# 🎓 What I Learned From This Project

The project demonstrates the fundamental workflow of training a Computer Vision model:

```text
Raw Images
     ↓
Tensor Conversion
     ↓
Batching
     ↓
CNN Feature Extraction
     ↓
Classification
     ↓
Loss Calculation
     ↓
Backpropagation
     ↓
Parameter Update
     ↓
Repeated Training
     ↓
Testing
     ↓
Prediction
```

The key idea behind CNN-based image classification is that the network automatically learns useful visual features rather than requiring us to manually define features.

The early convolutional layers can learn simple visual patterns, while deeper layers combine those patterns into more useful representations for classification.

---

# ⚠️ Important Technical Notes

### 1. This is a CNN, not an SNN

Although the project may be part of a repository named:

```text
snn_vision
```

the provided model is a conventional CNN.

There are no spiking neuron layers or temporal spike-processing mechanisms in the current implementation.

---

### 2. Accuracy Is Appropriate Here

Unlike regression projects where metrics such as R² are commonly used, this project is a classification problem.

Therefore:

```text
Accuracy = Appropriate Metric
```

The model achieved:

```text
98.65% Test Accuracy
```

---

### 3. Training Loss Is Accumulated

The code currently reports:

```python
total_loss += loss_value.item()
```

so the printed loss is the sum of batch losses.

For an average loss, use:

```python
average_loss = total_loss / len(train_loader)
```

---

### 4. Test Accuracy Does Not Guarantee Perfect Generalization

A 98.65% test accuracy is excellent for this small CNN on MNIST, but it does not mean the model will achieve the same performance on:

```text
Real handwritten images
Different image sizes
Different backgrounds
Different writing styles
Camera photographs
Scanned documents
```

MNIST is a relatively clean benchmark dataset.

---

# 🚀 Future Roadmap

```text
                    Current CNN
                         │
                         ▼
                  98.65% Accuracy
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
      Improve CNN                Build SNN
            │                         │
            ▼                         ▼
     BatchNorm                  Spiking Neurons
     Dropout                    Timesteps
     Augmentation               Membrane Potential
     More Layers                Surrogate Gradient
            │                         │
            └────────────┬────────────┘
                         ▼
                   Model Comparison
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Accuracy       Efficiency       Energy
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                Vision Model Analysis
```

---

# 📌 Conclusion

This project implements a lightweight Convolutional Neural Network using PyTorch for handwritten digit classification on MNIST.

The network consists of:

```text
2 Convolutional Layers
        +
2 Max Pooling Layers
        +
2 Fully Connected Layers
        +
ReLU Activations
```

The model was trained using:

```text
Optimizer      → Adam
Learning Rate  → 0.001
Loss Function  → CrossEntropyLoss
Batch Size     → 64
Epochs         → 5
```

The model achieved:

# 🏆 98.65% Test Accuracy

with approximately:

```text
52,138 trainable parameters
```

The trained model was saved as:

```text
simpleCNN_mnist.pth
```

This project provides a strong foundation for moving from conventional CNN-based Computer Vision toward **Spiking Neural Networks (SNNs)** and neuromorphic vision systems.

---

# 👨‍💻 Project Information

| Information | Details |
|---|---|
| Project | MNIST Digit Classification |
| Domain | Computer Vision |
| Framework | PyTorch |
| Dataset | MNIST |
| Model | Custom CNN |
| Hardware | CUDA GPU |
| Epochs | 5 |
| Batch Size | 64 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | CrossEntropyLoss |
| Test Accuracy | **98.65%** |
| Parameters | **52,138** |
| Model File | `simpleCNN_mnist.pth` |
| Status | ✅ Completed |

---

## ⭐ Project Status

```text
✅ Dataset Loaded
✅ GPU Acceleration
✅ CNN Implemented
✅ Model Trained
✅ Model Evaluated
✅ 98.65% Test Accuracy
✅ Model Saved
✅ Prediction Visualized
```

Niraj Singh
