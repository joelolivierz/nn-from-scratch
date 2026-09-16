# nn-from-scratch

A feedforward neural network built from first principles in NumPy — no PyTorch, no TensorFlow, no autograd. Forward pass, loss, and backpropagation are all written out by hand, then trained on MNIST to classify handwritten digits.

The point of the project is to understand what a framework hides.

**Current result: 96.09% accuracy on the 10,000-image MNIST test set**, after a single pass over the 60,000 training images.

## Contents

| Path                                 | Description                                                                |
| ------------------------------------ | -------------------------------------------------------------------------- |
| `nn-from-scratch.ipynb`              | Main notebook: the `Network` class, data loading, training, and evaluation |
| `dataset/`                           | MNIST in the original IDX binary format (train + test images and labels)   |
| `scripts/calculations_controller.nb` | Wolfram Mathematica notebook used to verify the forward-pass math by hand  |

## Requirements

- Python 3.12
- `numpy`
- `idx2numpy` — reads the raw IDX binary files in `dataset/`
- `jupyter` (or any notebook-capable editor)

```bash
pip install numpy idx2numpy jupyter
```

## Usage

```bash
git clone https://github.com/joelolivierz/nn-from-scratch.git
cd nn-from-scratch
jupyter notebook nn-from-scratch.ipynb
```

Run the cells top to bottom. The dataset is committed to the repo, so no download step is needed. Paths are relative to the repository root, so start the notebook from there.

Training and evaluation take a few minutes — every sample is processed individually, in Python, with no batching.

## The `Network` class

A network is defined by a single tuple of layer sizes, input and output layers included:

```python
network = Network((784, 256, 128, 10))
```

That gives a 784-dimensional input (a flattened 28×28 image), two hidden layers of 256 and 128 units, and a 10-way output — one unit per digit.

### Initialization

Biases start at zero. Weights use **He initialization**, drawing from a normal distribution scaled by `sqrt(2 / fan_in)`, which is the standard choice for ReLU networks.

### Forward pass

Each hidden layer computes `ReLU(W @ a + b)`. The output layer applies **softmax** instead, turning the final scores into a probability distribution over the ten digits. The softmax subtracts the maximum value before exponentiating — a numerical-stability trick that prevents `exp` from overflowing without changing the result.

Every intermediate activation is cached in `self.activations`, because backpropagation needs them.

### Loss

Categorical cross-entropy: `-sum(expected * log(predicted))`, with the expected label as a one-hot vector.

### Backward pass

Backpropagation is implemented in `backward_pass`. For softmax combined with cross-entropy, the gradient at the output layer collapses to `predicted - expected`.

From there the error is propagated backwards layer by layer. Weight gradients come from the outer product of the incoming error and the layer's activation; the error is then pushed through `W.T` and masked by the ReLU derivative, which is simply 1 wherever the activation was positive and 0 elsewhere. Weights and biases are updated in place with plain gradient descent.

```python
network.forward_pass(image)          # image: flattened, scaled to [0, 1]
network.backward_pass(one_hot_label, learning_rate=0.01)
```

Updates happen once per sample.

## Training setup

| Setting           | Value                         |
| ----------------- | ----------------------------- |
| Architecture      | 784 → 256 → 128 → 10          |
| Hidden activation | ReLU                          |
| Output activation | Softmax                       |
| Loss              | Categorical cross-entropy     |
| Optimizer         | SGD, one sample at a time     |
| Learning rate     | 0.01                          |
| Epochs            | 1                             |
| Preprocessing     | Flatten to 784, divide by 255 |

## Results

| Split         | Images | Errors | Accuracy |
| ------------- | ------ | ------ | -------- |
| Test (`t10k`) | 10,000 | 391    | 96.09%   |
| Train         | 60,000 | 2087   | 96.52%   |

## The Mathematica notebook

`scripts/calculations_controller.nb` was a debugging aid. It walks a small toy network — input in R^14, then layers of 7, 7, and 3 units — through the same sequence of operations, printing every matrix and intermediate vector: `z1 = W1·x + b1`, `h1 = ReLU(z1)`, and so on through the final softmax.

Having an independent implementation of the forward pass makes it easy to check computations from python.

## Known limitations

- **`loss_function` is never called during training.** It's correct and useful for inspecting a single prediction, but the training loop uses the analytic `predicted - expected` gradient directly and never evaluates the loss. Nothing tracks loss over time.
- **No batching, no shuffling, no validation split.** Samples are visited once, in file order.
- **No regularization or learning-rate schedule.** The learning rate is fixed at 0.01 throughout.

## Dataset

The MNIST database of handwritten digits, by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges. 60,000 training and 10,000 test images, each a 28×28 grayscale digit. The original IDX files are included in `dataset/` unchanged.
