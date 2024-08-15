# %% [markdown]
"""
# Exercise 1: Introduction to Deep Learning
<div>
    <table>
        <tr style="background-color:white">
            <td><img src="attachments/perceptron.png" width="100%"/></td>
            <td><img src="attachments/mlp.png" width="100%"/></td>
            <td><img src="attachments/neural_network.png" width="100%"/></td>
        </tr>
    </table>
</div>

In the following exercise we will explore the basic building blocks of deep learning: the perceptron and how to stack multiple perceptrons together into layers to build a neural network. We will also introduce convolutional neural networks (CNNs) for image classification.
In particular, we will:
- Implement a perceptron and a 2-layer perceptron to compute the XOR function using NumPy.
- Introduce PyTorch, a popular framework for deep learning.
- Implement and train a simple neural network (a multi-layer perceptron) to classify points in a 2D plane using PyTorch.
- Implement and train a simple deep convolutional neural network to classify hand-written digits from the MNIST dataset using PyTorch.
- Discuss important topics in ML/DL, such as data splitting, under/overfitting and model generalization.

<div class="alert alert-block alert-danger">
    Set your python kernel to <code>01_intro_dl</code>
    <tr style="background-color:white">
        <td><img src="attachments/kernel-change.png" width="100%"/></td>
    </tr>
</div>

### Acknowledgements

The original notebook was created by Nils Eckstein, Julia Buhmann, and Jan Funke. Albert Dominguez Mantes ported the notebook to PyTorch.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (5, 5) # this line sets the default size of plots

# %% [markdown]
"""
## Part 1: Perceptrons

<div>
    <img src="attachments/perceptron.png" width="600"/>
</div>

As we saw in the lecture ["Introduction to Deep Learning"](intro_dl_lecture.pdf), a perceptron is a simple unit that combines its inputs $x_i$ in a linear fashion (using weights $w_i$ and a bias $b$), followed by a non-linear function $f$.

<div class="alert alert-block alert-info">
    <b>Task 1</b>: Implement a Perceptron Function
</div>

Using only `numpy`, write a function `perceptron(x, w, b, f)` that returns `y` as computed by a perceptron, for arbitrary inputs `x` of dimension `n`. The arguments of your function should be:

* `x`: the input of the perceptron, a `numpy` array of shape `(n,)`
* `w`: the weights of the perceptron, a `numpy` array of shape `(n,)`
* `b`: a single scalar value for the bias
* `f`: a nonlinear function $f: \mathbb{R}\mapsto\mathbb{R}$

Test your perceptron function on 2D inputs (i.e., `n=2`) and plot the result. Change the weights, bias, and the function $f$ and see how the output of the perceptron changes.
"""


# %% tags=["task"]
def non_linearity(a):
    """Implement your non-linear function here."""
    return


# %% tags=["solution"]
def non_linearity(a):
    return a > 0


# %% tags=["task"]
def perceptron(x, w, b, f):
    """Implement your perceptron here."""
    return


# %% tags=["solution"]
def perceptron(x, w, b, f):
    return f(np.sum(x * w) + b)


# %%
def plot_perceptron(w, b, f):
    """This function will evaluate the perceptron on a grid of arbitrary points
       (equispaced across 0-1) and plot the result in each point, which will reveal
       the decision boundary of the perceptron.
    """
    
    num_samples = 100 # number of samples in each dimension
    domain_x1 = (0.0, 1.0) # domain of the plot (x-axis)
    domain_x2 = (0.0, 1.0) # domain of the plot (y-axis)

    domain = np.meshgrid(
        np.linspace(*domain_x1, num_samples), np.linspace(*domain_x2, num_samples)
    ) # create a grid of equispaced points in the domain

    xs = np.array([domain[0].flatten(), domain[1].flatten()]).T # format the points as a list of 2D points to evaluate the perceptron on

    values = np.array([perceptron(x, w, b, f) for x in xs]) # evaluate the perceptron on each point in the grid

    plt.contourf(domain[0], domain[1], values.reshape(num_samples, num_samples)) # plot the result as filled contours


# the following should show a linear classifier that is True (shown as green)
# for values below a line starting at (0.1, 0) through (1.0, 0.9)
plot_perceptron(w=[1.0, -1.0], b=-0.1, f=non_linearity)


# %% [markdown]
"""
<div class="alert alert-block alert-success">
<h2> Checkpoint 1 </h2>
You have implemented a perceptron using basic Python and NumPy functions, as well as checked what the perceptron decision boundary looks like.
We will now go over different ways to implement the perceptron together and discuss their efficiency. If you arrived here earlier, feel free to play around with the parameters of the perceptron (the weights and bias) as well as the activation function `f`.

Time: 20 working, + 10 discussion
</div>
"""

# %% [markdown]
"""
<div class="alert alert-block alert-info">
    <h2>Task 2</h2>
    
Create a 2-Layer Network for XOR
</div>

XOR is a fundamental logic gate that outputs `1` whenever there is an odd number of `1` in its input and `0` otherwise. For two inputs this can be thought of as an "exclusive or" operation and the associated boolean function is fully characterized by the following truth table.

| x1 | x2 | y = XOR(x1, x2) |
|---|---|----------|
| 0 | 0 |    0     |
| 0 | 1 |    1     |
| 1 | 0 |    1     |
| 1 | 1 |    0     |
"""

# %%
def generate_xor_data():
    """Generate XOR data for pairs of binary inputs:
    f(0,0) = 0
    f(0,1) = 1
    f(1,0) = 1
    f(1,1) = 0
    """
    xs = [np.array([i, j]) for i in [0, 1] for j in [0, 1]]
    ys = [int(np.logical_xor(x[0], x[1])) for x in xs]
    return xs, ys

def plot_xor_data():
    """Plot the XOR data.
    """
    xs, ys = generate_xor_data()
    for x, y in zip(xs, ys):
        plt.scatter(*x, color="green" if y else "red")
    
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.grid(True)
    plt.gca().set_frame_on(False)
    plt.show()

plot_xor_data()


# %% [markdown]
"""The function of an XOR gate can also be understood as a binary classification problem given a 2D binary inputs $x$ ($x$ \in \{0,1\}^2$ and we can think about designing a classifier acting as an XOR gate. It turns out that this problem is not solvable by a single perceptron (https://en.wikipedia.org/wiki/Perceptron) because the set of points $\{(0,0), (0,1), (1,0), (1,1)\}$ is not linearly separable.

![mlp.png](attachments/mlp.png)

Design a two layer perceptron using your `perceptron` function above that implements an XOR Gate on two inputs. Think about the flow of information through this simple network and set the weight values by hand such that the network produces the XOR function.

#### Hint

A single layer in a multilayer perceptron can be described by the equation $y = f(x^\intercal w + b)$ with $f$ a nonlinear function. $b$ is the so called bias (a constant offset vector) and $w$ a vector of weights. Since we are only interested in outputs of `0` or `1`, a good choice for $f$ is the threshold function. Think about which kind of logical operations you can implement with a single perceptron, then see how you can combine them to create an XOR. It might help to write down the equation for a two layer perceptron network.
"""


# %% tags=["task"]
def xor(x):
    """
    Implement your solution here
    """

    # We will refer to the weights and bias of the two perceptrons in the first layer
    # as w11 and b11, and the weights and bias of the perceptron in the last layer
    # as w2 and b2. Change their values below such that the whole network implements
    # the XOR function. You will also have to change the activation function f used
    # for the perceptrons (which currently is the identity).

    # TASK: set the weights, biases and activation function of the perceptrons
    w11 = [0.0, 0.0] # weights of the first perceptron in the first layer
    b11 = 0.0 # bias of the first perceptron in the first layer
    w12 = [0.0, 0.0] # weights of the second perceptron in the first layer
    b12 = 0.0 # bias of the second perceptron in the first layer
    w2 = [0.0, 0.0] # weights of the perceptron in the last layer
    b2 = 0.0 # bias of the perceptron in the last layer
    f = lambda a: a # activation function of the perceptrons.
    # END OF TASK

    # output of the two perceptrons in the first layer
    h1 = perceptron(x, w=w11, b=b11, f=f)
    h2 = perceptron(x, w=w12, b=b12, f=f)
    # output of the perceptron in the last layer
    y = perceptron(np.array([h1, h2]), w=w2, b=b2, f=f)  # h1 AND NOT h2

    return y


# %% tags=["solution"]
def xor(x):
    # SOLUTION
    w11 = [0.1, 0.1] # weights of the first perceptron in the first layer
    b11 = -0.05 # bias of the first perceptron in the first layer
    w12 = [0.1, 0.1] # weights of the second perceptron in the first layer
    b12 = -0.15 # bias of the second perceptron in the first layer
    w2 = [0.1, -0.1] # weights of the perceptron in the last layer
    b2 = -0.05 # bias of the perceptron in the last layer
    f = lambda a: a > 0 # activation function of the perceptrons (threshold function)

    # output of the two perceptrons in the first layer
    h1 = perceptron(x, w=w11, b=b11, f=f)
    h2 = perceptron(x, w=w12, b=b12, f=f)
    # output of the perceptron in the last layer
    y = perceptron(np.array([h1, h2]), w=w2, b=b2, f=f)  # h1 AND NOT h2

    return y


# %%
def test_xor():
    xs, ys = generate_xor_data()
    for x, y in zip(xs, ys):
        assert (
            xor(x) == y
        ), f"xor function returned {int(xor(x))} for input {x}, but should be {y}"
        print(f"XOR of {x} is {y}, your implementation returns {int(xor(x))}")
    print("\nCongratulations! You have implemented the XOR function correctly.")


test_xor()

# %% [markdown]
"""
<div class="alert alert-block alert-success">
<h2> Checkpoint 2 </h2>
You have been introduced to the XOR gate and its view as a binary classification problem. You have also solved XOR using a two-layer perceptron.
There are many ways to implement an XOR in a two-layer perceptron. We will review some of them and how we got to them (trial and error or pen and paper?).
    
<br/>
If you arrive here early, think about how to generalize the XOR function to an arbitrary number of inputs. For more than two inputs, the XOR returns True if the number of 1s in the inputs is odd, and False otherwise.

Time: 30 working + 15 min discussion
</div>
"""
# %% [markdown]
"""
## Part 2: "Deep" Neural Networks

<div>
    <img src="attachments/neural_network.png" width=500/>
</div>

<div class="alert alert-block alert-info">
    <h2>Task 3</h2>

Use PyTorch to Train a Simple Network
</div>

The previous task demonstrated that chosing the weights of a neural network by hand can be quite painful even for simple functions. This will certainly get out of hand once we have more complex networks with several layers and many neurons per layer. But more importantly, the reason why we want to use neural networks to approximate a function is that (in general) we do not know exactly what the function is. We only have data points that describe the function implicitly.

In this task, we will design, train, and evaluate a neural network that can classify points of two different classes on a 2D plane, i.e., the input to our network are the coordinates of points in a plane.

For that, we will create a training and a testing dataset. We will use stochastic gradient descent to train a network on the training dataset and evaluate its performance on the testing dataset.

#### Data

We create both training and testing dataset from the following function (in practice, we would not know this function but have only the data available):
"""


# %%
def generate_spiral_data(n_points, noise=1.0):
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    return (
        np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
        np.hstack((np.zeros(n_points), np.ones(n_points))),
    )


def plot_points(Xs, ys, titles):
    num_subplots = len(Xs)
    plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5))
    for i, (X, y, title) in enumerate(zip(Xs, ys, titles)):
        plt.subplot(1, num_subplots, i + 1)
        plt.title(title)
        plt.plot(X[y == 0, 0], X[y == 0, 1], ".", label="Class 1")
        plt.plot(X[y == 1, 0], X[y == 1, 1], ".", label="Class 2")
        plt.legend()
    plt.show()


X_train, y_train = generate_spiral_data(100)
X_test, y_test = generate_spiral_data(1000)

plot_points([X_train, X_test], [y_train, y_test], ["Training Data", "Testing Data"])

# %% [markdown]
"""
We will start with a simple baseline model. But first, we will explicitly write the training loop (required by vanilla PyTorch), which you have gone through in the lecture. Comments in the code will help you identify the different steps involved.
"""

# %%
import torch


def batch_generator(X, y, batch_size, shuffle=True):
    if shuffle:
        # Shuffle the data at each epoch
        indices = np.random.permutation(len(X))
    else:
        # Process the data in the order as it is
        indices = np.arange(len(X))
    for i in range(0, len(X), batch_size):
        yield X[indices[i : i + batch_size]], y[indices[i : i + batch_size]]


def run_epoch(model, optimizer, X_train, y_train, batch_size, loss_fn, device):
    n_samples = len(X_train)
    total_loss = 0

    # Set the model to training mode, essential when using certain layers
    model.train()
    for X_b, y_b in batch_generator(X_train, y_train, batch_size):
        # Convert the data to PyTorch tensors
        X_b = torch.tensor(X_b, dtype=torch.float32, device=device)
        y_b = torch.tensor(y_b, dtype=torch.float32, device=device)

        # Reset the optimizer state
        optimizer.zero_grad()

        # Forward pass: pass the data through the model and retrieve the prediction
        y_pred = model(X_b).squeeze()
        # Note: the .squeeze() method above removes dimensions of size 1, which is useful in this case as we are predicting a single value.
        # Before squeezing, the shape would be (B, 1). After squeezing, it is (B,), which is the shape of our target values y_b.
        # The inverse of .squeeze() is .unsqueeze(), which adds dimensions of size 1. This is useful when e.g. you want to add a batch dimension to a single sample, or a channel dimension in a single-channel image.

        # Compute the loss function with the prediction and the ground truth
        loss = loss_fn(y_pred, y_b)
        # Note: even if a single number is returned, it is still a tensor with an associated computational graph (--> more memory used).
        # Be extremely careful when using the loss tensor in other calculations (e.g. for monitoring issues), as it can lead to memory leaks and other errors.
        # For those, you should always use the .item() method to convert to a native Python number (see the last comment of the function).

        # Backward pass: compute the gradient of the loss w.r.t. the parameters
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Accumulate the loss (for monitoring purposes)
        total_loss += loss.item() # the .item() converts the single-number Tensor to a Python floating point number, avoiding retaining the computational graph in the loss tensor
    return total_loss / n_samples


# %% [markdown]
"""
Before continuing, you should know that PyTorch is object-oriented (OOP) and follows specific class structures. If you are not too familiar with Python or OOP, it may be a bit tricky to understand the structure at first, and what executes when. Don't despair! Getting the grasp on it is easier than it seems. Here we will focus on getting the architecture of the model right, so most of the boilerplate work will be already lifted.

So, let's now write the simple baseline model, consisting of one hidden layer with 12 neurons (or perceptrons). You will see that this baseline model performs pretty poorly. Read the following code snippets and try to understand the involved functions:
"""

# %%
import torch.nn as nn
from tqdm.auto import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("GPU not available. Will use CPU.")
    device = torch.device("cpu")


class BaselineModel(nn.Module):
    def __init__(self):
        """This method (:= `constructor`) is automatically called when the class instance is created, i.e. `model = BaselineModel()`
        Note that this initializes the model architecture, but does not yet apply it to any data. This is done in the `forward` method.
        """
        super().__init__()

        # The input to the next block is a tensor of size (B, 2), where 2 is the number of features.
        # The block then sequentially applies a linear transformation, a non-linear activation function, another linear transformation, and another non-linear activation function.
        # The output of the following block is a tensor of size (B, 1), where B is the batch size, which will be the predicted class of the input data.
        self.mlp = nn.Sequential(
            nn.Linear(in_features=2, out_features=12, bias=True), # this layer receives a tensor of size (B, 2) and returns a tensor of size (B, 12)
            nn.Tanh(), # Tanh is a non-linear activation function that squashes the output to the range [-1, 1]
            nn.Linear(in_features=12, out_features=1), # this layer receives a tensor of size (B, 12) and returns a tensor of size (B, 1)
            nn.Sigmoid(), # Sigmoid is a non-linear activation function that squashes the output to the range [0, 1], widely used for binary classification
        )
        # Note: the output of the block is a number between 0 and 1. In simplifying terms, you can think of it as "the probability of the input data belonging to class 1".

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method can be called to perform a forward pass of the model.
           It is automatically called when the class instance is called as a function, i.e. `model(x)`, which is highly recommended (so, in general, don't use model.forward(x) but model(x)).
           In this example we have one module, but you can have multiple modules and combine them here.

        Args:
            x (torch.Tensor): The input data, which should have the shape (B, 2) in this case, where B is the batch size

        Returns:
            torch.Tensor: results of applying the model to the input data, Shape will be (B, 1)
        """
        return self.mlp(x)


# Initialize the model, optimizer and set the loss function
bad_model = BaselineModel()
# The .to() method will move the model to the appropiate device (e.g. the GPU if available)
bad_model.to(device)
optimizer = torch.optim.SGD(
    bad_model.parameters(), lr=0.01
)  # SGD - Stochastic Gradient Descent
loss_fn = nn.MSELoss(reduction="sum")  # MSELoss - Mean Squared Error Loss

batch_size = 10
num_epochs = 1500


for epoch in (pbar := tqdm(range(num_epochs), total=num_epochs, desc="Training")):
    # Run an epoch over the training set
    curr_loss = run_epoch(
        bad_model, optimizer, X_train, y_train, batch_size, loss_fn, device
    )

    # Update the progress bar to display the training loss
    pbar.set_postfix({"training loss": curr_loss})

# %% [markdown]
"""
Now that we've trained the model, let's evaluate its performance on the testing dataset. The following code snippet will retrieve the predictions from the model, and will then plot them along with the test data:
"""


# %%
def predict(model, X, y, batch_size, device):
    predictions = np.empty((0,))
    model.eval() # set the model to evaluation mode
    with torch.inference_mode(): # this "context manager" is used to disable gradient computation (among others), which is not needed during inference and offers improved performance
        for X_b, y_b in batch_generator(X, y, batch_size, shuffle=False):
            X_b = torch.tensor(X_b, dtype=torch.float32, device=device)
            y_b = torch.tensor(y_b, dtype=torch.float32, device=device)
            y_pred = model(X_b).squeeze().detach().cpu().numpy()
            # Note: the last chain of methods (in order) do: remove a unit dimension (.squeeze()),
            # detach the tensor from the computational graph (.detach()),
            # move it to the CPU (.cpu()),
            # and convert it to a NumPy array (.numpy())
            predictions = np.concatenate((predictions, y_pred), axis=0)
    return np.round(predictions)


def accuracy(y_pred, y_gt):
    return np.sum(y_pred == y_gt) / len(y_gt)


bad_predictions = predict(bad_model, X_test, y_test, batch_size, device)
bad_accuracy = accuracy(bad_predictions, y_test)

plot_points(
    [X_test, X_test],
    [y_test, bad_predictions],
    ["Testing data", f"Bad Model Classification ({bad_accuracy * 100:.2f}% correct)"],
)

# %% [markdown]
"""
<div class="alert alert-block alert-info">
    <b>Task 3.1</b>: Improve the Baseline Model
</div>

Now, try to find a more advanced architecture that is able to solve the classification problem. You can vary width (number of neurons per layer) and depth (number of layers) of the network. You can also play around with [different activation functions](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity), [loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions), and [optimizers](https://pytorch.org/docs/stable/optim.html).

Hint: some commonly used losses are `nn.BCELoss()` (binary crossentropy loss), `nn.MSELoss()` or `nn.L1Loss()` (one of them is particularly used for binary problems... :)). Some commonly used optimizers, apart from `torch.optim.SGD()`, are `torch.optim.AdamW()`, or `torch.optim.Adagrad()`.
"""


# %% tags=["task"]
class GoodModel(nn.Module):
    def __init__(self):
        super().__init__()
        # TASK
        self.mlp = nn.Sequential(
            # add your layers and activation functions here
        )
        # END OF TASK

    def forward(self, x):
        return self.mlp(x)


# Initialize the model
good_model = GoodModel()
good_model.to(device)

# TASK: set the optimizer and the loss function
optimizer = None  # Remember to instantiate the class with the model parameters and the learning rate
loss_fn = (
    None  # Remember to instantiate the class with the reduction parameter set to "sum"
)

assert optimizer is not None, "Please set the optimizer!"
assert loss_fn is not None, "Please set the loss!"

good_model.train()

for epoch in (pbar := tqdm(range(num_epochs), total=num_epochs, desc="Training")):
    # Run an epoch over the training set
    curr_loss = run_epoch(
        good_model, optimizer, X_train, y_train, batch_size, loss_fn, device
    )

    # Update the progress bar to display the training loss
    pbar.set_postfix({"training loss": curr_loss})

good_predictions = predict(good_model, X_test, y_test, batch_size, device)
good_accuracy = accuracy(good_predictions, y_test)

plot_points(
    [X_test, X_test],
    [y_test, good_predictions],
    ["Testing data", f"Good Model Classification ({good_accuracy * 100:.2f}% correct)"],
)


# %% tags=["solution"]
class GoodModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            # SOLUTION
            nn.Linear(in_features=2, out_features=64, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.seq(x)


# Instantiate the model
good_model = GoodModel()
good_model.to(device)

# SOLUTION
optimizer = torch.optim.AdamW(good_model.parameters(), lr=0.001)
loss_fn = nn.BCELoss(reduction="sum")  # Binary Cross Entropy Loss


batch_size = 10
num_epochs = 1500


good_model.train()
for epoch in (pbar := tqdm(range(num_epochs), total=num_epochs, desc="Training")):
    # Run an epoch over the training set
    curr_loss = run_epoch(
        good_model, optimizer, X_train, y_train, batch_size, loss_fn, device
    )

    # Update the progress bar to display the training loss
    pbar.set_postfix({"training loss": curr_loss})

good_model.eval()
good_predictions = predict(good_model, X_test, y_test, batch_size, device)
good_accuracy = accuracy(good_predictions, y_test)

plot_points(
    [X_test, X_test],
    [y_test, good_predictions],
    ["Testing data", f"Good Model Classification ({good_accuracy * 100:.2f}% correct)"],
)

# %% [markdown]
"""
<div class="alert alert-block alert-info">
    <b>Task 3.2</b>: Visualize Your Model
</div>

The next cell visualizes the output of your model for all 2D inputs with coordinates between 0 and 1, similar to how we plotted the output of the perceptron in **Task 1**. Change the code below to show the domain -15 to 15 for both input dimensions and compare the outputs of the `bad_model` model with yours. See also how the model performs outside the intervals it was trained on by increasing the domain even further.
"""


# %% tags=["task"]
def plot_classifiers(classifier_1, classifier_2):

    plt.subplots(1, 2, figsize=(10, 5))

    num_samples = 200

    # TASK: change the plotted domain here
    domain_x1 = (0.0, 1.0)
    domain_x2 = (0.0, 1.0)
    # END OF TASK

    domain = np.meshgrid(
        np.linspace(*domain_x1, num_samples), np.linspace(*domain_x2, num_samples)
    )
    xs = np.array([domain[0].flatten(), domain[1].flatten()]).T

    values_1 = predict(classifier_1, xs, np.zeros(xs.shape[0]), batch_size, device)
    values_2 = predict(classifier_2, xs, np.zeros(xs.shape[0]), batch_size, device)

    plt.subplot(1, 2, 1)
    plt.title("Bad Model")
    plt.contourf(domain[0], domain[1], values_1.reshape(num_samples, num_samples))

    plt.subplot(1, 2, 2)
    plt.title("Good Model")
    plt.contourf(domain[0], domain[1], values_2.reshape(num_samples, num_samples))

    plt.show()


plot_classifiers(bad_model, good_model)


# %% tags=["solution"]
def plot_classifiers(classifier_1, classifier_2):

    plt.subplots(1, 2, figsize=(10, 5))

    num_samples = 200

    # SOLUTION
    domain_x1 = (-100.0, 100.0)
    domain_x2 = (-100.0, 100.0)

    domain = np.meshgrid(
        np.linspace(*domain_x1, num_samples), np.linspace(*domain_x2, num_samples)
    )
    xs = np.array([domain[0].flatten(), domain[1].flatten()]).T

    values_1 = predict(classifier_1, xs, np.zeros(xs.shape[0]), batch_size, device)
    values_2 = predict(classifier_2, xs, np.zeros(xs.shape[0]), batch_size, device)

    plt.subplot(1, 2, 1)
    plt.title("Bad Model")
    plt.contourf(domain[0], domain[1], values_1.reshape(num_samples, num_samples))

    plt.subplot(1, 2, 2)
    plt.title("Good Model")
    plt.contourf(domain[0], domain[1], values_2.reshape(num_samples, num_samples))

    plt.show()


plot_classifiers(bad_model, good_model)

# %% [markdown]
"""
<div class="alert alert-block alert-warning">
    <b>Question:</b>
    Looking at the classifier on an extended domain, what observations can you make?
</div>

<div class="alert alert-block alert-success">
<h2> Checkpoint 3</h2>
You have now been introduced to PyTorch and trained a simple neural network on a binary classification problem. You have also seen how to visualize the decision function of the model, and what happens if the model is applied to a domain it had not seen during training.
Let us know in the exercise channel when you got here and what accuracy your model achieved! We will compare different solutions and discuss why some of them are better than others. We will also discuss the generalization behaviour of the classifier outside of the domain it was trained on.

Time: 60 working + 15 discussion
</div>
"""

# %% [markdown]
"""
<div class="alert alert-block alert-info">
    <h2>Task 4</h2>

Classify Hand-Written Digits
</div>

In this task, we will classify data points of higher dimensions: Each data point is now an image of size 28 by 28 pixels depicting a hand-written digit from the famous MNIST dataset.

Instead of feeding the image as one long vector into a fully connected network (as in the previous task), we will take advantage of the spatial information in images and use a convolutional neural network. As a reminder, a convolutional neural network differs from a fully connected one in that not each pair of nodes is connected, and weights are shared between nodes in one layer:

<div>
<img src="attachments/convolutional_network.png" width="300"/>
</div>

However, the output of our network will be a 10-dimensional vector, indicating the probabilities for the input to be one of ten classes (corresponding to the digits 0 to 9). For that, we will use fully connected layers at the end of our network, once the dimensionality of a feature map is small enough to capture high-level information.

In principle, we could just use convolutional layers to reduce the size of each feature map by 2 until one feature map is small enough to allow using a fully connected layer. However, it is good practice to have a convolutional layer followed by a so-called downsampling layer, which effectively reduces the size of the feature map by the downsampling factor.
"""


# %% [markdown]
"""
### Data
The following snippet will download the MNIST dataset using the `torchvision` library. The `transforms=transforms.ToTensor()` parameter will ensure that the data format is appropriate for using it directly (adding a channel dimension, rescaling values between 0 and 1).
"""

# %%
from torchvision.datasets import MNIST
from torchvision import transforms

all_train_ds = MNIST(
    root=".mnist", train=True, download=True, transform=transforms.ToTensor()
)
test_ds = MNIST(
    root=".mnist", train=False, download=True, transform=transforms.ToTensor()
)

# %% [markdown]
"""
The dataset is already split into training and test data, but we will further split the training data into training and validation, and show a few samples in the next cell.
"""

# %%
num_all_train_samples = len(all_train_ds)
train_ds, val_ds = torch.utils.data.random_split(
    all_train_ds, [int(0.8 * num_all_train_samples), int(0.2 * num_all_train_samples)]
)

print(f"Training data has {len(train_ds)} samples")
print(f"Validation data has {len(val_ds)} samples")
print(f"Testing data has {len(test_ds)} samples")


def show_samples(dataset, title, predictions=None, num_samples=10):
    plt.close()
    fig, axs = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3))
    fig.suptitle(title, size=40, y=1.2)
    if predictions is not None:
        assert len(predictions) == len(
            dataset
        ), "Number of given predictions must match number of samples"
    for i in range(num_samples):
        img, label = dataset[i]
        if predictions is not None:
            label = int(predictions[i])
        img = img.squeeze().numpy()
        axs[i].imshow(img, cmap="gray")
        (
            axs[i].set_title(f"Label: {label}")
            if predictions is None
            else axs[i].set_title(f"Prediction: {label}")
        )
        axs[i].axis("off")
    plt.show()


show_samples(train_ds, "Training Data")
show_samples(val_ds, "Validation Data")
show_samples(test_ds, "Testing Data")

# %% [markdown]
"""
Let us make sure that the data is in the right format for using with `torch` modules. Convolutional layers expect an input shape of (B, C, H, W) (batch, channel, height and width). The batch dimension represents different samples in a batch. Therefore, each sample (image) in our dataset should be (1,28,28), as the data is single-channel. We will also check the labels to make sure they are integers between 0 and 9.

While manually checking a couple of images is fine (and recommended), it is also good to automatize this process and check the data format in general (as long as the size allows so!).
"""
# %%
print("Training image shape:", train_ds[0][0].shape)
print("Training image label:", train_ds[0][1])

print("Validation image shape:", val_ds[0][0].shape)
print("Validation image label:", val_ds[0][1])

print("Testing image shape:", test_ds[0][0].shape)
print("Testing image label:", test_ds[0][1])

assert all(
    img.shape == (1, 28, 28) and isinstance(label, int) and 0 <= label <= 9
    for img, label in train_ds
), "Unexpected shape, type or label for training data"

assert all(
    img.shape == (1, 28, 28) and isinstance(label, int) and 0 <= label <= 9
    for img, label in val_ds
), "Unexpected shape, type or label for validation data"

assert all(
    img.shape == (1, 28, 28) and isinstance(label, int) and 0 <= label <= 9
    for img, label in test_ds
), "Unexpected shape, type or label for test data"
print("\nData format is correct.")

# %% [markdown]
"""
<div class="alert alert-block alert-info">
    <b>Task 4.1</b>: Implement a Convolutional Neural Network
</div>

Create a CNN using `torch` module with the following specifications:
* one convolution, size 3x3, 32 output feature maps, padding=1, followed by a ReLU activation function
* one downsampling layer, size 2x2, via max-pooling
* one convolution, size 3x3, 32 output feature maps, padding=1, followed by a ReLU activation function
* one downsampling layer, size 2x2, via max-pooling
* one fully connected (linear) layer with 64 units (the previous feature maps need to be flattened for that), followed by a ReLU activation function
* one fully connected (linear) layer with 10 units, **without any activation function**. This will be the logits of the network.

The fact that we do not add any activation function in the output is because certain loss functions in PyTorch (e.g. [`nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)) expect the logits of the network and already apply the activation function in a more efficient manner when computing the loss, offering speedup and more numerical stability compared to explicitly adding it. Therefore, one should not to add an activation function in the output layer when using these loss functions during training (always double check what is the expected input for the loss function you want to use!).

Each layer above has a corresponding `torch` implementation (e.g., a convolutional layer is implemented by [`nn.Conv2D`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html), and the linear layer by [`nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html), which you have used before in Task 3). Please find the other necessary modules by browsing the [torch.nn documentation](https://pytorch.org/docs/stable/nn.html)! Flattening can be achieved by using the [`nn.Flatten` module](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html) with its default parameters.


<div class="alert alert-block alert-warning">
    <b>Question:</b>
    PyTorch requires explicitly giving the number of input features/channels to each Linear/Conv2D layer. Therefore, you need to know the number of input features/channels for those layers.
    What is the number of input features/channels for each layer in the CNN described above? Take particular care with the number of input features in the first fully connected layer (after flattening). You can assume the convolutional layers will preserve the input spatial size (thanks to the `padding=1`). Downsampling operations do not change the number of channels/feature maps, they simply reduce the spatial size by the pooling factor. 
</div>
"""


# %% tags=["task"]
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        # TASK: define the layers of the model
        self.conv = nn.Sequential(
            # Add here the convolutional and downsampling layers,
            # as well as the flattening module
        )
        self.dense = nn.Sequential(
            # Add here the fully connected layers
        )
        # END OF TASK

    def forward(self, x):
        y = self.conv(x)
        y = self.dense(y)
        return y


cnn_model = CNNModel()

try:
    cnn_model(torch.zeros(1, 1, 28, 28))
except RuntimeError as e:
    if str(e).startswith("mat1 and mat2 shapes cannot be multiplied"):
        print(
            f"The model does not work correctly with the input shape. Please double check the number of features/channels for the fully connected layers, as well as the `padding` argument of the convolutional layers. The full error is:\n{e}"
        )
    else:
        raise e
print(
    "Trainable params:",
    sum(p.numel() for p in cnn_model.parameters() if p.requires_grad),
)
del cnn_model  # clean up the temporary model


# %% tags=["solution"]
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Define the layers of the model
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
        )
        self.dense = nn.Sequential(
            nn.Linear(in_features=32 * 7 * 7, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10),
        )

    def forward(self, x):
        y = self.conv(x)
        y = self.dense(y)
        return y


cnn_model = CNNModel()

try:
    cnn_model(torch.zeros(1, 1, 28, 28))
except RuntimeError as e:
    if str(e).startswith("mat1 and mat2 shapes cannot be multiplied"):
        print(
            f"The model does not work with the input shape. Please double check the number of features/channels for the fully connected layers. The error is:\n{e}"
        )
    else:
        raise e
print(
    "Trainable params:",
    sum(p.numel() for p in cnn_model.parameters() if p.requires_grad),
)
del cnn_model  # clean up the temporary model

# %% [markdown]
"""
The last line in the previous cell prints the number of trainable parameters of your model. This number should be 110634.

<div class="alert alert-block alert-info">
    <b>Task 4.2</b>: Train the Network
</div>

As we did for Task 3, we will define some auxiliary functions for the training procedure which include, as before, the training loop (which we rewrite to add the computation of a metric to monitor during training), but also a validation procedure which will be used to evaluate the model on the validation dataset on every epoch.

Moreover, these procedures will use the very commonly used `DataLoader` class to deal with the data in batches. This PyTorch module allows an easy interface to iterate over the data in batches and comes with many benefits, such as the potential to load the data quicker with parallel processes.
"""


# %%
def run_epoch(model, optimizer, train_dataloader, loss_fn, device):
    n_samples = len(train_dataloader.dataset)
    total_loss = 0
    total_correct = 0

    # Set the model to training mode
    model.train()
    for X_b, y_b in train_dataloader:
        # Convert the data to PyTorch tensors
        X_b = X_b.to(device)
        y_b = y_b.long().to(
            device
        )  # Ensure the labels are of type long (int) as required by the loss function nn.CrossEntropyLoss

        # Reset the optimizer state
        optimizer.zero_grad()

        # Forward pass: pass the data through the model and retrieve the prediction
        y_pred = model(X_b).squeeze()

        # Compute the loss function with the prediction and the ground truth
        loss = loss_fn(y_pred, y_b)

        # Backward pass: compute the gradient of the loss w.r.t. the parameters
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Accumulate the loss (for monitoring purposes)
        total_loss += loss.item()

        # Compute the number of correct predictions
        total_correct += (y_pred.argmax(dim=1) == y_b).sum().item()
    train_loss = total_loss / n_samples
    train_accuracy = total_correct / n_samples
    return train_loss, train_accuracy


def validate(model, val_dataloader, loss_fn, device):
    total_loss = 0
    total_correct = 0
    n_samples = len(val_dataloader.dataset)

    model.eval()
    with torch.inference_mode():
        for X_b, y_b in val_dataloader:
            X_b = X_b.to(device)
            y_b = y_b.long().to(device)
            y_pred = model(X_b)
            total_loss += loss_fn(y_pred, y_b).item()
            total_correct += (y_pred.argmax(dim=1) == y_b).sum().item()
    val_loss = total_loss / n_samples
    val_accuracy = total_correct / n_samples
    return val_loss, val_accuracy


# %% [markdown]
"""
Below we define a helper function to visualize the training and validation loss and accuracies live during training. We will use it to monitor the training process of the network, and later to discuss some central concepts of ML.
"""

# %%
from IPython.display import clear_output


def live_training_plot(
    train_loss, val_loss, train_acc, val_acc, num_epochs=10, figsize=(10, 5)
):
    clear_output(wait=True)
    plt.close()
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    axs[0].plot(train_loss, label="Training loss")
    axs[0].plot(val_loss, label="Validation loss")

    axs[1].plot(train_acc, label="Training accuracy")
    axs[1].plot(val_acc, label="Validation accuracy")

    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")

    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")

    axs[0].set_title("Loss")
    axs[1].set_title("Accuracy")

    axs[0].set_xlim(0, num_epochs - 1)
    axs[1].set_xlim(0, num_epochs - 1)

    axs[0].legend(loc="upper right")
    axs[1].legend(loc="lower right")

    plt.show()
    return


# %% [markdown]
"""
We are now ready to train the network!

Instantiate and fit your `cnn_model` similar to how you did for the spiral classifier above, but this time:
* use `nn.CrossEntropyLoss` as the loss, with `reduction="sum"`
* use `torch.optim.AdamW` as the optimizer, with learning rate `lr=0.001`
* set a batch size of 128 samples
* train for 10 epochs
"""

# %% tags=["task"]
cnn_model = CNNModel()
cnn_model.to(device)


# TASK: set the optimizer, loss function, batch size and number of epochs
optimizer = None
loss_fn = None
batch_size = None
num_epochs = None
# END OF TASK

assert optimizer is not None, "Please set the optimizer!"
assert loss_fn is not None, "Please set the loss function!"
assert batch_size is not None, "Please set the batch size!"
assert num_epochs is not None, "Please set the number of epochs!"

train_dataloader = torch.utils.data.DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, pin_memory=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_ds, batch_size=batch_size, shuffle=False, pin_memory=True
)

for epoch in (pbar := tqdm(range(num_epochs), desc="Training", total=num_epochs)):
    train_loss, train_acc = run_epoch(
        cnn_model, optimizer, train_dataloader, loss_fn, device
    )
    val_loss, val_acc = validate(cnn_model, val_dataloader, loss_fn, device)
    pbar.set_postfix(
        {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
    )

# %% tags=["solution"]
cnn_model = CNNModel()
cnn_model.to(device)


optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss(reduction="sum")

batch_size = 128
num_epochs = 10

train_dataloader = torch.utils.data.DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, pin_memory=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_ds, batch_size=batch_size, shuffle=False, pin_memory=True
)
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(num_epochs):
    train_loss, train_acc = run_epoch(
        cnn_model, optimizer, train_dataloader, loss_fn, device
    )
    val_loss, val_acc = validate(cnn_model, val_dataloader, loss_fn, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    live_training_plot(
        train_losses, val_losses, train_accs, val_accs, num_epochs, figsize=(10, 5)
    )

# %% [markdown]
"""
Now that we trained our model, let's evaluate its performance on the test dataset.
"""


# %%
def predict(model, test_dataloader, device):
    predictions = np.empty((0,))
    model.eval()
    with torch.inference_mode():
        for X_b, y_b in tqdm(
            test_dataloader, desc="Predicting", total=len(test_dataloader)
        ):
            X_b = X_b.to(device)
            y_b = y_b.long().to(device)
            y_pred = model(X_b).argmax(axis=1).cpu().numpy()
            predictions = np.concatenate((predictions, y_pred), axis=0)
    return predictions


y_test_gt = np.array([y for _, y in test_ds])

test_dataloader = torch.utils.data.DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, pin_memory=True
)
y_test_predicted = predict(cnn_model, test_dataloader, device)

test_acc = accuracy(y_test_predicted, y_test_gt)
print("Testing accuracy =", test_acc)
show_samples(test_ds, "Testing Data", predictions=y_test_predicted, num_samples=10)

# %% [markdown]
"""
<div class="alert alert-block alert-success">
<h2> Checkpoint 4</h2>

You reached the end, congratulations! In this last part, you have been introduced to CNNs as well as trained one on the infamous MNIST dataset for digit classification. 
After 10 epochs, your model should achieve a training, validation, and test accuracy of more than 95%. We will use this checkpoint to discuss why we use training, validation, and testing datasets in practice.

time: 65 working + 20 discussion
</div>
"""
