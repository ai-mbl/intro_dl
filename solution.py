# %% [markdown]
"""
# Exercise 1: Introduction to Deep Learning
<div>
    <table>
        <tr style="background-color:white">
            <td><img src="attachment:perceptron.png" width="100%"/></td>
            <td><img src="attachment:mlp.png" width="100%"/></td>
            <td><img src="attachment:neural_network.png" width="100%"/></td>
        </tr>
    </table>
</div>

In the following exercise we explore the basic building blocks of deep learning: the perceptron and how to stack multiple perceptrons together into layers to build a neural network.

<div class="alert alert-block alert-danger">
    Set your python kernel to <code>02_intro_dl</code>
</div>

### Acknowledgements

This notebook was created by Albert Dominguez Mantes, Nils Eckstein, Julia Buhmann, and Jan Funke.

<div class="alert alert-danger">
Set your python kernel to <code>02_intro_dl</code>
</div>
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (5, 5)

# %% [markdown]
"""
## Part 1: Perceptrons

<div>
    <img src="attachment:perceptron.png" width="600"/>
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


# %%
def non_linearity(a):
    """Implement your non-linear function here."""
    return


# %% tags=["solution"]
def non_linearity(a):
    return a > 0


# %%
def perceptron(x, w, b, f):
    """Implement your perceptron here."""
    return


# %% tags=["solution"]
def perceptron(x, w, b, f):
    return f(np.sum(x * w) + b)


# %%
def plot_perceptron(w, b, f):

    num_samples = 100
    domain_x1 = (0.0, 1.0)
    domain_x2 = (0.0, 1.0)

    domain = np.meshgrid(
        np.linspace(*domain_x1, num_samples), np.linspace(*domain_x2, num_samples)
    )
    xs = np.array([domain[0].flatten(), domain[1].flatten()]).T

    values = np.array([perceptron(x, w, b, f) for x in xs])

    plt.contourf(domain[0], domain[1], values.reshape(num_samples, num_samples))


# the following should show a linear classifier that is True (shown as green)
# for values below a line starting at (0.1, 0) through (1.0, 0.9)
plot_perceptron(w=[1.0, -1.0], b=-0.1, f=non_linearity)


# %% [markdown]
"""
<div class="alert alert-block alert-success">
<h2> Checkpoint 1 </h2>

We will go over different ways to implement the perceptron together and discuss their efficiency. If you arrived here earlier, feel free to play around with the parameters of the perceptron (the weights and bias) as well as the activation function `f`.
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

The function of an XOR gate can also be understood as a classification problem on $x \in \{0,1\}^2$ and we can think about designing a classifier acting as an XOR gate. It turns out that this problem is not solvable by a single perceptron (https://en.wikipedia.org/wiki/Perceptron) because the set of points $\{(0,0), (0,1), (1,0), (1,1)\}$ is not linearly separable.

![mlp.png](attachment:mlp.png)

Design a two layer perceptron using your `perceptron` function above that implements an XOR Gate on two inputs. Think about the flow of information through this simple network and set the weight values by hand such that the network produces the XOR function.

#### Hint

A single layer in a multilayer perceptron can be described by the equation $y = f(x^\intercal w + b)$ with $f$ a nonlinear function. $b$ is the so called bias (a constant offset vector) and $w$ a vector of weights. Since we are only interested in outputs of `0` or `1`, a good choice for $f$ is the threshold function. Think about which kind of logical operations you can implement with a single perceptron, then see how you can combine them to create an XOR. It might help to write down the equation for a two layer perceptron network.
"""


# %%
def generate_xor_data():
    xs = [np.array([i, j]) for i in [0, 1] for j in [0, 1]]
    ys = [int(np.logical_xor(x[0], x[1])) for x in xs]
    return xs, ys


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
    w11 = [0.0, 0.0]
    b11 = 0.0
    w12 = [0.0, 0.0]
    b12 = 0.0
    w2 = [0.0, 0.0]
    b2 = 0.0
    f = lambda a: a

    # output of the two perceptrons in the first layer
    h1 = perceptron(x, w=w11, b=b11, f=f)
    h2 = perceptron(x, w=w12, b=b12, f=f)
    # output of the perceptron in the last layer
    y = perceptron(np.array([h1, h2]), w=w2, b=b2, f=f)  # h1 AND NOT h2

    return y


# %% tags=["solution"]
def xor(x):
    w11 = [0.1, 0.1]
    b11 = -0.05
    w12 = [0.1, 0.1]
    b12 = -0.15
    w2 = [0.1, -0.1]
    b2 = -0.05
    f = lambda a: a > 0

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
        ), f"xor function returned {xor(x)} for input {x}, but should be {y}"
        print(f"XOR of {x} is {y}, your implementation returns {xor(x)}")
    print("\nCongratulations! You have implemented the XOR function correctly.")


test_xor()

# %% [markdown]
"""
<div class="alert alert-block alert-success">
<h2> Checkpoint 2 </h2>

There are many ways to implement an XOR in a two-layer perceptron. We will review some of them and how we got to them (trial and error or pen and paper?).
    
<br/>
If you arrive here early, think about how to generalize the XOR function to an arbitrary number of inputs. For more than two inputs, the XOR returns True if the number of 1s in the inputs is odd, and False otherwise.

</div>
"""
# %% [markdown]
"""
## Part 2: "Deep" Neural Networks

<div>
    <img src="attachment:neural_network.png" width=500/>
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
        indices = np.random.permutation(len(X))
    else:
        indices = np.arange(len(X))
    for i in range(0, len(X), batch_size):
        yield X[indices[i : i + batch_size]], y[indices[i : i + batch_size]]


def run_epoch(model, optimizer, X_train, y_train, batch_size, loss_fn, device):
    total_loss = 0

    # Set the model to training mode, essential when using certain layers, such as BatchNorm or Dropout
    model.train()
    for X_b, y_b in batch_generator(X_train, y_train, batch_size):
        # Convert the data to PyTorch tensors
        X_b = torch.tensor(X_b, dtype=torch.float32, device=device)
        y_b = torch.tensor(y_b, dtype=torch.float32, device=device)

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
    return total_loss


# %% [markdown]
"""
Let's now write the simple baseline model, consisting of one hidden layer with 12 neurons (or perceptrons). You will see that this baseline model performs pretty poorly. Read the following code snippets and try to understand the involved functions:
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
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_features=2, out_features=12, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=12, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.seq(x)


# Initialize the model, optimizer and set the loss function
bad_model = BaselineModel()
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
    for X_b, y_b in batch_generator(X, y, batch_size, shuffle=False):
        X_b = torch.tensor(X_b, dtype=torch.float32, device=device)
        y_b = torch.tensor(y_b, dtype=torch.float32, device=device)
        y_pred = model(X_b).squeeze().detach().cpu().numpy()
        predictions = np.concatenate((predictions, y_pred), axis=0)
    return np.round(predictions)


def accuracy(y_pred, y_gt):
    return np.sum(y_pred == y_gt) / len(y_gt)


bad_model.eval()  # set the model to evaluation mode, essential when using certain layers, such as BatchNorm or Dropout
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
"""


# %% tags=["task"]
class GoodModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            # Add your layers and activation functions here
        )

    def forward(self, x):
        return self.seq(x)


# Initialize the model
good_model = GoodModel()
good_model.to(device)

# Set the optimizer and the loss function
optimizer = None  # Set the optimizer instance here
loss_fn = None  # Set the loss function instance here


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

# Set the optimizer and the loss function
optimizer = torch.optim.AdamW(good_model.parameters(), lr=0.001)
loss_fn = nn.BCELoss(reduction="sum")


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
good_predictions = predict(good_model, X_test, y_test, batch_size)
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

Looking at the classifier on an extended domain, what observations can you make?
"""


# %% tags=["task"]
def plot_classifiers(classifier_1, classifier_2):

    plt.subplots(1, 2, figsize=(10, 5))

    num_samples = 200
    # change the plotted domain here
    domain_x1 = (0.0, 1.0)
    domain_x2 = (0.0, 1.0)

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
<div class="alert alert-block alert-success">
<h2> Checkpoint 3</h2>

Let us know in the exercise channel when you got here and what accuracy your model achieved. We will compare different solutions and discuss why some of them are better than others. We will also discuss the generalization behaviour of the classifier outside of the domain it was trained on.

Time: 60 working + 15 discussion
</div>
"""
