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
    """Implement your non-linear function here."""
    return a > 0


# %%
def perceptron(x, w, b, f):
    """Implement your perceptron here."""
    return


# %% tags=["solution"]
def perceptron(x, w, b, f):
    """Implement your perceptron here."""
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

The function of an XOR gate can also be understood as a classification problem on $x \in \{0,1\}^2$ and we can think about designing a classifier acting as an XOR gate. It turns out that this problem is not solvable by a single perceptron (https://en.wikipedia.org/wiki/Perceptron) because the set of points $\{(0,0), (0,1), (1,0), (1,1)\}$ is not linearly separable.

![mlp.png](attachment:mlp.png)

Design a two layer perceptron using your `perceptron` function above that implements an XOR Gate on two inputs. Think about the flow of information through this simple network and set the weight values by hand such that the network produces the XOR function.

#### Hint

A single layer in a multilayer perceptron can be described by the equation $y = f(x^\intercal w + b)$ with $f$ a nonlinear function. $b$ is the so called bias (a constant offset vector) and $w$ a vector of weights. Since we are only interested in outputs of `0` or `1`, a good choice for $f$ is the threshold function. Think about which kind of logical operations you can implement with a single perceptron, then see how you can combine them to create an XOR. It might help to write down the equation for a two layer perceptron network.
"""
# %%
