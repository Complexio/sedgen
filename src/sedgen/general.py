import numpy as np


def linear_fit(x, a, b):
    """Linear fit function"""
    y = a * x + b

    return y


def linear2_fit(x, a, b, c):
    """Linear fit with power function"""
    y = a * x ** b + c

    return y


def sigmoid_fit(x, x0, k):
    """Sigmoid fit function"""
    y = 1 / (1 + np.exp(-k * (x - x0)))

    return y


def lognormal_fit(x, a, b):
    """Lognormal fit function"""
    y = a + b * np.log(x)

    return y


# def exponential_fit(x, a, b):
#     """Exponential fit function"""
#     y = a * np.exp(b * x)

#     return y

def reciprocal_fit(x, a, b):
    """Inverse proportional fit function"""
    y = a / x + b

    return y


def reciprocal2_fit(x, a, b):
    """Inverse proportional fit with power function"""
    y = a * x ** b

    return y


def exponential_fit(x, a, c, d):
    """Exponential fit function"""
    y = a * np.exp(c * x) + d

    return y


def simple_exponential_fit(x, a, b):
    """Exponential fit function"""
    y = a * np.exp(b * x)

    return y


def power_law_fit(x, a, b, c):
    """Power law fit function"""
    y = a + b * x ** c

    return y


def power_law_fit_fixed(x, a, b):
    y = a + b * 1 / x

    return y
